import struct
from typing import NamedTuple
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
import asyncio
import multiprocessing as mp
import time

import numpy as np


class StylusReading(NamedTuple):
    accel: np.ndarray | None
    gyro: np.ndarray | None
    mag: np.ndarray | None
    quat: np.ndarray | None
    t: int  # Sensor timestamp in ms
    pressure: float
    aligned_host_time: float | None = None

    def format_aligned(self):
        if self.quat is not None:
            return f"p={self.pressure:<8.5f}: q={self.quat}, t={self.t}, host={self.aligned_host_time}"
        mag_text = "None" if self.mag is None else f"{self.mag}"
        accel_norm = float(np.linalg.norm(self.accel)) if self.accel is not None else 0.0
        return f"p={self.pressure:<8.5f}: |a|={accel_norm:<7.3f} a={self.accel}, g={self.gyro}, m={mag_text}, host={self.aligned_host_time}"
    
    def to_json(self):
        payload = {
            "t": self.t,
            "pressure": self.pressure,
            "aligned_host_time": self.aligned_host_time,
        }
        if self.quat is not None:
            payload["quat"] = self.quat.tolist()
        if self.accel is not None:
            payload["accel"] = self.accel.tolist()
        if self.gyro is not None:
            payload["gyro"] = self.gyro.tolist()
        if self.mag is not None:
            payload["mag"] = self.mag.tolist()
        return payload
    
    @staticmethod
    def from_json(payload):
        mag = payload.get("mag")
        accel = payload.get("accel")
        gyro = payload.get("gyro")
        quat = payload.get("quat")
        return StylusReading(
            None if accel is None else np.array(accel),
            None if gyro is None else np.array(gyro),
            None if mag is None else np.array(mag),
            None if quat is None else np.array(quat),
            payload["t"],
            payload["pressure"],
            payload.get("aligned_host_time"),
        )


class StopCommand(NamedTuple):
    pass


def calc_accel(a):
    """Remap raw accelerometer measurements to Gs."""
    accel_range = 4  # Should match settings.accelRange in microcontroller code
    return a * 0.061 * (accel_range / 2) / 1000


def calc_gyro(g):
    """Remap raw gyro measurements to degrees per second."""
    gyro_range = 500  # Should match settings.gyroRange in microcontroller code
    return g * 4.375 * (gyro_range / 125) / 1000


def calc_mag(m):
    """Magnetometer values are stored in raw sensor units until calibrated."""
    return m.astype(np.float64)


def calc_quat(q):
    """Convert packed int16 quaternion components back to float and normalize."""
    quat = q.astype(np.float64) / 32767.0
    norm = np.linalg.norm(quat)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def unpack_imu_data_packet(data: bytearray, t_system_fallback: float = 0.0):
    """Unpacks an IMUDataPacket struct from the given data buffer."""
    # Quaternion format: int16_t quat_wxyz[4], uint16_t pressure,
    # uint16_t reserved, uint32_t timestamp (16 bytes)
    # Legacy format: int16_t accel[3], gyro[3], uint16_t pressure (14 bytes)
    # Master clock format: int16_t accel[3], gyro[3], uint16_t pressure, uint32_t timestamp (18 bytes)
    # Nicla mag format: int16_t accel[3], gyro[3], mag[3], uint16_t pressure (20 bytes)
    # Nicla master clock format: int16_t accel[3], gyro[3], mag[3], uint16_t pressure, uint32_t timestamp (24 bytes)
    
    quat = None
    mag = None
    accel = None
    gyro = None
    if len(data) == 16:
        qw, qx, qy, qz, pressure, _reserved, t_sensor = struct.unpack("<4hHHI", data)
        quat = calc_quat(np.array([qw, qx, qy, qz], dtype=np.float64))
    elif len(data) == 24:
        ax, ay, az, gx, gy, gz, mx, my, mz, pressure, t_sensor = struct.unpack("<9hHI", data)
        mag = calc_mag(np.array([mx, my, mz], dtype=np.float64))
    elif len(data) == 20:
        ax, ay, az, gx, gy, gz, mx, my, mz, pressure = struct.unpack("<9hH", data)
        t_sensor = int(t_system_fallback * 1000)
        mag = calc_mag(np.array([mx, my, mz], dtype=np.float64))
    elif len(data) == 18:
        # Master Clock format
        ax, ay, az, gx, gy, gz, pressure, t_sensor = struct.unpack("<3h3hHI", data)
    elif len(data) == 14:
        # Legacy format - use fallback system time converted to ms
        ax, ay, az, gx, gy, gz, pressure = struct.unpack("<3h3hH", data)
        t_sensor = int(t_system_fallback * 1000)
    else:
        raise ValueError(f"Unexpected packet size: {len(data)} bytes")

    if quat is None:
        accel = calc_accel(np.array([ax, ay, az], dtype=np.float64) * 9.8)
        gyro = calc_gyro(np.array([gx, gy, gz], dtype=np.float64) * np.pi / 180.0)
    return StylusReading(accel, gyro, mag, quat, t_sensor, pressure / 2**16)


# Global synchronization variables
sync_offset = None  # t_host - t_sensor (in seconds)
sync_lock = mp.Lock()
offset_samples: list[float] = []
OFFSET_SAMPLE_COUNT = 50


def get_sync_offset():
    global sync_offset
    return sync_offset


def align_reading_to_host_time(reading: StylusReading, t_host_arrival: float) -> StylusReading:
    global sync_offset
    t_sensor_sec = reading.t / 1000.0
    aligned_host_time = None

    with sync_lock:
        if sync_offset is None:
            offset_samples.append(t_host_arrival - t_sensor_sec)
            if len(offset_samples) >= OFFSET_SAMPLE_COUNT:
                sync_offset = float(np.median(np.array(offset_samples, dtype=np.float64)))
                print("\n[Sync] Median host/device offset established:")
                print(f"[Sync] samples: {len(offset_samples)}")
                print(f"[Sync] offset (t_host - t_imu): {sync_offset:.6f}\n")

        if sync_offset is not None:
            aligned_host_time = t_sensor_sec + sync_offset

    return reading._replace(aligned_host_time=aligned_host_time)


characteristic = "19B10013-E8F2-537E-4F6C-D104768A1214"


async def monitor_ble_async(data_queue: mp.Queue, command_queue: mp.Queue):
    while True:
        device = await BleakScanner.find_device_by_name("DPOINT", timeout=5)
        if device is None:
            print("could not find device with name DPOINT. Retrying in 1 second...")
            await asyncio.sleep(1)
            continue

        def queue_notification_handler(_: BleakGATTCharacteristic, data: bytearray):
            # Capture system time immediately upon packet arrival
            t_system_arrival = time.monotonic()
            
            try:
                # Pass arrival time for legacy fallback
                reading = unpack_imu_data_packet(data, t_system_fallback=t_system_arrival)
                reading = align_reading_to_host_time(reading, t_system_arrival)
                data_queue.put(reading)
            except Exception as e:
                print(f"Error in notification handler: {e}")

        disconnected_event = asyncio.Event()
        print("Connecting to BLE device...")
        try:
            async with BleakClient(
                device, disconnected_callback=lambda _: disconnected_event.set()
            ) as client:
                print("Connected!")
                await client.start_notify(characteristic, queue_notification_handler)
                command = asyncio.create_task(
                    asyncio.to_thread(lambda: command_queue.get())
                )
                disconnected_task = asyncio.create_task(disconnected_event.wait())
                await asyncio.wait(
                    {disconnected_task, command}, return_when=asyncio.FIRST_COMPLETED
                )
                if command.done():
                    print("Quitting BLE process")
                    return
                print("Disconnected from BLE")
        except Exception as e:
            print(f"BLE Exception: {e}")
            print("Retrying in 1 second...")
            await asyncio.sleep(1)


def monitor_ble(data_queue: mp.Queue, command_queue: mp.Queue):
    asyncio.run(monitor_ble_async(data_queue, command_queue))
