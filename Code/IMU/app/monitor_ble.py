import struct
from typing import NamedTuple
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
import asyncio
import multiprocessing as mp

import numpy as np


class StylusReading(NamedTuple):
    accel: np.ndarray
    gyro: np.ndarray
    mag: np.ndarray | None
    t: int  # Sensor timestamp in ms
    pressure: float

    def format_aligned(self):
        mag_text = "None" if self.mag is None else f"{self.mag}"
        return (
            f"p={self.pressure:<8.5f}: |a|={np.linalg.norm(self.accel):<7.3f} "
            f"a={self.accel}, g={self.gyro}, m={mag_text}"
        )
    
    def to_json(self):
        return {
            "accel": self.accel.tolist(),
            "gyro": self.gyro.tolist(),
            "mag": None if self.mag is None else self.mag.tolist(),
            "t": self.t,
            "pressure": self.pressure
        }
    
    def from_json(dict):
        mag = dict.get("mag")
        return StylusReading(
            np.array(dict["accel"]),
            np.array(dict["gyro"]),
            None if mag is None else np.array(mag),
            dict["t"],
            dict["pressure"]
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


def unpack_imu_data_packet(data: bytearray, t_system_fallback: float = 0.0):
    """Unpacks an IMUDataPacket struct from the given data buffer."""
    # Legacy format: int16_t accel[3], gyro[3], uint16_t pressure (14 bytes)
    # Master clock format: int16_t accel[3], gyro[3], uint16_t pressure, uint32_t timestamp (18 bytes)
    # Nicla mag format: int16_t accel[3], gyro[3], mag[3], uint16_t pressure (20 bytes)
    # Nicla master clock format: int16_t accel[3], gyro[3], mag[3], uint16_t pressure, uint32_t timestamp (24 bytes)
    
    mag = None
    if len(data) == 24:
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

    accel = calc_accel(np.array([ax, ay, az], dtype=np.float64) * 9.8)
    gyro = calc_gyro(np.array([gx, gy, gz], dtype=np.float64) * np.pi / 180.0)
    return StylusReading(accel, gyro, mag, t_sensor, pressure / 2**16)


# Global synchronization variables
sync_offset = None  # t_sensor - t_system (in seconds)
sync_lock = mp.Lock()


def get_sync_offset():
    global sync_offset
    return sync_offset


characteristic = "19B10013-E8F2-537E-4F6C-D104768A1214"


async def monitor_ble_async(data_queue: mp.Queue, command_queue: mp.Queue):
    while True:
        device = await BleakScanner.find_device_by_name("DPOINT", timeout=5)
        if device is None:
            print("could not find device with name DPOINT. Retrying in 1 second...")
            await asyncio.sleep(1)
            continue

        def queue_notification_handler(_: BleakGATTCharacteristic, data: bytearray):
            global sync_offset
            import time
            
            # Capture system time immediately upon packet arrival
            t_system_arrival = time.monotonic()
            
            try:
                # Pass arrival time for legacy fallback
                reading = unpack_imu_data_packet(data, t_system_fallback=t_system_arrival)
                
                # Establish master clock offset on first valid packet.
                # Timestamped packets are 18 bytes for the legacy format and
                # 24 bytes for the Nicla mag-enabled format.
                if sync_offset is None and len(data) in (18, 24):
                    with sync_lock:
                        if sync_offset is None:
                            t_sensor_sec = reading.t / 1000.0
                            sync_offset = t_sensor_sec - t_system_arrival
                            print(f"\n[Sync] Master Clock Established:")
                            print(f"[Sync] t_system_sync: {t_system_arrival:.6f}")
                            print(f"[Sync] t_sensor_sync: {t_sensor_sec:.6f}")
                            print(f"[Sync] Offset (t_sensor - t_system): {sync_offset:.6f}\n")
                
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
