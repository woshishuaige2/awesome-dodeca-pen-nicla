import multiprocessing as mp
import queue
import time

import numpy as np

from app.monitor_ble import (
    StopCommand,
    StylusReading,
    align_reading_to_host_time,
    calc_quat,
)


def list_serial_ports() -> list[str]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []

    return [port.device for port in list_ports.comports()]


def open_usb_serial(port: str, baud: int = 115200):
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is required for USB IMU transport. "
            "Install Code/IMU/requirements.txt or run: pip install pyserial"
        ) from exc

    return serial.Serial(port, baudrate=baud, timeout=0.1)


def _parse_quaternion_line(line: str) -> StylusReading | None:
    """
    Parse USB serial packets emitted by the Nicla firmware:
    Q,<seq>,<timestamp_ms>,<qw>,<qx>,<qy>,<qz>,<pressure>
    """
    parts = line.strip().split(",")
    if len(parts) != 8 or parts[0] != "Q":
        return None

    try:
        seq = int(parts[1])
        timestamp_ms = int(parts[2])
        quat_raw = np.array([int(value) for value in parts[3:7]], dtype=np.float64)
        pressure = int(parts[7]) / 2**16
    except ValueError:
        return None

    return StylusReading(
        accel=None,
        gyro=None,
        mag=None,
        quat=calc_quat(quat_raw),
        t=timestamp_ms,
        pressure=pressure,
        seq=seq,
    )


def monitor_usb_serial(
    data_queue: mp.Queue,
    command_queue: mp.Queue,
    serial_port,
):
    ser = serial_port
    ignored_lines = 0
    packet_count = 0
    last_seq = None
    last_timestamp_ms = None
    try:
        while True:
            try:
                command = command_queue.get_nowait()
                if isinstance(command, StopCommand):
                    print("Quitting USB serial IMU process")
                    return
            except queue.Empty:
                pass

            raw_line = ser.readline()
            if not raw_line:
                continue

            t_system_arrival = time.monotonic()
            try:
                line = raw_line.decode("ascii", errors="ignore")
                reading = _parse_quaternion_line(line)
                if reading is None:
                    if line.startswith("[Diag]"):
                        continue
                    ignored_lines += 1
                    if ignored_lines <= 5:
                        print(f"[USB IMU] Ignoring non-packet line: {line.strip()!r}")
                    elif ignored_lines == 6:
                        print("[USB IMU] Further non-packet lines suppressed.")
                    continue
                reading = align_reading_to_host_time(reading, t_system_arrival)
                packet_count += 1
                parts = line.strip().split(",")
                seq = int(parts[1])
                timestamp_ms = int(parts[2])
                if packet_count <= 5:
                    print(f"[USB IMU] Packet {packet_count}: {line.strip()!r}")
                if last_seq is not None:
                    seq_gap = (seq - last_seq) % 65536
                    timestamp_gap = timestamp_ms - last_timestamp_ms
                    if seq_gap != 1 or timestamp_gap > 100:
                        print(
                            "[USB IMU] Packet gap: "
                            f"seq {last_seq}->{seq} (gap {seq_gap}), "
                            f"timestamp dt={timestamp_gap} ms"
                        )
                last_seq = seq
                last_timestamp_ms = timestamp_ms
                data_queue.put(reading)
            except Exception as exc:
                print(f"Error in USB serial IMU handler: {exc}")
    finally:
        ser.close()
