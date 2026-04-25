import struct
from typing import NamedTuple
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
import asyncio
import multiprocessing as mp
import math
from math import sin, cos, tan
from time import sleep, time
import matplotlib.pyplot as plt

import numpy as np

class StylusReading(NamedTuple):
    accel: np.ndarray
    gyro: np.ndarray
    mag: np.ndarray | None
    t: int
    pressure: float

    def format_aligned(self):
        return f"p={self.pressure:<8.5f}: |a|={np.linalg.norm(self.accel):<7.3f} a={self.accel}, g={self.gyro}"
    
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
    return m.astype(np.float64)

def unpack_imu_data_packet(data: bytearray):
    """Unpacks an IMUDataPacket struct from the given data buffer."""
    mag = None
    t_sensor = 0

    if len(data) == 24:
        ax, ay, az, gx, gy, gz, mx, my, mz, pressure, t_sensor = struct.unpack("<9hHI", data)
        mag = calc_mag(np.array([mx, my, mz], dtype=np.float64))
    elif len(data) == 20:
        ax, ay, az, gx, gy, gz, mx, my, mz, pressure = struct.unpack("<9hH", data)
        mag = calc_mag(np.array([mx, my, mz], dtype=np.float64))
    else:
        # Legacy packet layout from the old 6-axis board.
        ay, ax, az, gy, gx, gz, pressure = struct.unpack("<3h3hH", data)
        ay = -ay
        az = -az
        gx = -gx
        ax, ay, az, gx, gy, gz = ax, ay, az, gx, gy, gz

    accel = calc_accel(np.array([ax, ay, az], dtype=np.float64) * 9.8)
    gyro = calc_gyro(np.array([gx, gy, gz], dtype=np.float64) * np.pi / 180.0)

    print(accel)

    return StylusReading(accel, gyro, mag, t_sensor, pressure / 2**16)


characteristic = "19B10013-E8F2-537E-4F6C-D104768A1214"

sleep_time = 0.01

# Initialise matrices and variables
C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
P = np.eye(4)
Q = np.eye(4)
R = np.eye(2)

state_estimate = np.array([[0], [0], [0], [0]]) # [phi_hat, phi_dot, theta_hat, theta_dot]

phi_hat = 0.0
theta_hat = 0.0
yaw_hat = 0.0

# Temporary gyro x/y bias correction (rad/s), visually tuned from live plots.
# Fine-tune these values with a short stationary capture.
GX_BIAS = 0.0015
GY_BIAS = -0.0050

# Calculate accelerometer offsets
N = 100
phi_offset = 0.0
theta_offset = 0.0

for i in range(N):
    [phi_acc, theta_acc] = [0,0] 
    phi_offset += phi_acc
    theta_offset += theta_acc
    sleep(sleep_time)

phi_offset = float(phi_offset) / float(N)
theta_offset = float(theta_offset) / float(N)

sleep(2)

# Measured sampling time
dt = 0.0
start_time = time()

counter = 0
async def monitor_ble_async(
    data_queue: mp.Queue,
    command_queue: mp.Queue,
    phi_queue: mp.Queue,
    theta_queue: mp.Queue,
    yaw_queue: mp.Queue,
):    
    while True:
        device = await BleakScanner.find_device_by_name("DPOINT", timeout=5)
        if device is None:
            print("could not find device with name DPOINT. Retrying in 1 second...")
            await asyncio.sleep(1)
            continue
        
        def queue_notification_handler(_: BleakGATTCharacteristic, data: bytearray):
            reading = unpack_imu_data_packet(data)
            data_queue.put(reading)

            # Sampling time
            global start_time
            global phi_hat
            global theta_hat
            global yaw_hat
            global state_estimate
            global P
            global counter          

            dt = time() - start_time
            start_time = time()

            ax = reading.accel[0]
            ay = reading.accel[1]
            az = reading.accel[2]
            p = reading.gyro[0] - GX_BIAS
            q = reading.gyro[1] - GY_BIAS
            r = reading.gyro[2]
            # Get accelerometer measurements and remove offsets
            [phi_acc, theta_acc] = [math.atan2(ay, math.sqrt(ax ** 2.0 + az ** 2.0)), math.atan2(-ax, math.sqrt(ay ** 2.0 + az ** 2.0))] 
            phi_acc -= phi_offset
            theta_acc -= theta_offset
            
            # Get gyro measurements and calculate Euler angle derivatives gx, gy, gz
            phi_dot = p + math.sin(phi_hat) * math.tan(theta_hat) * q + math.cos(phi_hat) * math.tan(theta_hat) * r
            theta_dot = math.cos(phi_hat) * q - math.sin(phi_hat) * r

            # Kalman filter
            A = np.array([[1, -dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, -dt], [0, 0, 0, 1]])
            B = np.array([[dt, 0], [0, 0], [0, dt], [0, 0]])

            gyro_input = np.array([[phi_dot], [theta_dot]])
            state_estimate = A.dot(state_estimate) + B.dot(gyro_input)
            P = A.dot(P.dot(np.transpose(A))) + Q

            measurement = np.array([[phi_acc], [theta_acc]])
            y_tilde = measurement - C.dot(state_estimate)
            S = R + C.dot(P.dot(np.transpose(C)))
            K = P.dot(np.transpose(C).dot(np.linalg.inv(S)))
            state_estimate = state_estimate + K.dot(y_tilde)
            P = (np.eye(4) - K.dot(C)).dot(P)

            phi_hat = float(state_estimate[0, 0])
            theta_hat = float(state_estimate[2, 0])

            # Approximate yaw by integrating the body rates using the current
            # roll/pitch estimate. Without a magnetometer this will drift over time.
            cos_theta = math.cos(theta_hat)
            if abs(cos_theta) > 0.1:
                yaw_dot = (
                    math.sin(phi_hat) / cos_theta * q
                    + math.cos(phi_hat) / cos_theta * r
                )
                yaw_hat += yaw_dot * dt
            
            phi_degrees = np.round(phi_hat * 180.0 / math.pi, 2)
            theta_degrees = np.round(theta_hat * 180.0 / math.pi, 2)
            yaw_degrees = np.round(yaw_hat * 180.0 / math.pi, 2)
            
            # Enqueue the data for plotting
            if counter%2 == 0:
                phi_queue.put(phi_degrees)
                theta_queue.put(theta_degrees)
                yaw_queue.put(yaw_degrees)
            counter += 1

            # Display results
            # print("Phi: " + str(phi_degrees) + " Theta: " + str(theta_degrees))

            sleep(sleep_time)

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
                    [disconnected_task, command], return_when=asyncio.FIRST_COMPLETED
                )
                if command.done():
                    print("Quitting BLE process")
                    return
                print("Disconnected from BLE")
        except Exception as e:
            print(f"BLE Exception: {e}")
            print("Retrying in 1 second...")
            await asyncio.sleep(1)


def monitor_ble(
    data_queue: mp.Queue,
    command_queue: mp.Queue,
    phi_queue: mp.Queue,
    theta_queue: mp.Queue,
    yaw_queue: mp.Queue,
):
    asyncio.run(
        monitor_ble_async(data_queue, command_queue, phi_queue, theta_queue, yaw_queue)
    )
    
# Function to update the plot
def live_plot(phi_queue, theta_queue, yaw_queue):
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    x_data, phi_data, theta_data, yaw_data = [], [], [], []
    line1, = ax1.plot(x_data, phi_data, label="Roll (Phi)", color="blue")
    line2, = ax2.plot(x_data, theta_data, label="Pitch (Theta)", color="red")
    line3, = ax3.plot(x_data, yaw_data, label="Yaw (Psi)", color="green")
    
    ax1.set_title("Roll (Phi) vs Time")
    ax1.set_ylabel("Phi (degrees)")
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title("Pitch (Theta) vs Time")
    ax2.set_ylabel("Theta (degrees)")
    ax2.legend()
    ax2.grid(True)

    ax3.set_title("Yaw (Psi) vs Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Yaw (degrees)")
    ax3.legend()
    ax3.grid(True)
    
    start_time = time()
    
    while True:
        try:
            current_time = time() - start_time
            if (
                not phi_queue.empty()
                and not theta_queue.empty()
                and not yaw_queue.empty()
            ):
                phi = phi_queue.get()
                theta = theta_queue.get()
                yaw = yaw_queue.get()
                
                x_data.append(current_time)
                phi_data.append(phi)
                theta_data.append(theta)
                yaw_data.append(yaw)
                
                line1.set_xdata(x_data)
                line1.set_ydata(phi_data)
                line2.set_xdata(x_data)
                line2.set_ydata(theta_data)
                line3.set_xdata(x_data)
                line3.set_ydata(yaw_data)
                
                ax1.set_xlim(0, max(current_time, 10))
                ax2.set_xlim(0, max(current_time, 10))
                ax3.set_xlim(0, max(current_time, 10))
                
                ax1.set_ylim(min(phi_data)-5, max(phi_data)+5)
                ax2.set_ylim(min(theta_data)-5, max(theta_data)+5)
                ax3.set_ylim(min(yaw_data)-5, max(yaw_data)+5)
                
                plt.pause(0.01)
        except KeyboardInterrupt:
            print("Stopping live plot...")
            plt.close(fig)
            break
