import json
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# Constants for Kalman Filter (matching monitor_ble.py)
C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
P = np.eye(4)
Q = np.eye(4) * 0.001 # Process noise
R = np.eye(2) * 0.1   # Measurement noise

# Gyro bias correction (rad/s) - tuned from stationary calibration tests
GZ_BIAS = -0.018  # Corrects yaw drift observed when pen is stationary

def process_imu_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    readings = data['imu_readings']
    
    if not readings or len(readings) == 0:
        print(f"Error: No IMU readings found in {json_path}")
        print(f"The imu_readings array is empty. Please record IMU data first.")
        return None
    
    # State: [phi_hat, phi_dot, theta_hat, theta_dot]
    state_estimate = np.array([[0.0], [0.0], [0.0], [0.0]])
    global P
    
    phi_hat = 0.0
    theta_hat = 0.0
    yaw = 0.0 # Simple integration for yaw since we don't have magnetometer
    
    results = {
        'timestamps': [],
        'phi_degrees': [],
        'theta_degrees': [],
        'yaw_degrees': [],
        'accel_x': [],
        'accel_y': [],
        'accel_z': []
    }
    
    last_time = None
    start_ts = readings[0]['local_timestamp']
    
    for i, r in enumerate(readings):
        current_time = r['local_timestamp']
        if last_time is None:
            dt = 0.01 
        else:
            dt = current_time - last_time
        
        last_time = current_time
        
        ax, ay, az = r['accel']
        gx, gy, gz = r['gyro'] # Based on monitor_ble.py, these are rad/s
        
        # Apply gyro bias correction
        gz = gz - GZ_BIAS
        
        # Accelerometer angles
        phi_acc = math.atan2(ay, math.sqrt(ax ** 2.0 + az ** 2.0))
        theta_acc = math.atan2(-ax, math.sqrt(ay ** 2.0 + az ** 2.0))
        
        # Gyro rates to Euler angle derivatives
        phi_dot = gx + math.sin(phi_hat) * math.tan(theta_hat) * gy + math.cos(phi_hat) * math.tan(theta_hat) * gz
        theta_dot = math.cos(phi_hat) * gy - math.sin(phi_hat) * gz
        
        # Kalman Filter Predict
        A = np.array([[1, -dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, -dt], [0, 0, 0, 1]])
        B = np.array([[dt, 0], [0, 0], [0, dt], [0, 0]])
        
        gyro_input = np.array([[phi_dot], [theta_dot]])
        state_estimate = A.dot(state_estimate) + B.dot(gyro_input)
        P = A.dot(P.dot(np.transpose(A))) + Q
        
        # Kalman Filter Update
        measurement = np.array([[phi_acc], [theta_acc]])
        y_tilde = measurement - C.dot(state_estimate)
        S = R + C.dot(P.dot(np.transpose(C)))
        K = P.dot(np.transpose(C).dot(np.linalg.inv(S)))
        state_estimate = state_estimate + K.dot(y_tilde)
        P = (np.eye(4) - K.dot(C)).dot(P)
        
        phi_hat = state_estimate[0, 0]
        theta_hat = state_estimate[2, 0]
        
        # Yaw integration (simplified)
        # psi_dot = sin(phi)/cos(theta) * q + cos(phi)/cos(theta) * r
        if abs(math.cos(theta_hat)) > 0.1:
            yaw_dot = (math.sin(phi_hat) / math.cos(theta_hat)) * gy + (math.cos(phi_hat) / math.cos(theta_hat)) * gz
            yaw += yaw_dot * dt
        
        results['timestamps'].append(current_time - start_ts)
        results['phi_degrees'].append(phi_hat * 180.0 / math.pi)
        results['theta_degrees'].append(theta_hat * 180.0 / math.pi)
        results['yaw_degrees'].append(yaw * 180.0 / math.pi)
        results['accel_x'].append(ax)
        results['accel_y'].append(ay)
        results['accel_z'].append(az)
        
    return results

def plot_results(results, output_path):
    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    
    axes[0].plot(results['timestamps'], results['phi_degrees'], label='Roll (Phi)', color='#1f77b4', linewidth=1.5)
    axes[0].set_title('Roll Angle (Phi)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Degrees', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend(loc='upper right')
    
    axes[1].plot(results['timestamps'], results['theta_degrees'], label='Pitch (Theta)', color='#d62728', linewidth=1.5)
    axes[1].set_title('Pitch Angle (Theta)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Degrees', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend(loc='upper right')

    axes[2].plot(results['timestamps'], results['yaw_degrees'], label='Yaw (Psi)', color='#2ca02c', linewidth=1.5)
    axes[2].set_title('Yaw Angle (Psi) - Integrated', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Degrees', fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].legend(loc='upper right')
    
    axes[3].plot(results['timestamps'], results['accel_x'], label='Accel X', color='#ff7f0e', linewidth=1.5)
    axes[3].plot(results['timestamps'], results['accel_y'], label='Accel Y', color='#9467bd', linewidth=1.5)
    axes[3].plot(results['timestamps'], results['accel_z'], label='Accel Z', color='#8c564b', linewidth=1.5)
    axes[3].set_title('Acceleration Components', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Time (s)', fontsize=12)
    axes[3].set_ylabel('Acceleration (m/s²)', fontsize=12)
    axes[3].grid(True, linestyle='--', alpha=0.7)
    axes[3].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Professional plot saved to {output_path}")

if __name__ == "__main__":
    # Get the script directory and construct relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_json = os.path.join(script_dir, "..", "IMU", "outputs", "imu_data.json")
    output_img = os.path.join(script_dir, "imu_plots_v2.png")
    
    if os.path.exists(input_json):
        print(f"Processing {input_json}...")
        res = process_imu_data(input_json)
        if res is not None:
            plot_results(res, output_img)
    else:
        print(f"Error: {input_json} not found.")
