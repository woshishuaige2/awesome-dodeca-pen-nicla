from monitor_ble import monitor_ble, live_plot
import multiprocessing as mp
from pynput import keyboard
import matplotlib.pyplot as plt
from time import sleep, time

# Global variables
ble_process = None
ble_queue = mp.Queue()
ble_command_queue = mp.Queue()

plot_process = None

# Initialize data queues
phi_queue = mp.Queue() 
theta_queue = mp.Queue()
    
def start_ble_process():
    global ble_process
    global plot_process
    
    ble_process = mp.Process(
        target=monitor_ble, args=(ble_queue, ble_command_queue, phi_queue, theta_queue), daemon=False
    )
    ble_process.start()
    print("BLE process started.")
    
    plot_process = mp.Process(target=live_plot, args=(phi_queue, theta_queue))
    plot_process.start()
    print("Plot process started.")

def stop_processes():
    """Terminate both processes and wait for them to finish."""
    global ble_process
    global plot_process

    if plot_process:
        print("Terminating plot process...")
        plot_process.terminate()
        plot_process.join()
        print("Plot process terminated.")

    if ble_process:
        print("Terminating BLE process...")
        ble_process.terminate()
        ble_process.join()
        print("BLE process terminated.")
        
def on_press(key):
    try:
        if key.char == 'q':
            print("'q' key pressed. Terminating BLE process...")
            stop_processes()
            
            return False  # Stop the listener
    except AttributeError:
        pass  # Ignore special keys

def main():
    start_ble_process()  # Start BLE process at program launch
    
    # Start keyboard listener in a separate thread
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("Program running. Press 'q' to terminate BLE process.")
    listener.join()  # Wait for the listener to complete
    
    stop_processes()

if __name__ == '__main__':
    main()
