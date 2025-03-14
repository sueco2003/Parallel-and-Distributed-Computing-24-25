import matplotlib.pyplot as plt
import numpy as np

# Data - easy to modify
num_threads = [1, 2, 4, 6]  # Number of threads
# Times in seconds - replace these with your actual measurements
times_list = [
    [115.9, 61.2, 33.7, 33.4],  # Execution time for each thread count for program 1
    [65.9, 38.8, 27.5, 31.1],  # Execution time for each thread count for program 2
    [134.8, 67.7, 36.2, 29.0],     # Execution time for each thread count for program 3
]
serial_times = [117.8, 64.3, 134.9]  # Time for serial version (non-threaded) for each program

for i, times in enumerate(times_list):
    serial_time = serial_times[i]
    # Calculate speedup
    speedup = [serial_time / t for t in times]

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Time vs Number of Threads
    ax1.plot(num_threads, times, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.axhline(y=serial_time, color='r', linestyle='--', label='Serial Time')
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Execution Time vs Number of Threads')
    ax1.grid(True)
    ax1.set_xticks(num_threads)
    ax1.set_xticks([1, 2, 3, 4, 5, 6])
    ax1.legend()

    # Subplot 2: Speedup vs Number of Threads
    ax2.plot(num_threads, speedup, 'o-', linewidth=2, markersize=8, color='green')
    ax2.axhline(y=1, color='r', linestyle='--', label='Serial (baseline)')
    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Speedup (Serial Time / Parallel Time)')
    ax2.set_title('Speedup vs Number of Threads')
    ax2.grid(True)
    ax2.set_xticks(num_threads)
    ax2.set_xticks([1, 2, 3, 4, 5, 6])
    ax2.legend()

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plt.savefig(f'figures/times_speedup_{i+1}.png')
