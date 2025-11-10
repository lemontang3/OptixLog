#!/usr/bin/env python3
"""
OptixLog Complete Tutorial

This script demonstrates all the key features of OptixLog using simple Python loops - 
no complex dependencies like Meep required!

## What You'll Learn
1. **Basic Metrics Logging** - Log scalar values and track them over time
2. **Image Logging** - Upload PNG/JPG images from your experiments
3. **CSV Data Logging** - Upload CSV files with your data
4. **MP4 Video Logging** - Create and upload animated videos
5. **Comparing Multiple Runs** - Organize and compare different experiments
6. **Multi-Metric Tracking** - Track multiple metrics simultaneously with proper visualization
"""

# ============================================================================
# Setup and Installation
# ============================================================================

# Install optixlog (if not already installed)
# pip install ../Coupler/web/public/optixlog-0.0.2-py3-none-any.whl

# Import required libraries
import optixlog
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
import csv
from io import BytesIO

# Set your OptixLog credentials
API_KEY = os.getenv("s")  # Replace with your actual API key
API_URL = os.getenv("OPTIX_API_URL", "http://coupler.onrender.com")
PROJECT_NAME = "Tutorial"

print("✓ Imports successful!")
print(f"API URL: {API_URL}")
print(f"Project: {PROJECT_NAME}")

# ============================================================================
# 1. Basic Metrics Logging
# ============================================================================

print("\n" + "=" * 60)
print("1. Basic Metrics Logging")
print("=" * 60)

# Initialize OptixLog client
client = optixlog.init(
    api_key=API_KEY,
    api_url=API_URL,
    project=PROJECT_NAME,
    run_name="basic_metrics_example",
    create_project_if_not_exists=True,
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "optimizer": "adam"
    }
)

# Simulate a training loop with random metrics
print("Simulating training with metrics logging...")
for step in range(50):
    # Simulate some computation
    time.sleep(0.05)
    
    # Generate fake metrics (in real use, these would come from your simulation/training)
    loss = 1.0 / (step + 1) + np.random.normal(0, 0.01)
    accuracy = 1 - np.exp(-step/10) + np.random.normal(0, 0.02)
    validation_loss = loss * 1.2 + np.random.normal(0, 0.015)
    
    # Log metrics to OptixLog
    client.log(
        step=step,
        loss=loss,
        accuracy=accuracy,
        validation_loss=validation_loss
    )
    
    if step % 10 == 0:
        print(f"Step {step}: loss={loss:.4f}, accuracy={accuracy:.4f}")

print("✓ Metrics logged successfully!")

# ============================================================================
# 2. Image Logging
# ============================================================================

print("\n" + "=" * 60)
print("2. Image Logging")
print("=" * 60)

# Create a new run for image logging
client = optixlog.init(
    api_key=API_KEY,
    api_url=API_URL,
    project=PROJECT_NAME,
    run_name="image_logging_example",
    create_project_if_not_exists=True,
    config={
        "experiment_type": "visualization",
        "resolution": "high"
    }
)

print("Creating and logging images...")

# Example 1: Generate a simple mathematical plot
fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-x/5)
ax.plot(x, y, 'b-', linewidth=2)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Oscillatory Decay Function')
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Convert to PIL Image and log
buf = BytesIO()
plt.savefig(buf, format='png', dpi=100)
buf.seek(0)
image = Image.open(buf)
client.log_image("oscillatory_decay", image)
plt.close()

# Example 2: Generate a heatmap
fig, ax = plt.subplots(figsize=(8, 6))
data = np.random.rand(20, 20)
im = ax.imshow(data, cmap='hot', interpolation='nearest')
ax.set_title('Random Heatmap')
plt.colorbar(im)
buf = BytesIO()
plt.savefig(buf, format='png', dpi=100)
buf.seek(0)
image = Image.open(buf)
client.log_image("heatmap", image)
plt.close()

# Example 3: Generate a 3D surface plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title('3D Surface Plot')
plt.colorbar(surf)
buf = BytesIO()
plt.savefig(buf, format='png', dpi=100)
buf.seek(0)
image = Image.open(buf)
client.log_image("3d_surface", image)
plt.close()

print("✓ Images logged successfully!")

# ============================================================================
# 3. CSV Data Logging
# ============================================================================

print("\n" + "=" * 60)
print("3. CSV Data Logging")
print("=" * 60)

# Create a new run for CSV logging
client = optixlog.init(
    api_key=API_KEY,
    api_url=API_URL,
    project=PROJECT_NAME,
    run_name="csv_logging_example",
    create_project_if_not_exists=True,
    config={
        "data_type": "parameter_sweep",
        "num_samples": 100
    }
)

print("Creating and logging CSV files...")

# Example 1: Parameter sweep data
csv_file = "parameter_sweep.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['wavelength', 'transmission', 'reflection', 'phase'])
    
    # Generate fake parameter sweep data
    for wavelength in np.linspace(400, 800, 100):
        transmission = 0.9 - 0.3 * np.sin(wavelength / 100)
        reflection = 1 - transmission + np.random.normal(0, 0.02)
        phase = wavelength / 50 + np.random.normal(0, 0.1)
        writer.writerow([wavelength, transmission, reflection, phase])

client.log_file("parameter_sweep", csv_file, "text/csv")

# Example 2: Training metrics over time
csv_file = "training_metrics.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
    
    for epoch in range(50):
        train_loss = 1.0 / (epoch + 1) + np.random.normal(0, 0.01)
        val_loss = train_loss * 1.2 + np.random.normal(0, 0.01)
        train_acc = 1 - np.exp(-epoch / 15) + np.random.normal(0, 0.01)
        val_acc = train_acc * 0.95 + np.random.normal(0, 0.01)
        writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc])

client.log_file("training_metrics", csv_file, "text/csv")

# Example 3: Device characterization data
csv_file = "device_characterization.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frequency', 's21_magnitude', 's21_phase', 'q_factor'])
    
    frequencies = np.linspace(150, 200, 200)
    for freq in frequencies:
        # Simulate a resonant peak
        resonance_freq = 175
        s21_mag = 1 - 0.8 * np.exp(-((freq - resonance_freq) / 2)**2)
        s21_phase = np.pi * (freq - resonance_freq) / 10
        q_factor = 50 + np.random.normal(0, 2)
        writer.writerow([freq, s21_mag, s21_phase, q_factor])

client.log_file("device_characterization", csv_file, "text/csv")

# Cleanup temporary files
for f in ["parameter_sweep.csv", "training_metrics.csv", "device_characterization.csv"]:
    if os.path.exists(f):
        os.remove(f)

print("✓ CSV files logged successfully!")

# ============================================================================
# 4. MP4 Video Logging
# ============================================================================

print("\n" + "=" * 60)
print("4. MP4 Video Logging")
print("=" * 60)

# Create a new run for video logging
client = optixlog.init(
    api_key=API_KEY,
    api_url=API_URL,
    project=PROJECT_NAME,
    run_name="video_logging_example",
    create_project_if_not_exists=True,
    config={
        "animation_type": "wave_propagation",
        "frames": 100
    }
)

print("Creating and logging MP4 video...")

# Check if ffmpeg is available
import subprocess
try:
    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    has_ffmpeg = True
except:
    has_ffmpeg = False
    print("⚠ Warning: ffmpeg not found. Install it to create MP4 videos.")
    print("   On Mac: brew install ffmpeg")
    print("   On Ubuntu: sudo apt-get install ffmpeg")

if has_ffmpeg:
    # Generate frames for animation
    import matplotlib.animation as animation
    
    # Create a simple wave propagation animation
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(0, 10, 200)
    
    def animate(frame):
        ax.clear()
        # Create a traveling wave
        t = frame * 0.1
        y = np.sin(x - t) * np.exp(-(x - 5)**2 / 4)
        ax.plot(x, y, 'b-', linewidth=2)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('Position')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Traveling Wave Animation (Frame {frame})')
        ax.grid(True, alpha=0.3)
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=False)
    
    # Save as MP4
    mp4_file = "wave_animation.mp4"
    anim.save(mp4_file, writer='ffmpeg', fps=20, bitrate=1800)
    plt.close()
    
    # Log the MP4 file
    client.log_file("wave_animation", mp4_file, "video/mp4")
    
    # Cleanup
    if os.path.exists(mp4_file):
        os.remove(mp4_file)
    
    print("✓ MP4 video logged successfully!")
else:
    print("⏭ Skipping video creation (ffmpeg not available)")

print("💡 Tip: You can create videos from any sequence of images or simulations!")

# ============================================================================
# 5. Comparing Multiple Runs
# ============================================================================

print("\n" + "=" * 60)
print("5. Comparing Multiple Runs")
print("=" * 60)

print("Creating multiple runs for comparison...")

# Define different configurations to compare
configs = [
    {"learning_rate": 0.001, "optimizer": "adam", "batch_size": 32},
    {"learning_rate": 0.01, "optimizer": "adam", "batch_size": 32},
    {"learning_rate": 0.001, "optimizer": "sgd", "batch_size": 32},
    {"learning_rate": 0.001, "optimizer": "adam", "batch_size": 64}
]

run_names = []

for i, config in enumerate(configs):
    # Create a unique run for each configuration
    run_name = f"comparison_run_{i+1}"
    run_names.append(run_name)
    
    client = optixlog.init(
        api_key=API_KEY,
        api_url=API_URL,
        project=PROJECT_NAME,
        run_name=run_name,
        create_project_if_not_exists=True,
        config=config
    )
    
    print(f"\n{run_name} - Config: {config}")
    
    # Simulate training with this configuration
    for step in range(30):
        time.sleep(0.05)
        
        # Simulate metrics that depend on configuration
        lr = config["learning_rate"]
        batch = config["batch_size"]
        opt = config["optimizer"]
        
        # Metrics vary based on configuration
        if opt == "adam":
            loss = 1.0 / (step + 1) + np.random.normal(0, 0.01)
        else:
            loss = 1.5 / (step + 1) + np.random.normal(0, 0.015)
        
        accuracy = 1 - np.exp(-step * lr * 10) + np.random.normal(0, 0.02)
        
        client.log(
            step=step,
            loss=loss,
            accuracy=accuracy,
            learning_rate=lr,
            batch_size=batch,
            optimizer=opt
        )
    
    print(f"✓ Completed {run_name}")

print(f"\n✓ Created {len(run_names)} runs for comparison!")
print(f"Run names: {run_names}")
print("\n💡 Tip: In the OptixLog dashboard, you can now compare these runs side by side!")

# ============================================================================
# 6. Multi-Metric Tracking
# ============================================================================

print("\n" + "=" * 60)
print("6. Multi-Metric Tracking")
print("=" * 60)

# Create a comprehensive run with multiple metrics
client = optixlog.init(
    api_key=API_KEY,
    api_url=API_URL,
    project=PROJECT_NAME,
    run_name="multi_metric_comprehensive",
    create_project_if_not_exists=True,
    config={
        "simulation_type": "waveguide",
        "wavelength": 1550,
        "resolution": 50,
        "pml_thickness": 1.0,
        "simulation_time": 100
    }
)

print("Running comprehensive multi-metric simulation...")
print("Tracking: power, transmission, reflection, efficiency, phase, field_energy")

# Simulate a waveguide transmission study with multiple metrics
num_steps = 100

for step in range(num_steps):
    time.sleep(0.02)
    
    # Simulate different physics metrics
    power = 1.0 - 0.3 * np.exp(-step/20) + np.random.normal(0, 0.005)
    transmission = 0.95 * power + np.random.normal(0, 0.01)
    reflection = 0.05 * (1 - power) + np.random.normal(0, 0.005)
    efficiency = transmission / (transmission + reflection + 1e-10)
    phase = 2 * np.pi * step / 50 + np.random.normal(0, 0.1)
    field_energy = power * 0.8 + np.random.normal(0, 0.01)
    
    # Log all metrics
    client.log(
        step=step,
        power=power,
        transmission=transmission,
        reflection=reflection,
        efficiency=efficiency,
        phase=phase,
        field_energy=field_energy
    )
    
    if step % 20 == 0:
        print(f"Step {step:3d}: power={power:.3f}, trans={transmission:.3f}, eff={efficiency:.3f}")

print("\n✓ Multi-metric simulation complete!")

# Create comprehensive visualization plots
print("\nCreating visualization plots...")

# Plot 1: Power and field energy over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

steps = np.arange(num_steps)
power_data = 1.0 - 0.3 * np.exp(-steps/20)
field_energy_data = power_data * 0.8

ax1.plot(steps, power_data, 'b-', linewidth=2, label='Power')
ax1.plot(steps, field_energy_data, 'r-', linewidth=2, label='Field Energy')
ax1.set_xlabel('Simulation Step')
ax1.set_ylabel('Amplitude')
ax1.set_title('Power and Field Energy Evolution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Transmission and reflection
transmission_data = 0.95 * power_data
reflection_data = 0.05 * (1 - power_data)
efficiency_data = transmission_data / (transmission_data + reflection_data + 1e-10)

ax2.plot(steps, transmission_data, 'g-', linewidth=2, label='Transmission')
ax2.plot(steps, reflection_data, 'orange', linewidth=2, label='Reflection')
ax2.plot(steps, efficiency_data, 'purple', linewidth=2, label='Efficiency')
ax2.set_xlabel('Simulation Step')
ax2.set_ylabel('Coefficient')
ax2.set_title('Transmission Metrics')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
buf = BytesIO()
plt.savefig(buf, format='png', dpi=150)
buf.seek(0)
image = Image.open(buf)
client.log_image("comprehensive_metrics", image)
plt.close()

# Plot 3: Phase evolution
fig, ax = plt.subplots(figsize=(10, 6))
phase_data = 2 * np.pi * steps / 50
ax.plot(steps, phase_data, 'b-', linewidth=2)
ax.set_xlabel('Simulation Step')
ax.set_ylabel('Phase (radians)')
ax.set_title('Phase Evolution Over Time')
ax.grid(True, alpha=0.3)
buf = BytesIO()
plt.savefig(buf, format='png', dpi=150)
buf.seek(0)
image = Image.open(buf)
client.log_image("phase_evolution", image)
plt.close()

print("✓ Comprehensive visualizations created!")

# ============================================================================
# Complete Workflow Example
# ============================================================================

print("\n" + "=" * 60)
print("Complete Workflow Example")
print("=" * 60)

print("Running comprehensive workflow example...")
print("=" * 60)

# Create the main run
client = optixlog.init(
    api_key=API_KEY,
    api_url=API_URL,
    project=PROJECT_NAME,
    run_name="complete_workflow_demo",
    create_project_if_not_exists=True,
    config={
        "experiment_id": "complete_demo_001",
        "simulation_type": "photonic_device",
        "parameters": {
            "wavelength": 1550,
            "temperature": 25,
            "material": "silicon"
        }
    }
)

print("\n1. Logging metrics during simulation...")
# Simulate a complete photonic device simulation
num_iterations = 100
for step in range(num_iterations):
    time.sleep(0.02)
    
    # Various simulation metrics
    insertion_loss = -0.5 * (1 - np.exp(-step/30)) + np.random.normal(0, 0.005)
    return_loss = -20 - 10 * np.exp(-step/20) + np.random.normal(0, 0.5)
    crosstalk = -40 - 20 * (1 - np.exp(-step/40)) + np.random.normal(0, 0.5)
    extinction_ratio = 30 * np.exp(-step/50) + np.random.normal(0, 0.5)
    phase_error = np.deg2rad(5) * np.exp(-step/25) + np.random.normal(0, 0.01)
    
    client.log(
        step=step,
        insertion_loss=insertion_loss,
        return_loss=return_loss,
        crosstalk=crosstalk,
        extinction_ratio=extinction_ratio,
        phase_error=phase_error
    )
    
    if step % 25 == 0:
        print(f"   Step {step}: Loss={insertion_loss:.3f}dB, ER={extinction_ratio:.2f}dB")

print("\n2. Creating and logging visualization images...")

# Power vs wavelength plot
fig, ax = plt.subplots(figsize=(10, 6))
wls = np.linspace(1500, 1600, 200)
power = 1 - 0.3 * np.exp(-((wls - 1550) / 20)**2)
ax.plot(wls, power, 'b-', linewidth=2)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Transmitted Power (normalized)')
ax.set_title('Spectral Response')
ax.grid(True, alpha=0.3)
buf = BytesIO()
plt.savefig(buf, format='png', dpi=150)
buf.seek(0)
image = Image.open(buf)
client.log_image("spectral_response", image)
plt.close()

# Device geometry visualization
fig, ax = plt.subplots(figsize=(10, 8))
# Create a simple device layout
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-x/5) + 2
ax.fill_between(x, y, y-0.3, alpha=0.6, color='blue')
ax.plot(x, y, 'b-', linewidth=2)
ax.set_xlabel('Position (μm)')
ax.set_ylabel('Position (μm)')
ax.set_title('Device Geometry')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
buf = BytesIO()
plt.savefig(buf, format='png', dpi=150)
buf.seek(0)
image = Image.open(buf)
client.log_image("device_geometry", image)
plt.close()

print("\n3. Creating and logging CSV data...")

# Generate measurement data
csv_file = "device_measurements.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['wavelength', 'power_out', 'phase', 'il', 'rl'])
    
    for wl in np.linspace(1530, 1570, 50):
        power_out = 1 - 0.3 * np.exp(-((wl - 1550) / 20)**2)
        phase = 2 * np.pi * (wl - 1550) / 100
        il = -10 * np.log10(power_out) if power_out > 0 else 100
        rl = -15 - 5 * np.exp(-((wl - 1550) / 30)**2)
        writer.writerow([wl, power_out, phase, il, rl])

client.log_file("device_measurements", csv_file, "text/csv")

# Create summary statistics CSV
csv_file = "summary_stats.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'min', 'max', 'mean', 'std'])
    writer.writerow(['insertion_loss', -0.5, 0, -0.25, 0.12])
    writer.writerow(['return_loss', -30, -20, -25, 3.5])
    writer.writerow(['crosstalk', -60, -40, -50, 5.2])
    writer.writerow(['extinction_ratio', 15, 30, 22.5, 4.8])

client.log_file("summary_statistics", csv_file, "text/csv")

# Cleanup
for f in ["device_measurements.csv", "summary_stats.csv"]:
    if os.path.exists(f):
        os.remove(f)

print("\n4. Creating multi-panel comparison plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Insertion and return loss
steps = np.arange(num_iterations)
il_data = -0.5 * (1 - np.exp(-steps/30))
rl_data = -20 - 10 * np.exp(-steps/20)
axes[0, 0].plot(steps, il_data, 'b-', linewidth=2, label='Insertion Loss')
axes[0, 0].plot(steps, rl_data, 'r-', linewidth=2, label='Return Loss')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Loss (dB)')
axes[0, 0].set_title('Loss Metrics')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Crosstalk and extinction ratio
xt_data = -40 - 20 * (1 - np.exp(-steps/40))
er_data = 30 * np.exp(-steps/50)
axes[0, 1].plot(steps, xt_data, 'g-', linewidth=2, label='Crosstalk')
axes[0, 1].plot(steps, er_data, 'orange', linewidth=2, label='Extinction Ratio')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Metric (dB)')
axes[0, 1].set_title('Performance Metrics')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Panel 3: Phase error
phase_err_data = np.deg2rad(5) * np.exp(-steps/25)
axes[1, 0].plot(steps, np.rad2deg(phase_err_data), 'purple', linewidth=2)
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Phase Error (degrees)')
axes[1, 0].set_title('Phase Error Evolution')
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Final metric comparison
final_metrics = [il_data[-1], rl_data[-1], xt_data[-1], er_data[-1]]
metric_names = ['IL', 'RL', 'XT', 'ER']
colors = ['blue', 'red', 'green', 'orange']
axes[1, 1].bar(metric_names, final_metrics, color=colors, alpha=0.7)
axes[1, 1].set_ylabel('Value (dB)')
axes[1, 1].set_title('Final Metrics Summary')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
buf = BytesIO()
plt.savefig(buf, format='png', dpi=150)
buf.seek(0)
image = Image.open(buf)
client.log_image("comprehensive_analysis", image)
plt.close()

print("\n" + "=" * 60)
print("✓ Complete workflow demonstration finished!")
print("\nSummary of what was logged:")
print("  • 100 metric log entries with 5 different metrics")
print("  • 3 visualization images")
print("  • 2 CSV data files")
print("  • 1 comprehensive multi-panel analysis plot")
print("\n💡 Check your OptixLog dashboard to see all the logged data!")

print("\n" + "=" * 60)
print("✓ All tutorials completed successfully!")
print("=" * 60)

