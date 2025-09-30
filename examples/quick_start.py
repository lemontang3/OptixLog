"""
Quick Start Example - OptixLog Integration

This example demonstrates the basic integration of OptixLog with a simple simulation.
It shows how to:
1. Initialize OptixLog client
2. Log scalar metrics during simulation
3. Upload artifacts (plots, data files)
4. Track simulation parameters

Usage:
    export OPTIX_API_KEY="proj_your_api_key_here"
    python examples/01_quick_start.py

Author: OptixLog Team
"""

import optixlog
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    """Main simulation function with OptixLog integration"""
    
    # Check if this is the master process
    if not optixlog.is_master_process():
        mpi_info = optixlog.get_mpi_info()
        print(f"Worker process (rank {mpi_info[1]}/{mpi_info[2]}) - skipping simulation")
        return
    
    print("🚀 Starting OptixLog Quick Start Example")
    
    # Initialize OptixLog client
    try:
        client = optixlog.init(
            run_name="quick_start_demo",
            config={
                "simulation_type": "demo",
                "description": "Basic OptixLog integration example",
                "parameters": {
                    "num_steps": 10,
                    "frequency": 1.0,
                    "amplitude": 0.5
                }
            },
            create_project_if_not_exists=True
        )
        print("✅ OptixLog client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize OptixLog: {e}")
        print("💡 Make sure to set your API key: export OPTIX_API_KEY='proj_your_key'")
        return
    
    # Simulation parameters
    num_steps = 10
    frequency = 1.0
    amplitude = 0.5
    
    print(f"📊 Running simulation for {num_steps} steps...")
    
    # Run simulation and log metrics
    for step in range(num_steps):
        # Simulate some computation
        time.sleep(0.2)
        
        # Calculate some metrics (simulating a physical process)
        signal = amplitude * np.sin(2 * np.pi * frequency * step / num_steps)
        noise = random.random() * 0.1
        signal_with_noise = signal + noise
        
        # Calculate derived metrics
        power = signal_with_noise ** 2
        snr = abs(signal) / (noise + 1e-10)  # Signal-to-noise ratio
        
        # Log metrics to OptixLog
        client.log(
            step=step,
            signal=signal_with_noise,
            power=power,
            snr=snr,
            noise_level=noise,
            frequency=frequency,
            amplitude=amplitude
        )
        
        print(f"  Step {step}: signal={signal_with_noise:.3f}, power={power:.3f}, SNR={snr:.3f}")
    
    print("📈 Creating visualization...")
    
    # Create a simple plot
    steps = np.arange(num_steps)
    signals = [amplitude * np.sin(2 * np.pi * frequency * s / num_steps) for s in steps]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, signals, 'b-', linewidth=2, label='Signal')
    plt.xlabel('Time Step')
    plt.ylabel('Amplitude')
    plt.title('Quick Start Simulation Results')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plot_filename = 'quick_start_results.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Upload plot to OptixLog
    try:
        client.log_file(
            plot_filename, 
            "quick_start_results.png", 
            "image/png"
        )
        print(f"✅ Uploaded plot: {plot_filename}")
    except Exception as e:
        print(f"❌ Failed to upload plot: {e}")
    
    # Create and upload data file
    data_filename = 'quick_start_data.csv'
    try:
        data = np.column_stack([steps, signals])
        np.savetxt(data_filename, data, delimiter=',', 
                  header='step,amplitude', comments='')
        
        client.log_file(
            data_filename, 
            "quick_start_data.csv", 
            "text/csv"
        )
        print(f"✅ Uploaded data: {data_filename}")
    except Exception as e:
        print(f"❌ Failed to upload data: {e}")
    
    # Clean up local files
    for filename in [plot_filename, data_filename]:
        if os.path.exists(filename):
            os.remove(filename)
    
    print("🎉 Quick Start example completed successfully!")
    print("📱 Check your OptixLog dashboard to see the results:")
    print("   https://optixlog.com")

if __name__ == "__main__":
    main()
