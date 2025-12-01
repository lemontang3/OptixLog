"""
Absorber 1D.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: absorber-1d.py
"""

import os
import argparse
import optixlog
import meep as mp
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

# OptixLog Configuration
api_key = os.getenv("OPTIX_API_KEY", "proj_YOUR_API_KEY_HERE")
api_url = os.getenv("OPTIX_API_URL", "https://optixlog.com")
project_name = os.getenv("OPTIX_PROJECT", "MeepExamples")

def main(args):
    """Main simulation function with OptixLog integration"""
    
    # Check if this is the master process
    if not optixlog.is_master_process():
        mpi_info = optixlog.get_mpi_info()
        print(f"Worker process (rank {mpi_info[1]}/{mpi_info[2]}) - skipping simulation")
        return
    
    print(f"🚀 Initializing OptixLog client for project: {project_name}")
    
    try:
        # Initialize OptixLog client
        client = optixlog.init(
            api_key=api_key,
            api_url=api_url,
            project=project_name,
            run_name="absorber-1d_simulation",
            config={
                "simulation_type": "general",
                "description": "Electromagnetic simulation",
                "framework": "meep",
                "original_file": "absorber-1d.py"
            },
            create_project_if_not_exists=True
        )
        print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
        print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")
        
        # Log simulation parameters
        client.log(step=0,
            resolution=40
        )
    except Exception as e:
        print(f"❌ Failed to initialize OptixLog: {e}")
        client = None

    from meep.materials import Al

    resolution = 40
    cell_size = mp.Vector3(z=10)

    boundary_layers = [
        mp.PML(1, direction=mp.Z) if args.pml else mp.Absorber(1, direction=mp.Z)
    ]

    sources = [
        mp.Source(
            src=mp.GaussianSource(1 / 0.803, fwidth=0.1),
            center=mp.Vector3(),
            component=mp.Ex,
        )
    ]

    # Lists to store simulation data for plotting
    times = []
    field_values = []
    field_amplitudes = []
    
    def print_stuff(sim):
        p = sim.get_field_point(mp.Ex, mp.Vector3())
        current_time = sim.meep_time()
        field_real = p.real
        field_imag = p.imag
        field_amp = abs(p)
        
        times.append(current_time)
        field_values.append(field_real)
        field_amplitudes.append(field_amp)
        
        print(f"ex:, {current_time}, {field_real}, {field_imag}, {field_amp}")

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        dimensions=1,
        default_material=Al,
        boundary_layers=boundary_layers,
        sources=sources,
    )

    sim.run(
        mp.at_every(10, print_stuff),
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(), 1e-6),
    )
    
    # Log simulation completion
    if client:
        client.log(step=1, simulation_completed=True)
    
    # Create comprehensive plots
    if times and field_values:
        # Plot 1: Field evolution over time
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(times, field_values, 'b-', linewidth=2, label='Real part')
        plt.xlabel('Time')
        plt.ylabel('Ex Field (Real)')
        plt.title('Electric Field Evolution (Real Part)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(times, field_amplitudes, 'r-', linewidth=2, label='Amplitude')
        plt.xlabel('Time')
        plt.ylabel('|Ex| Field Amplitude')
        plt.title('Field Amplitude Evolution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Field decay analysis
        plt.subplot(2, 2, 3)
        # Calculate decay rate
        if len(field_amplitudes) > 10:
            # Fit exponential decay to the tail of the simulation
            tail_start = len(field_amplitudes) // 2
            tail_times = np.array(times[tail_start:])
            tail_amps = np.array(field_amplitudes[tail_start:])
            
            # Log-linear plot for exponential decay
            plt.semilogy(tail_times, tail_amps, 'go-', markersize=4, label='Field decay')
            plt.xlabel('Time')
            plt.ylabel('|Ex| Field Amplitude (log scale)')
            plt.title('Field Decay Analysis')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Plot 3: Spatial field distribution at final time
        plt.subplot(2, 2, 4)
        # Get field data along z-axis
        z_coords = np.linspace(-5, 5, 100)
        field_profile = []
        for z in z_coords:
            field_val = sim.get_field_point(mp.Ex, mp.Vector3(z=z))
            field_profile.append(abs(field_val))
        
        plt.plot(z_coords, field_profile, 'purple', linewidth=2)
        plt.xlabel('Z Position')
        plt.ylabel('|Ex| Field Amplitude')
        plt.title('Final Field Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("absorber_1d_analysis.png", dpi=150, bbox_inches="tight")
        
        # Log the analysis plot to OptixLog
        if client:
            from PIL import Image
            analysis_img = Image.open("absorber_1d_analysis.png")
            client.log_image("field_analysis", analysis_img, 
                            {"description": "Comprehensive field evolution and decay analysis"})
        
        # Plot 4: Simulation setup visualization
        plt.figure(figsize=(10, 6))
        
        # Create a simple visualization of the simulation setup
        z_positions = np.linspace(-5, 5, 200)
        material_profile = []
        source_profile = []
        
        for z in z_positions:
            # Material profile (Aluminum everywhere)
            material_profile.append(1.0)  # Simplified representation
            
            # Source profile (Gaussian at center)
            source_strength = np.exp(-((z - 0)**2) / (2 * 0.5**2))
            source_profile.append(source_strength)
        
        plt.subplot(1, 2, 1)
        plt.plot(z_positions, material_profile, 'b-', linewidth=3, label='Aluminum medium')
        plt.xlabel('Z Position')
        plt.ylabel('Material')
        plt.title('Simulation Setup: Material Distribution')
        plt.ylim(0, 1.2)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(z_positions, source_profile, 'r-', linewidth=2, label='Gaussian source')
        plt.xlabel('Z Position')
        plt.ylabel('Source Strength')
        plt.title('Simulation Setup: Source Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("absorber_1d_setup.png", dpi=150, bbox_inches="tight")
        
        # Log the setup plot to OptixLog
        if client:
            setup_img = Image.open("absorber_1d_setup.png")
            client.log_image("simulation_setup", setup_img, 
                            {"description": "Simulation setup showing material and source distribution"})
        
        # Log detailed simulation metrics
        if client:
            max_field = max(field_amplitudes) if field_amplitudes else 0
            final_field = field_amplitudes[-1] if field_amplitudes else 0
            decay_ratio = final_field / max_field if max_field > 0 else 0
            
            client.log(step=2,
                      max_field_amplitude=max_field,
                      final_field_amplitude=final_field,
                      field_decay_ratio=decay_ratio,
                      simulation_time=times[-1] if times else 0,
                      total_time_steps=len(times),
                      boundary_type="PML" if args.pml else "Absorber",
                      material="Aluminum",
                      resolution=resolution,
                      cell_size_z=10)
        
        print(f"\n📊 Simulation Analysis:")
        print(f"   Max field amplitude: {max_field:.6f}")
        print(f"   Final field amplitude: {final_field:.6f}")
        print(f"   Decay ratio: {decay_ratio:.6f}")
        print(f"   Total simulation time: {times[-1] if times else 0:.2f}")
        print(f"   Time steps recorded: {len(times)}")
    else:
        print("⚠️ No field data collected during simulation")
        # Set default values for variables that might be referenced later
        max_field = 0
        final_field = 0
        decay_ratio = 0


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-pml", action="store_true", default=False, help="Use PML as boundary layer"
        )
        args = parser.parse_args()
        main(args)
    except ValueError as e:
        print(f"\n❌ OptixLog Error: {e}")
        print("Please ensure your API key and URL are correct.")
    except Exception as e:
        print(f"\n❌ Simulation Error: {e}")
    finally:
        # Clean up generated files
        import glob
        for file_path in glob.glob("*.png") + glob.glob("*.csv") + glob.glob("*.txt"):
            if os.path.exists(file_path):
                os.remove(file_path)
