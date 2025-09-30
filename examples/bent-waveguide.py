"""
Bent Waveguide Analysis with OptixLog Integration

Waveguide simulation and analysis with comprehensive logging

Based on the Meep tutorial: bent-waveguide.py
"""

import os
import optixlog
import meep as mp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("agg")

def main():
    """Main simulation function with OptixLog integration"""
    
    # Check if this is the master process
    if not optixlog.is_master_process():
        mpi_info = optixlog.get_mpi_info()
        print(f"Worker process (rank {mpi_info[1]}/{mpi_info[2]}) - skipping simulation")
        return
    
    print("🚀 Starting Bent Waveguide Analysis")
    
    # Initialize OptixLog client
    try:
        client = optixlog.init(
            run_name="bent_waveguide_analysis",
            config={
                "simulation_type": "bent_waveguide",
                "description": "Bent waveguide simulation with comprehensive analysis",
                "framework": "meep",
                "original_file": "bent-waveguide.py"
            },
            create_project_if_not_exists=True
        )
        print("✅ OptixLog client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize OptixLog: {e}")
        print("💡 Make sure to set your API key: export OPTIX_API_KEY='proj_your_key'")
        client = None

    # Simulation parameters
    cell = mp.Vector3(16, 16, 0)
    geometry = [
        mp.Block(
            mp.Vector3(12, 1, mp.inf),
            center=mp.Vector3(-2.5, -3.5),
            material=mp.Medium(epsilon=12),
        ),
        mp.Block(
            mp.Vector3(1, 12, mp.inf),
            center=mp.Vector3(3.5, 2),
            material=mp.Medium(epsilon=12),
        ),
    ]
    pml_layers = [mp.PML(1.0)]
    resolution = 10
    wavelength = 2 * (11**0.5)
    
    sources = [
        mp.Source(
            mp.ContinuousSource(wavelength=wavelength, width=20),
            component=mp.Ez,
            center=mp.Vector3(-7, -3.5),
            size=mp.Vector3(0, 1),
        )
    ]
    
    # Log initial simulation parameters
    if client:
        client.log(step=0,
            resolution=resolution,
            wavelength=wavelength,
            cell_size_x=cell.x,
            cell_size_y=cell.y,
            waveguide_epsilon=12,
            pml_thickness=1.0,
            simulation_type="bent_waveguide"
        )
    
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
    )
    
    print("🔄 Running bent waveguide simulation...")
    
    # Run simulation with field output
    sim.run(
        mp.at_beginning(mp.output_epsilon),
        mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
        until=200,
    )
    
    # Log simulation completion
    if client:
        client.log(step=1, simulation_completed=True)
    
    # Create comprehensive analysis plots
    print("📊 Creating analysis plots...")
    
    # 1. Permittivity plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot permittivity
    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    im1 = ax1.imshow(eps_data, interpolation='sinc', cmap='binary')
    ax1.set_title('Permittivity Distribution')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)
    
    # Plot field distribution
    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    im2 = ax2.imshow(np.real(ez_data), interpolation='sinc', cmap='RdBu')
    ax2.set_title('Ez Field Distribution')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig("bent_waveguide_analysis.png", dpi=150, bbox_inches="tight")
    
    # 2. Field intensity plot
    fig, ax = plt.subplots(figsize=(8, 6))
    intensity = np.abs(ez_data)**2
    im = ax.imshow(intensity, interpolation='sinc', cmap='hot')
    ax.set_title('Field Intensity Distribution')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax)
    plt.savefig("bent_waveguide_intensity.png", dpi=150, bbox_inches="tight")
    
    # Calculate and log comprehensive metrics
    if client:
        # Calculate field statistics
        max_field = np.max(np.abs(ez_data))
        mean_field = np.mean(np.abs(ez_data))
        field_std = np.std(np.abs(ez_data))
        
        # Calculate intensity statistics
        max_intensity = np.max(intensity)
        mean_intensity = np.mean(intensity)
        total_power = np.sum(intensity)
        
        # Calculate waveguide metrics
        waveguide_area = np.sum(eps_data > 1.5)  # Approximate waveguide area
        field_confinement = np.sum(intensity * (eps_data > 1.5)) / total_power if total_power > 0 else 0
        
        # Log comprehensive results
        client.log(step=2,
            # Field analysis
            max_field_amplitude=max_field,
            mean_field_amplitude=mean_field,
            field_std=field_std,
            
            # Intensity analysis
            max_intensity=max_intensity,
            mean_intensity=mean_intensity,
            total_power=total_power,
            
            # Waveguide analysis
            waveguide_area_pixels=waveguide_area,
            field_confinement_ratio=field_confinement,
            
            # Simulation metrics
            simulation_time=200,
            resolution=resolution,
            wavelength=wavelength,
            cell_size_x=cell.x,
            cell_size_y=cell.y
        )
        
        # Log the analysis plots
        from PIL import Image
        analysis_img = Image.open("bent_waveguide_analysis.png")
        client.log_image("bent_waveguide_analysis", analysis_img, 
                        {"description": "Bent waveguide permittivity and field distribution"})
        
        intensity_img = Image.open("bent_waveguide_intensity.png")
        client.log_image("bent_waveguide_intensity", intensity_img, 
                        {"description": "Bent waveguide field intensity distribution"})
        
        # Print comprehensive analysis summary
        print(f"\n📊 Bent Waveguide Analysis Results:")
        print(f"   Max field amplitude: {max_field:.6f}")
        print(f"   Mean field amplitude: {mean_field:.6f}")
        print(f"   Field standard deviation: {field_std:.6f}")
        print(f"   Max intensity: {max_intensity:.6f}")
        print(f"   Mean intensity: {mean_intensity:.6f}")
        print(f"   Total power: {total_power:.6f}")
        print(f"   Waveguide area: {waveguide_area} pixels")
        print(f"   Field confinement ratio: {field_confinement:.3f}")
    else:
        print("⚠️ No OptixLog client available for detailed logging")

if __name__ == "__main__":
    try:
        main()
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
