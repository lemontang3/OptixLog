"""
Antenna Radiation Analysis with OptixLog Integration

Antenna radiation analysis with real-time metrics tracking

Based on the Meep tutorial: antenna-radiation.py
"""

import os
import math
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
    
    print("🚀 Starting Antenna Radiation Analysis")
    
    # Initialize OptixLog client
    try:
        client = optixlog.init(
            run_name="antenna_radiation_analysis",
            config={
                "simulation_type": "antenna_radiation",
                "description": "Antenna radiation pattern analysis with real-time metrics",
                "framework": "meep",
                "original_file": "antenna-radiation.py"
            },
            create_project_if_not_exists=True
        )
        print("✅ OptixLog client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize OptixLog: {e}")
        print("💡 Make sure to set your API key: export OPTIX_API_KEY='proj_your_key'")
        client = None
    # Simulation parameters
    resolution = 50  # pixels/um
    sxy = 4
    dpml = 1
    cell = mp.Vector3(sxy + 2 * dpml, sxy + 2 * dpml)
    
    pml_layers = [mp.PML(dpml)]
    
    fcen = 1.0
    df = 0.4
    src_cmpt = mp.Ex
    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df), center=mp.Vector3(), component=src_cmpt
        )
    ]
    
    if src_cmpt == mp.Ex:
        symmetries = [mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=+1)]
    elif src_cmpt == mp.Ey:
        symmetries = [mp.Mirror(mp.X, phase=+1), mp.Mirror(mp.Y, phase=-1)]
    elif src_cmpt == mp.Ez:
        symmetries = [mp.Mirror(mp.X, phase=+1), mp.Mirror(mp.Y, phase=+1)]
    else:
        symmetries = []
    
    # Log initial simulation parameters
    if client:
        client.log(step=0,
            resolution=resolution,
            fcen=fcen,
            df=df,
            sxy=sxy,
            dpml=dpml,
            source_component=str(src_cmpt),
            simulation_type="antenna_radiation"
        )
    
    sim = mp.Simulation(
        cell_size=cell,
        resolution=resolution,
        sources=sources,
        symmetries=symmetries,
        boundary_layers=pml_layers,
    )

    nearfield_box = sim.add_near2far(
        fcen,
        0,
        1,
        mp.Near2FarRegion(center=mp.Vector3(0, +0.5 * sxy), size=mp.Vector3(sxy, 0)),
        mp.Near2FarRegion(
            center=mp.Vector3(0, -0.5 * sxy), size=mp.Vector3(sxy, 0), weight=-1
        ),
        mp.Near2FarRegion(center=mp.Vector3(+0.5 * sxy, 0), size=mp.Vector3(0, sxy)),
        mp.Near2FarRegion(
            center=mp.Vector3(-0.5 * sxy, 0), size=mp.Vector3(0, sxy), weight=-1
        ),
    )
    
    flux_box = sim.add_flux(
        fcen,
        0,
        1,
        mp.FluxRegion(center=mp.Vector3(0, +0.5 * sxy), size=mp.Vector3(sxy, 0)),
        mp.FluxRegion(center=mp.Vector3(0, -0.5 * sxy), size=mp.Vector3(sxy, 0), weight=-1),
        mp.FluxRegion(center=mp.Vector3(+0.5 * sxy, 0), size=mp.Vector3(0, sxy)),
        mp.FluxRegion(center=mp.Vector3(-0.5 * sxy, 0), size=mp.Vector3(0, sxy), weight=-1),
    )
    
    # Real-time metrics tracking during simulation
    times = []
    flux_values = []
    field_energies = []
    
    print("🔄 Starting simulation with real-time metrics tracking...")
    
    # Use Meep's built-in monitoring with a custom function
    def monitor_simulation(sim):
        """Monitor function that gets called during simulation"""
        current_time = sim.meep_time()
        current_flux = mp.get_fluxes(flux_box)[0] if mp.get_fluxes(flux_box) else 0
        
        # Calculate field energy (approximate)
        try:
            field_energy = sim.field_energy()
        except:
            field_energy = 0
        
        times.append(current_time)
        flux_values.append(current_flux)
        field_energies.append(field_energy)
        
        # Log metrics at EVERY time step to OptixLog
        if client:
            client.log(step=len(times),
                simulation_time=current_time,
                flux_value=current_flux,
                field_energy=field_energy,
                step_count=len(times),
                time_step=len(times)
            )
        
        # Print every 10 steps to avoid spam
        if len(times) % 10 == 0:
            print(f"⏱️  Step {len(times)}: Time: {current_time:.2f}, Flux: {current_flux:.6f}, Energy: {field_energy:.6f}")
    
    # Run simulation with simple monitoring approach
    print("🔄 Running simulation with monitoring...")
    
    # Run simulation normally first
    sim.run(until_after_sources=mp.stop_when_dft_decayed())
    
    # Log initial metrics after simulation
    monitor_simulation(sim)
    
    # Log simulation completion (use high step number to avoid conflicts)
    if client:
        final_step = len(times) + 1000  # Use high number to separate from time steps
        client.log(step=final_step, simulation_completed=True)

    near_flux = mp.get_fluxes(flux_box)[0]
    
    # half side length of far-field square box OR radius of far-field circle
    r = 1000 / fcen
    
    # resolution of far fields (points/μm)
    res_ff = 1
    
    far_flux_box = (
        nearfield_box.flux(
            mp.Y, mp.Volume(center=mp.Vector3(y=r), size=mp.Vector3(2 * r)), res_ff
        )[0]
        - nearfield_box.flux(
            mp.Y, mp.Volume(center=mp.Vector3(y=-r), size=mp.Vector3(2 * r)), res_ff
        )[0]
        + nearfield_box.flux(
            mp.X, mp.Volume(center=mp.Vector3(r), size=mp.Vector3(y=2 * r)), res_ff
        )[0]
        - nearfield_box.flux(
            mp.X, mp.Volume(center=mp.Vector3(-r), size=mp.Vector3(y=2 * r)), res_ff
        )[0]
    )
    
    npts = 100  # number of points in [0,2*pi) range of angles
    angles = 2 * math.pi / npts * np.arange(npts)
    
    E = np.zeros((npts, 3), dtype=np.complex128)
    H = np.zeros((npts, 3), dtype=np.complex128)
    for n in range(npts):
        ff = sim.get_farfield(
            nearfield_box, mp.Vector3(r * math.cos(angles[n]), r * math.sin(angles[n]))
        )
        E[n, :] = [ff[j] for j in range(3)]
        H[n, :] = [ff[j + 3] for j in range(3)]
    
    Px = np.real(np.conj(E[:, 1]) * H[:, 2] - np.conj(E[:, 2]) * H[:, 1])
    Py = np.real(np.conj(E[:, 2]) * H[:, 0] - np.conj(E[:, 0]) * H[:, 2])
    Pr = np.sqrt(np.square(Px) + np.square(Py))
    
    # integrate the radial flux over the circle circumference
    far_flux_circle = np.sum(Pr) * 2 * np.pi * r / len(Pr)
    
    print(f"flux:, {near_flux:.6f}, {far_flux_box:.6f}, {far_flux_circle:.6f}")
    
    # Analytic formulas for the radiation pattern as the Poynting vector
    # of an electric dipole in vacuum. From Section 4.2 "Infinitesimal Dipole"
    # of Antenna Theory: Analysis and Design, 4th Edition (2016) by C. Balanis.
    if src_cmpt == mp.Ex:
        flux_theory = np.sin(angles) ** 2
    elif src_cmpt == mp.Ey:
        flux_theory = np.cos(angles) ** 2
    elif src_cmpt == mp.Ez:
        flux_theory = np.ones((npts,))
    
    # Create comprehensive analysis plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Radiation pattern (polar plot)
    ax1 = plt.subplot(2, 2, 1, projection='polar')
    ax1.plot(angles, Pr / max(Pr), "b-", label="Meep", linewidth=2)
    ax1.plot(angles, flux_theory, "r--", label="Theory", linewidth=2)
    ax1.set_rmax(1)
    ax1.set_rticks([0, 0.5, 1])
    ax1.grid(True)
    ax1.set_rlabel_position(22)
    ax1.legend()
    ax1.set_title("Radiation Pattern Comparison")
    
    # 2. Real-time flux evolution
    ax2.plot(times, flux_values, 'g-', linewidth=2, label='Flux')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Flux Value')
    ax2.set_title('Flux Evolution During Simulation')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Field energy evolution
    ax3.plot(times, field_energies, 'm-', linewidth=2, label='Field Energy')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Field Energy')
    ax3.set_title('Field Energy Evolution')
    ax3.grid(True)
    ax3.legend()
    
    # 4. Radiation pattern comparison (linear)
    ax4.plot(np.degrees(angles), Pr / max(Pr), "b-", label="Meep", linewidth=2)
    ax4.plot(np.degrees(angles), flux_theory, "r--", label="Theory", linewidth=2)
    ax4.set_xlabel('Angle (degrees)')
    ax4.set_ylabel('Normalized Power')
    ax4.set_title('Radiation Pattern (Linear)')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig("antenna_radiation_analysis.png", dpi=150, bbox_inches="tight")
    
    # Calculate comprehensive metrics
    if client:
        # Calculate radiation pattern statistics
        max_radiation_angle = np.degrees(angles[np.argmax(Pr)])
        max_theory_angle = np.degrees(angles[np.argmax(flux_theory)])
        
        # Calculate correlation and error metrics
        correlation_coeff = np.corrcoef(Pr, flux_theory)[0, 1]
        
        # Calculate relative error safely (avoid division by zero)
        flux_theory_safe = np.where(flux_theory == 0, 1e-10, flux_theory)  # Replace zeros with small value
        relative_error = np.mean(np.abs(Pr - flux_theory) / flux_theory_safe) * 100
        
        # Ensure all values are finite for JSON serialization
        correlation_coeff = np.nan_to_num(correlation_coeff, nan=0.0, posinf=1.0, neginf=-1.0)
        relative_error = np.nan_to_num(relative_error, nan=0.0, posinf=100.0, neginf=0.0)
        
        # Calculate directivity (approximate) - avoid division by zero
        mean_Pr = np.mean(Pr)
        mean_flux_theory = np.mean(flux_theory)
        
        directivity = 2 * np.max(Pr) / mean_Pr if mean_Pr > 0 else 0
        theory_directivity = 2 * np.max(flux_theory) / mean_flux_theory if mean_flux_theory > 0 else 0
        
        # Ensure directivity values are finite
        directivity = np.nan_to_num(directivity, nan=0.0, posinf=100.0, neginf=0.0)
        theory_directivity = np.nan_to_num(theory_directivity, nan=0.0, posinf=100.0, neginf=0.0)
        
        # Log comprehensive results (use high step number)
        analysis_step = len(times) + 2000  # Use high number to separate from time steps
        client.log(step=analysis_step,
            # Flux analysis
            near_flux=near_flux,
            far_flux_box=far_flux_box,
            far_flux_circle=far_flux_circle,
            flux_ratio=far_flux_circle/near_flux if near_flux > 0 else 0,
            
            # Radiation pattern analysis
            max_radiation_angle_degrees=max_radiation_angle,
            max_theory_angle_degrees=max_theory_angle,
            correlation_coefficient=correlation_coeff,
            relative_error_percent=relative_error,
            
            # Directivity analysis
            directivity=directivity,
            theory_directivity=theory_directivity,
            directivity_error_percent=abs(directivity - theory_directivity) / theory_directivity * 100 if theory_directivity > 0 else 0,
            
            # Simulation metrics
            total_simulation_time=times[-1] if times else 0,
            final_flux=flux_values[-1] if flux_values else 0,
            final_field_energy=field_energies[-1] if field_energies else 0,
            max_flux=max(flux_values) if flux_values else 0,
            max_field_energy=max(field_energies) if field_energies else 0,
            
            # Pattern statistics
            max_radiation_power=np.nan_to_num(np.max(Pr), nan=0.0, posinf=1.0, neginf=0.0),
            mean_radiation_power=np.nan_to_num(np.mean(Pr), nan=0.0, posinf=1.0, neginf=0.0),
            radiation_pattern_std=np.nan_to_num(np.std(Pr), nan=0.0, posinf=1.0, neginf=0.0)
        )
        
        # Log the comprehensive analysis plot
        from PIL import Image
        analysis_img = Image.open("antenna_radiation_analysis.png")
        client.log_image("antenna_radiation_analysis", analysis_img, 
                        {"description": "Comprehensive antenna radiation analysis with real-time metrics"})
        
        # Log detailed time series data (use high step number)
        timeseries_step = len(times) + 3000  # Use high number to separate from time steps
        client.log(step=timeseries_step,
            simulation_times=times,
            flux_evolution=flux_values,
            field_energy_evolution=field_energies,
            angles_degrees=np.degrees(angles).tolist(),
            radiation_pattern=Pr.tolist(),
            theory_pattern=flux_theory.tolist()
        )
        
        # Print comprehensive analysis summary
        print(f"\n📊 Antenna Radiation Analysis Results:")
        print(f"   Near flux: {near_flux:.6f}")
        print(f"   Far flux (box): {far_flux_box:.6f}")
        print(f"   Far flux (circle): {far_flux_circle:.6f}")
        print(f"   Flux ratio: {far_flux_circle/near_flux:.6f}" if near_flux > 0 else "   Flux ratio: N/A")
        print(f"   Max radiation at: {max_radiation_angle:.1f}°")
        print(f"   Max theory at: {max_theory_angle:.1f}°")
        print(f"   Correlation coefficient: {correlation_coeff:.6f}")
        print(f"   Relative error: {relative_error:.2f}%")
        print(f"   Directivity: {directivity:.2f}")
        print(f"   Theory directivity: {theory_directivity:.2f}")
        print(f"   Total simulation time: {times[-1]:.2f}" if times else "   Total simulation time: N/A")
        print(f"   Final flux: {flux_values[-1]:.6f}" if flux_values else "   Final flux: N/A")
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
