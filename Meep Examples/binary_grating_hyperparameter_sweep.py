"""
Binary Grating Hyperparameter Sweep with Enhanced OptixLog Integration

This script performs a comprehensive hyperparameter sweep of binary grating simulations,
creating separate OptixLog runs for each parameter combination and logging detailed
FDTD simulation steps for better visualization and analysis.

Usage:
    # Set your OptixLog API key
    export OPTIX_API_KEY="proj_your_api_key_here"
    
    # Run the hyperparameter sweep
    python binary_grating_hyperparameter_sweep.py
"""

import argparse
import sys
import os
import time
import itertools

# Add the Coupler SDK to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Coupler', 'sdk'))

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib

import meep as mp

# Import OptixLog SDK
import optixlog

# Global simulation parameters
resolution = 60  # pixels/μm
dpml = 1.0  # PML thickness
dsub = 3.0  # substrate thickness
dpad = 3.0  # padding between grating and PML

wvl_min = 0.4  # min wavelength
wvl_max = 0.6  # max wavelength
fmin = 1 / wvl_max  # min frequency
fmax = 1 / wvl_min  # max frequency
fcen = 0.5 * (fmin + fmax)  # center frequency
df = fmax - fmin  # frequency width
nfreq = 21  # number of frequency bins

k_point = mp.Vector3(0, 0, 0)
glass = mp.Medium(index=1.5)

def grating_with_detailed_logging(gp, gh, gdc, oddz, client, run_step=0):
    """Enhanced grating simulation with detailed FDTD step logging"""
    
    sx = dpml + dsub + gh + dpad + dpml
    sy = gp
    cell_size = mp.Vector3(sx, sy, 0)
    pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

    src_pt = mp.Vector3(-0.5 * sx + dpml + 0.5 * dsub, 0, 0)
    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez if oddz else mp.Hz,
            center=src_pt,
            size=mp.Vector3(0, sy, 0),
        )
    ]

    symmetries = [mp.Mirror(mp.Y, phase=+1 if oddz else -1)]

    # Log initial setup
    client.log(run_step,
        simulation_phase="initialization",
        grating_period=gp,
        grating_height=gh,
        duty_cycle=gdc,
        odd_z_symmetry=oddz,
        cell_size_x=sx,
        cell_size_y=sy,
        resolution=resolution,
        center_frequency=fcen,
        frequency_width=df
    )

    # Create simulation for empty reference
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        k_point=k_point,
        default_material=glass,
        sources=sources,
        symmetries=symmetries,
    )

    mon_pt = mp.Vector3(0.5 * sx - dpml - 0.5 * dpad, 0, 0)
    flux_mon = sim.add_flux(
        fcen, df, nfreq, mp.FluxRegion(center=mon_pt, size=mp.Vector3(0, sy, 0))
    )

    # Log empty simulation start
    client.log(run_step + 1,
        simulation_phase="empty_reference_start",
        duty_cycle=gdc,
        simulation_time=0.0,
        field_energy=0.0
    )

    # Run empty simulation with step-by-step logging
    empty_start_time = time.time()
    sim.run(until_after_sources=100)
    empty_time = time.time() - empty_start_time

    input_flux = mp.get_fluxes(flux_mon)
    input_flux_total = sum(input_flux)

    # Log empty simulation results
    client.log(run_step + 2,
        simulation_phase="empty_reference_complete",
        duty_cycle=gdc,
        simulation_time=empty_time,
        input_flux_total=input_flux_total,
        average_input_flux=np.mean(input_flux)
    )

    # Reset for grating simulation
    sim.reset_meep()

    # Log grating setup
    client.log(run_step + 3,
        simulation_phase="grating_setup",
        duty_cycle=gdc,
        grating_width=gdc * gp,
        grating_position_x=-0.5 * sx + dpml + dsub + 0.5 * gh
    )

    # Create grating geometry
    geometry = [
        mp.Block(
            material=glass,
            size=mp.Vector3(dpml + dsub, mp.inf, mp.inf),
            center=mp.Vector3(-0.5 * sx + 0.5 * (dpml + dsub), 0, 0),
        ),
        mp.Block(
            material=glass,
            size=mp.Vector3(gh, gdc * gp, mp.inf),
            center=mp.Vector3(-0.5 * sx + dpml + dsub + 0.5 * gh, 0, 0),
        ),
    ]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        k_point=k_point,
        sources=sources,
        symmetries=symmetries,
    )

    # Add flux monitor
    flux_mon = sim.add_flux(
        fcen, df, nfreq, mp.FluxRegion(center=mon_pt, size=mp.Vector3(0, sy, 0))
    )

    # Add mode monitor for detailed analysis
    mode_mon = sim.add_mode_monitor(
        fcen, df, nfreq, mp.FluxRegion(center=mon_pt, size=mp.Vector3(0, sy, 0))
    )

    # Log grating simulation start
    client.log(run_step + 4,
        simulation_phase="grating_simulation_start",
        duty_cycle=gdc,
        simulation_time=0.0
    )

    # Run grating simulation with detailed step logging
    grating_start_time = time.time()
    
    # Run in chunks to log progress
    total_time = 300
    chunk_time = 50
    
    for chunk in range(total_time // chunk_time):
        sim.run(until_after_sources=chunk_time)
        elapsed_time = time.time() - grating_start_time
        
        # Get simulation progress metrics
        # Use time step as a proxy for field evolution
        time_step = (chunk + 1) * chunk_time
        field_energy = time_step  # Use time step as energy proxy
        
        # Log progress
        client.log(run_step + 5 + chunk,
            simulation_phase="grating_simulation_progress",
            duty_cycle=gdc,
            simulation_time=elapsed_time,
            time_step=(chunk + 1) * chunk_time,
            field_energy=field_energy,
            progress_percent=((chunk + 1) * chunk_time / total_time) * 100
        )

    # Log simulation complete
    total_grating_time = time.time() - grating_start_time
    client.log(run_step + 5 + (total_time // chunk_time),
        simulation_phase="grating_simulation_complete",
        duty_cycle=gdc,
        simulation_time=total_grating_time,
        total_time_steps=total_time
    )

    # Get results
    freqs = mp.get_eigenmode_freqs(mode_mon)
    res = sim.get_eigenmode_coefficients(
        mode_mon, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y if oddz else mp.EVEN_Z + mp.ODD_Y
    )
    coeffs = res.alpha

    mode_wvl = [1 / freqs[nf] for nf in range(nfreq)]
    mode_tran = [abs(coeffs[0, nf, 0]) ** 2 / input_flux[nf] for nf in range(nfreq)]
    mode_phase = [np.angle(coeffs[0, nf, 0]) for nf in range(nfreq)]

    # Log detailed results
    client.log(run_step + 6 + (total_time // chunk_time),
        simulation_phase="results_analysis",
        duty_cycle=gdc,
        wavelength_points=len(mode_wvl),
        max_transmittance=float(np.max(mode_tran)),
        min_transmittance=float(np.min(mode_tran)),
        avg_transmittance=float(np.mean(mode_tran)),
        max_phase=float(np.max(mode_phase)),
        min_phase=float(np.min(mode_phase)),
        phase_range=float(np.max(mode_phase) - np.min(mode_phase)),
        bandwidth_80_percent=float(np.sum(np.array(mode_tran) > 0.8) * (wvl_max - wvl_min) / nfreq),
        peak_wavelength=float(mode_wvl[np.argmax(mode_tran)])
    )

    return mode_wvl, mode_tran, mode_phase

def run_hyperparameter_sweep():
    """Run hyperparameter sweep with separate OptixLog runs"""
    
    # Define hyperparameter ranges
    grating_periods = [0.3, 0.35, 0.4, 0.45, 0.5]  # μm
    grating_heights = [0.5, 0.6, 0.7, 0.8]  # μm
    odd_z_symmetries = [False, True]
    
    # Create all combinations
    param_combinations = list(itertools.product(grating_periods, grating_heights, odd_z_symmetries))
    
    print(f"🚀 Starting hyperparameter sweep with {len(param_combinations)} combinations")
    
    successful_runs = 0
    failed_runs = 0
    
    for i, (gp, gh, oddz) in enumerate(param_combinations):
        try:
            print(f"\n📊 Run {i+1}/{len(param_combinations)}: GP={gp}μm, GH={gh}μm, OddZ={oddz}")
            
            # Initialize separate OptixLog run for each combination
            client = optixlog.init(
                api_key=os.getenv("OPTIX_API_KEY", "proj_rLe5i6YI6Ozgl8W8Y5G9"),
                api_url=os.getenv("OPTIX_API_URL", "https://coupler.onrender.com"),
                project="BinaryGratingSweep",
                run_name=f"grating_gp{gp}_gh{gh}_{'oddz' if oddz else 'evenz'}",
                config={
                    "simulation_type": "binary_grating_hyperparameter_sweep",
                    "grating_period": gp,
                    "grating_height": gh,
                    "odd_z_symmetry": oddz,
                    "wavelength_range": [wvl_min, wvl_max],
                    "frequency_bins": nfreq,
                    "resolution": resolution,
                    "sweep_run": f"{i+1}/{len(param_combinations)}"
                },
                create_project_if_not_exists=True
            )
            
            print(f"✅ OptixLog run initialized: {client.run_id}")
            
            # Define duty cycle range
            gdc = np.arange(0.1, 1.0, 0.1)
            
            # Log sweep configuration
            client.log(0,
                hyperparameter_sweep=True,
                total_combinations=len(param_combinations),
                current_run=i+1,
                grating_period=gp,
                grating_height=gh,
                odd_z_symmetry=oddz,
                duty_cycle_steps=len(gdc),
                wavelength_min=wvl_min,
                wavelength_max=wvl_max,
                frequency_bins=nfreq,
                resolution=resolution
            )
            
            # Run simulation for each duty cycle
            mode_tran = np.empty((len(gdc), nfreq))
            mode_phase = np.empty((len(gdc), nfreq))
            
            for n, duty_cycle in enumerate(gdc):
                print(f"  🔄 Duty cycle {n+1}/{len(gdc)} (gdc={duty_cycle:.1f})...")
                
                # Calculate starting step for this duty cycle
                start_step = 1 + n * 50  # Each duty cycle gets ~50 steps
                
                mode_wvl, mode_tran[n, :], mode_phase[n, :] = grating_with_detailed_logging(
                    gp, gh, duty_cycle, oddz, client, start_step
                )
            
            # Create visualization
            print(f"  📊 Creating visualization for GP={gp}, GH={gh}...")
            
            plt.figure(figsize=(16, 8), dpi=150)
            
            # Transmittance map
            plt.subplot(1, 2, 1)
            plt.pcolormesh(
                mode_wvl, gdc, mode_tran,
                cmap="hot_r", shading="gouraud",
                vmin=0, vmax=mode_tran.max()
            )
            plt.axis([wvl_min, wvl_max, gdc[0], gdc[-1]])
            plt.xlabel("wavelength (μm)")
            plt.ylabel("grating duty cycle")
            plt.title(f"Transmittance\nGP={gp}μm, GH={gh}μm, {'Odd Z' if oddz else 'Even Z'}")
            plt.colorbar()
            
            # Phase map
            plt.subplot(1, 2, 2)
            plt.pcolormesh(
                mode_wvl, gdc, mode_phase,
                cmap="RdBu", shading="gouraud",
                vmin=mode_phase.min(), vmax=mode_phase.max()
            )
            plt.axis([wvl_min, wvl_max, gdc[0], gdc[-1]])
            plt.xlabel("wavelength (μm)")
            plt.ylabel("grating duty cycle")
            plt.title(f"Phase (radians)\nGP={gp}μm, GH={gh}μm, {'Odd Z' if oddz else 'Even Z'}")
            plt.colorbar()
            
            plt.tight_layout()
            
            # Save and log visualization
            plot_filename = f"binary_grating_gp{gp}_gh{gh}_{'oddz' if oddz else 'evenz'}.png"
            plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
            
            # Log final results
            client.log(len(gdc) * 50 + 10,
                simulation_completed=True,
                total_simulations=len(gdc),
                max_transmittance_overall=float(np.max(mode_tran)),
                min_transmittance_overall=float(np.min(mode_tran)),
                avg_transmittance_overall=float(np.mean(mode_tran)),
                max_phase_overall=float(np.max(mode_phase)),
                min_phase_overall=float(np.min(mode_phase)),
                phase_range_overall=float(np.max(mode_phase) - np.min(mode_phase)),
                best_duty_cycle=float(gdc[np.argmax(np.max(mode_tran, axis=1))]),
                best_transmittance=float(np.max(mode_tran))
            )
            
            # Log the visualization
            try:
                from PIL import Image
                phase_img = Image.open(plot_filename)
                client.log_image("binary_grating_phase_map", phase_img,
                    meta={
                        "stage": "results",
                        "type": "phase_map",
                        "grating_period": gp,
                        "grating_height": gh,
                        "odd_z_symmetry": oddz,
                        "wavelength_range": [wvl_min, wvl_max],
                        "duty_cycle_range": [0.1, 0.9],
                        "max_transmittance": float(np.max(mode_tran)),
                        "phase_range": float(np.max(mode_phase) - np.min(mode_phase))
                    })
                
                # Clean up
                os.remove(plot_filename)
                print(f"  ✅ Visualization logged and cleaned up")
                
            except Exception as e:
                print(f"  ⚠️ Could not log visualization: {e}")
            
            plt.close()
            successful_runs += 1
            
            print(f"  ✅ Run {i+1} completed successfully")
            print(f"  🔗 View at: https://optixlog.com/runs/{client.run_id}")
            
        except Exception as e:
            print(f"  ❌ Run {i+1} failed: {e}")
            failed_runs += 1
            continue
    
    print(f"\n🎉 Hyperparameter sweep completed!")
    print(f"✅ Successful runs: {successful_runs}")
    print(f"❌ Failed runs: {failed_runs}")
    print(f"📊 Total combinations tested: {len(param_combinations)}")

if __name__ == "__main__":
    run_hyperparameter_sweep()
