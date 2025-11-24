"""
Binary Grating Hyperparameter Sweep with OptixLog SDK v0.0.4

This script performs a comprehensive hyperparameter sweep of binary grating simulations,
creating separate OptixLog runs for each parameter combination.

NEW in v0.0.4:
✓ Context managers - auto-cleanup
✓ log_matplotlib() - one-line plotting
✓ Return values with URLs
✓ Colored console output
✓ 80% less boilerplate!

Usage:
    export OPTIX_API_KEY="proj_your_api_key_here"
    python binary_grating_hyperparameter_sweep.py
"""

import argparse
import sys
import os
import time
import itertools

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib

import meep as mp
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

    # NEW in v0.0.4: Log with return values
    result = client.log(run_step,
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

    # Run reference simulation
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ez if oddz else mp.Hz, mon_pt, 1e-9
        )
    )

    # Log reference simulation completion
    client.log(run_step + 10,
        simulation_phase="reference_complete",
        simulation_time=sim.meep_time(),
        flux_computed=True
    )

    freqs = mp.get_flux_freqs(flux_mon)
    input_flux = mp.get_fluxes(flux_mon)

    # Save flux data
    sim.reset_meep()

    # Create simulation with grating
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

    mode_mon = sim.add_flux(
        fcen, df, nfreq, mp.FluxRegion(center=mon_pt, size=mp.Vector3(0, sy, 0))
    )

    sim.load_minus_flux_data(mode_mon, input_flux)

    # Log grating simulation start
    client.log(run_step + 20,
        simulation_phase="grating_start",
        geometry_defined=True,
        num_blocks=len(geometry)
    )

    # Run grating simulation
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ez if oddz else mp.Hz, mon_pt, 1e-9
        )
    )

    # Log grating simulation completion
    client.log(run_step + 30,
        simulation_phase="grating_complete",
        simulation_time=sim.meep_time()
    )

    # Calculate transmission and phase
    mode_tran = mp.get_fluxes(mode_mon)
    mode_tran = np.asarray(mode_tran)
    mode_tran /= np.asarray(input_flux)

    angles = [np.angle(m) for m in mp.get_flux_data(mode_mon)]
    mode_phase = np.unwrap(angles)

    # Log final results for this duty cycle
    client.log(run_step + 40,
        simulation_phase="analysis_complete",
        max_transmittance=float(np.max(mode_tran)),
        min_transmittance=float(np.min(mode_tran)),
        avg_transmittance=float(np.mean(mode_tran)),
        phase_range=float(np.max(mode_phase) - np.min(mode_phase))
    )

    mode_wvl = 1 / freqs

    return mode_wvl, mode_tran, mode_phase


def run_hyperparameter_sweep():
    """Run hyperparameter sweep with v0.0.4 features"""
    
    # Check API key
    api_key = os.getenv("OPTIX_API_KEY")
    if not api_key:
        print("❌ Error: OPTIX_API_KEY environment variable not set!")
        print("   Set it with: export OPTIX_API_KEY='your-api-key'")
        return
    
    # Define parameter grid
    gp_values = [0.6]  # grating period (μm)
    gh_values = [0.5]  # grating height (μm)
    oddz_values = [True, False]  # z-symmetry
    
    # Duty cycles to sweep
    gdc = np.arange(0.1, 1.0, 0.1)  # duty cycle sweep
    
    # Create all combinations
    param_combinations = list(itertools.product(gp_values, gh_values, oddz_values))
    
    print("=" * 70)
    print("Binary Grating Hyperparameter Sweep - OptixLog SDK v0.0.4")
    print("=" * 70)
    print(f"\nParameter Grid:")
    print(f"  Grating Periods: {gp_values}")
    print(f"  Grating Heights: {gh_values}")
    print(f"  Z-Symmetries: {oddz_values}")
    print(f"  Duty Cycles: {len(gdc)} values from {gdc[0]:.1f} to {gdc[-1]:.1f}")
    print(f"\nTotal Combinations: {len(param_combinations)}")
    print(f"Simulations per combination: {len(gdc)}")
    print(f"Total simulations: {len(param_combinations) * len(gdc)}")
    print("=" * 70)
    
    successful_runs = 0
    failed_runs = 0
    
    for i, (gp, gh, oddz) in enumerate(param_combinations):
        print(f"\n{'='*70}")
        print(f"Run {i+1}/{len(param_combinations)}")
        print(f"Parameters: GP={gp}μm, GH={gh}μm, Z-symmetry={'Odd' if oddz else 'Even'}")
        print(f"{'='*70}")
        
        try:
            # NEW in v0.0.4: Context manager!
            with optixlog.run(
                run_name=f"binary_grating_GP{gp}_GH{gh}_{'oddz' if oddz else 'evenz'}",
                project="BinaryGratingSweep",
                config={
                    "grating_period_um": gp,
                    "grating_height_um": gh,
                    "odd_z_symmetry": oddz,
                    "duty_cycle_min": 0.1,
                    "duty_cycle_max": 0.9,
                    "duty_cycle_steps": len(gdc),
                    "resolution": resolution,
                    "wavelength_range_um": [wvl_min, wvl_max],
                    "num_frequencies": nfreq,
                    "sdk_version": "0.0.4"
                },
                create_project_if_not_exists=True
            ) as client:
                
                print(f"✅ OptixLog run initialized: {client.run_id}")
                
                # Run simulation for each duty cycle
                mode_tran = np.empty((len(gdc), nfreq))
                mode_phase = np.empty((len(gdc), nfreq))
                
                for n, duty_cycle in enumerate(gdc):
                    print(f"  🔄 Duty cycle {n+1}/{len(gdc)} (gdc={duty_cycle:.1f})...")
                    
                    start_step = 1 + n * 50
                    
                    mode_wvl, mode_tran[n, :], mode_phase[n, :] = grating_with_detailed_logging(
                        gp, gh, duty_cycle, oddz, client, start_step
                    )
                
                # Create visualization
                print(f"  📊 Creating visualization for GP={gp}, GH={gh}...")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                # Transmittance map
                im1 = ax1.pcolormesh(
                    mode_wvl, gdc, mode_tran,
                    cmap="hot_r", shading="gouraud",
                    vmin=0, vmax=mode_tran.max()
                )
                ax1.axis([wvl_min, wvl_max, gdc[0], gdc[-1]])
                ax1.set_xlabel("wavelength (μm)")
                ax1.set_ylabel("grating duty cycle")
                ax1.set_title(f"Transmittance\nGP={gp}μm, GH={gh}μm, {'Odd Z' if oddz else 'Even Z'}")
                plt.colorbar(im1, ax=ax1)
                
                # Phase map
                im2 = ax2.pcolormesh(
                    mode_wvl, gdc, mode_phase,
                    cmap="RdBu", shading="gouraud",
                    vmin=mode_phase.min(), vmax=mode_phase.max()
                )
                ax2.axis([wvl_min, wvl_max, gdc[0], gdc[-1]])
                ax2.set_xlabel("wavelength (μm)")
                ax2.set_ylabel("grating duty cycle")
                ax2.set_title(f"Phase (radians)\nGP={gp}μm, GH={gh}μm, {'Odd Z' if oddz else 'Even Z'}")
                plt.colorbar(im2, ax=ax2)
                
                plt.tight_layout()
                
                # NEW in v0.0.4: One line instead of 10!
                viz_result = client.log_matplotlib("binary_grating_phase_map", fig,
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
                
                if viz_result and viz_result.success:
                    print(f"  ✅ Visualization logged: {viz_result.url}")
                
                plt.close(fig)
                
                # Log final summary
                summary_result = client.log(len(gdc) * 50 + 10,
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
                
                successful_runs += 1
                
                print(f"  ✅ Run {i+1} completed successfully")
                print(f"  🔗 View at: https://optixlog.com/runs/{client.run_id}")
            
            # Context manager automatically cleans up!
            
        except Exception as e:
            print(f"  ❌ Run {i+1} failed: {e}")
            failed_runs += 1
            continue
    
    print(f"\n{'='*70}")
    print(f"🎉 Hyperparameter sweep completed!")
    print(f"✅ Successful runs: {successful_runs}")
    print(f"❌ Failed runs: {failed_runs}")
    print(f"📊 Total combinations tested: {len(param_combinations)}")
    print(f"\n🎊 SDK v0.0.4 Features Used:")
    print(f"  ✓ Context managers - automatic cleanup")
    print(f"  ✓ log_matplotlib() - zero boilerplate plotting")
    print(f"  ✓ Return values - immediate feedback")
    print(f"  ✓ Colored output - beautiful terminal")
    print(f"  Boilerplate reduction: ~80%!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    run_hyperparameter_sweep()
