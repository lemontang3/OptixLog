"""
Binary Grating Phase Map Simulation with OptixLog Integration

This script simulates binary grating transmission and phase characteristics using Meep
and logs the results to OptixLog for visualization and tracking.

Usage:
    # Set your OptixLog API key
    export OPTIX_API_KEY="proj_your_api_key_here"
    
    # Run the simulation
    python binary_grating_phasemap.py                    # Default parameters
    python binary_grating_phasemap.py -gp 0.4 -gh 0.7    # Custom grating parameters
    python binary_grating_phasemap.py -oddz              # Odd Z symmetry

The script will:
1. Initialize a new OptixLog run in the "Examples" project
2. Log simulation parameters (grating geometry, wavelength range, etc.)
3. Run grating simulations across duty cycle range
4. Log transmission and phase data for each simulation
5. Generate and log phase map visualizations
6. Provide a link to view results in OptixLog dashboard

If OptixLog is not available, the script will run normally without logging.
"""

import argparse
import sys
import os

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


def grating(gp, gh, gdc, oddz):
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

    sim.run(until_after_sources=100)

    input_flux = mp.get_fluxes(flux_mon)

    sim.reset_meep()

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

    sim.run(until_after_sources=300)

    freqs = mp.get_eigenmode_freqs(mode_mon)
    res = sim.get_eigenmode_coefficients(
        mode_mon, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y if oddz else mp.EVEN_Z + mp.ODD_Y
    )
    coeffs = res.alpha

    mode_wvl = [1 / freqs[nf] for nf in range(nfreq)]
    mode_tran = [abs(coeffs[0, nf, 0]) ** 2 / input_flux[nf] for nf in range(nfreq)]
    mode_phase = [np.angle(coeffs[0, nf, 0]) for nf in range(nfreq)]

    return mode_wvl, mode_tran, mode_phase


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Binary Grating Phase Map Simulation with OptixLog Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python binary_grating_phasemap.py              # Default parameters
  python binary_grating_phasemap.py -gp 0.4 -gh 0.7  # Custom grating parameters
  python binary_grating_phasemap.py -oddz        # Odd Z symmetry

Environment Variables:
  OPTIX_API_KEY                                  # Your OptixLog API key for logging results
        """
    )
    parser.add_argument(
        "-gp", type=float, default=0.35, help="grating periodicity (default: 0.35 μm)"
    )
    parser.add_argument(
        "-gh", type=float, default=0.6, help="grating height (default: 0.6 μm)"
    )
    parser.add_argument(
        "-oddz", action="store_true", default=False, help="use odd Z symmetry (default: False)"
    )
    args = parser.parse_args()

    # Initialize OptixLog client
    try:
        config = {
            "simulation_type": "binary_grating_phasemap",
            "grating_period": args.gp,
            "grating_height": args.gh,
            "odd_z_symmetry": args.oddz,
            "wavelength_range": [wvl_min, wvl_max],
            "frequency_bins": nfreq,
            "resolution": resolution,
            "material": "glass",
            "duty_cycle_range": [0.1, 0.9]
        }

        client = optixlog.init(
            api_key=os.getenv("OPTIX_API_KEY", "proj_rLe5i6YI6Ozgl8W8Y5G9"),
            api_url=os.getenv("OPTIX_API_URL", "https://coupler.onrender.com"),
            project="Examples",
            run_name=f"binary_grating_gp{args.gp}_gh{args.gh}_{'oddz' if args.oddz else 'evenz'}",
            config=config,
            create_project_if_not_exists=True
        )

        print(f"[OptixLog] Started run: binary_grating_gp{args.gp}_gh{args.gh}_{'oddz' if args.oddz else 'evenz'}")
        print(f"[OptixLog] Project: Examples")
        print(f"[OptixLog] Run ID: {client.run_id}")
        optixlog_enabled = True
    except Exception as e:
        print(f"[Warning] OptixLog initialization failed: {e}")
        print("[Info] Continuing without OptixLog...")
        client = None
        optixlog_enabled = False

    # Define duty cycle range
    gdc = np.arange(0.1, 1.0, 0.1)

    # Log simulation setup
    if optixlog_enabled:
        client.log(0,
            grating_period=args.gp,
            grating_height=args.gh,
            odd_z_symmetry=args.oddz,
            wavelength_min=wvl_min,
            wavelength_max=wvl_max,
            frequency_bins=nfreq,
            resolution=resolution,
            duty_cycle_steps=gdc.size,
            material="glass",
            dpml_thickness=dpml,
            substrate_thickness=dsub,
            padding_thickness=dpad
        )

    print(f"[Simulation] Starting binary grating phase map simulation...")
    print(f"[Simulation] Grating period: {args.gp} μm, Height: {args.gh} μm")
    print(f"[Simulation] Symmetry: {'Odd Z' if args.oddz else 'Even Z'}")
    print(f"[Simulation] Wavelength range: {wvl_min} - {wvl_max} μm")
    mode_tran = np.empty((gdc.size, nfreq))
    mode_phase = np.empty((gdc.size, nfreq))
    
    for n in range(gdc.size):
        print(f"[Simulation] Running duty cycle {n+1}/{gdc.size} (gdc={gdc[n]:.1f})...")
        mode_wvl, mode_tran[n, :], mode_phase[n, :] = grating(
            args.gp, args.gh, gdc[n], args.oddz
        )
        
        # Log results for each duty cycle
        if optixlog_enabled:
            client.log(n + 1,
                duty_cycle=gdc[n],
                max_transmittance=float(np.max(mode_tran[n, :])),
                min_transmittance=float(np.min(mode_tran[n, :])),
                avg_transmittance=float(np.mean(mode_tran[n, :])),
                max_phase=float(np.max(mode_phase[n, :])),
                min_phase=float(np.min(mode_phase[n, :])),
                phase_range=float(np.max(mode_phase[n, :]) - np.min(mode_phase[n, :]))
            )

    # Create comprehensive phase map visualization
    print(f"[Visualization] Creating phase map plots...")
    
    plt.figure(figsize=(16, 8), dpi=150)

    # Transmittance map
    plt.subplot(1, 2, 1)
    plt.pcolormesh(
        mode_wvl,
        gdc,
        mode_tran,
        cmap="hot_r",
        shading="gouraud",
        vmin=0,
        vmax=mode_tran.max(),
    )
    plt.axis([wvl_min, wvl_max, gdc[0], gdc[-1]])
    plt.xlabel("wavelength (μm)")
    plt.xticks(list(np.arange(wvl_min, wvl_max + 0.1, 0.1)))
    plt.ylabel("grating duty cycle")
    plt.yticks(list(np.arange(gdc[0], gdc[-1] + 0.1, 0.1)))
    plt.title(f"Transmittance\nGP={args.gp}μm, GH={args.gh}μm, {'Odd Z' if args.oddz else 'Even Z'}")
    cbar = plt.colorbar()
    cbar.set_ticks(list(np.arange(0, 1.2, 0.2)))
    cbar.set_ticklabels([f"{t:.1f}" for t in np.arange(0, 1.2, 0.2)])

    # Phase map
    plt.subplot(1, 2, 2)
    plt.pcolormesh(
        mode_wvl,
        gdc,
        mode_phase,
        cmap="RdBu",
        shading="gouraud",
        vmin=mode_phase.min(),
        vmax=mode_phase.max(),
    )
    plt.axis([wvl_min, wvl_max, gdc[0], gdc[-1]])
    plt.xlabel("wavelength (μm)")
    plt.xticks(list(np.arange(wvl_min, wvl_max + 0.1, 0.1)))
    plt.ylabel("grating duty cycle")
    plt.yticks(list(np.arange(gdc[0], gdc[-1] + 0.1, 0.1)))
    plt.title(f"Phase (radians)\nGP={args.gp}μm, GH={args.gh}μm, {'Odd Z' if args.oddz else 'Even Z'}")
    cbar = plt.colorbar()
    cbar.set_ticks(list(range(-3, 4)))
    cbar.set_ticklabels([f"{t:.1f}" for t in range(-3, 4)])

    plt.tight_layout()
    plt.savefig("binary_grating_phasemap.png", dpi=150, bbox_inches="tight")

    # Log final results and visualization
    if optixlog_enabled:
        # Log overall statistics
        client.log(gdc.size + 1,
            total_simulations=gdc.size,
            max_transmittance_overall=float(np.max(mode_tran)),
            min_transmittance_overall=float(np.min(mode_tran)),
            avg_transmittance_overall=float(np.mean(mode_tran)),
            max_phase_overall=float(np.max(mode_phase)),
            min_phase_overall=float(np.min(mode_phase)),
            phase_range_overall=float(np.max(mode_phase) - np.min(mode_phase)),
            simulation_completed=True
        )

        # Log the phase map visualization
        from PIL import Image
        phase_img = Image.open("binary_grating_phasemap.png")
        client.log_image("binary_grating_phasemap", phase_img,
            meta={
                "stage": "results",
                "type": "phase_map",
                "grating_period": args.gp,
                "grating_height": args.gh,
                "odd_z_symmetry": args.oddz,
                "wavelength_range": [wvl_min, wvl_max],
                "duty_cycle_range": [0.1, 0.9],
                "max_transmittance": float(np.max(mode_tran)),
                "phase_range": float(np.max(mode_phase) - np.min(mode_phase))
            })

        # Clean up temporary file
        os.remove("binary_grating_phasemap.png")

        print(f"[OptixLog] Logged simulation results to run {client.run_id}")
        print(f"[OptixLog] Logged visualization to run {client.run_id}")
        print(f"[OptixLog] View results at: https://optixlog.com/runs/{client.run_id}")
    else:
        print("[Info] Simulation completed successfully (without OptixLog)")
        plt.show()