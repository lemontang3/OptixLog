"""
Binary Grating Diffraction Analysis with OptixLog Integration

This example demonstrates binary grating diffraction analysis with comprehensive
logging of grating parameters, diffraction orders, and transmission spectra
via OptixLog.

Based on the Meep tutorial: binary grating diffraction analysis
"""

import os
import math
import optixlog
import meep as mp
import matplotlib.pyplot as plt
import numpy as np

# OptixLog Configuration
api_key = os.getenv("OPTIX_API_KEY", "proj_YOUR_API_KEY_HERE")
api_url = os.getenv("OPTIX_API_URL", "https://coupler.onrender.com")
project_name = os.getenv("OPTIX_PROJECT", "MeepExamples")

print(f"🚀 Initializing OptixLog client for project: {project_name}")

try:
    # Initialize OptixLog client
    client = optixlog.init(
        api_key=api_key,
        api_url=api_url,
        project=project_name,
        run_name="binary_grating_diffraction",
        config={
            "simulation_type": "binary_grating",
            "description": "Binary grating diffraction order analysis",
            "framework": "meep"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")

    # Grating parameters
    resolution = 60  # pixels/μm
    dpml = 1.0  # PML thickness
    dsub = 3.0  # substrate thickness
    dpad = 3.0  # padding between grating and PML
    gp = 10.0  # grating period
    gh = 0.5  # grating height
    gdc = 0.5  # grating duty cycle
    glass_index = 1.5

    # Wavelength range
    wvl_min = 0.4  # min wavelength
    wvl_max = 0.6  # max wavelength
    fmin = 1 / wvl_max  # min frequency
    fmax = 1 / wvl_min  # max frequency
    fcen = 0.5 * (fmin + fmax)  # center frequency
    df = fmax - fmin  # frequency width

    # Cell size
    sx = dpml + dsub + gh + dpad + dpml
    sy = gp
    cell_size = mp.Vector3(sx, sy, 0)
    
    nfreq = 21
    nmode = 10

    # Log simulation configuration
    client.log(step=0,
               resolution=resolution,
               grating_period=gp,
               grating_height=gh,
               grating_duty_cycle=gdc,
               substrate_thickness=dsub,
               glass_index=glass_index,
               wavelength_min=wvl_min,
               wavelength_max=wvl_max,
               center_frequency=fcen,
               num_frequencies=nfreq,
               num_modes=nmode,
               cell_size_x=sx, cell_size_y=sy)

    pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]
    glass = mp.Medium(index=glass_index)
    symmetries = [mp.Mirror(mp.Y)]

    # Source definition
    src_pt = mp.Vector3(-0.5 * sx + dpml + 0.5 * dsub, 0, 0)
    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=src_pt,
            size=mp.Vector3(0, sy, 0),
        )
    ]

    # First simulation: empty cell for normalization
    print("🏃 Running empty cell simulation for normalization...")
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        k_point=mp.Vector3(0, 0, 0),
        default_material=glass,
        sources=sources,
        symmetries=symmetries,
    )

    mon_pt = mp.Vector3(0.5 * sx - dpml - 0.5 * dpad, 0, 0)
    flux_mon = sim.add_flux(
        fcen, df, nfreq, mp.FluxRegion(center=mon_pt, size=mp.Vector3(0, sy, 0))
    )

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-9))
    input_flux = mp.get_fluxes(flux_mon)

    client.log(step=1,
               simulation_phase="empty_cell",
               max_input_flux=max(input_flux),
               min_input_flux=min(input_flux),
               mean_input_flux=np.mean(input_flux))

    sim.reset_meep()

    # Second simulation: with grating
    print("🏃 Running grating simulation...")
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
        k_point=mp.Vector3(0, 0, 0),
        sources=sources,
        symmetries=symmetries,
    )

    mode_mon = sim.add_flux(
        fcen, df, nfreq, mp.FluxRegion(center=mon_pt, size=mp.Vector3(0, sy, 0))
    )

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-9))

    # Get eigenmode coefficients
    freqs = mp.get_eigenmode_freqs(mode_mon)
    res = sim.get_eigenmode_coefficients(
        mode_mon, range(1, nmode + 1), eig_parity=mp.ODD_Z + mp.EVEN_Y
    )
    coeffs = res.alpha
    kdom = res.kdom

    # Process diffraction data
    mode_wvl = []
    mode_angle = []
    mode_tran = []

    print("📊 Processing diffraction orders...")
    for nm in range(nmode):
        for nf in range(nfreq):
            wvl = 1 / freqs[nf]
            angle = math.degrees(math.acos(kdom[nm * nfreq + nf].x / freqs[nf]))
            tran = abs(coeffs[nm, nf, 0]) ** 2 / input_flux[nf]
            tran = 0.5 * tran if nm != 0 else tran
            
            mode_wvl.append(wvl)
            mode_angle.append(angle)
            mode_tran.append(tran)

    # Log diffraction analysis results
    client.log(step=2,
               simulation_phase="grating_analysis",
               num_diffraction_orders=nmode,
               max_transmission=max(mode_tran),
               min_transmission=min(mode_tran),
               mean_transmission=np.mean(mode_tran),
               angle_range_min=min(mode_angle),
               angle_range_max=max(mode_angle),
               wavelength_range_min=min(mode_wvl),
               wavelength_range_max=max(mode_wvl))

    # Generate diffraction plot
    print("📊 Generating diffraction order plot...")
    tran_max = round(max(mode_tran), 1)
    
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(
        np.reshape(mode_wvl, (nmode, nfreq)),
        np.reshape(mode_angle, (nmode, nfreq)),
        np.reshape(mode_tran, (nmode, nfreq)),
        cmap="Blues",
        shading="nearest",
        vmin=0,
        vmax=tran_max,
    )
    plt.axis([min(mode_wvl), max(mode_wvl), min(mode_angle), max(mode_angle)])
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Diffraction Angle (degrees)")
    plt.xticks(list(np.arange(0.4, 0.7, 0.1)))
    plt.yticks(list(range(0, 35, 5)))
    plt.title("Binary Grating: Transmittance of Diffraction Orders")
    cbar = plt.colorbar()
    cbar.set_ticks(list(np.arange(0, tran_max + 0.1, 0.1)))
    cbar.set_ticklabels([f"{t:.1f}" for t in np.arange(0, tran_max + 0.1, 0.1)])
    cbar.set_label("Transmittance")
    plt.grid(True, alpha=0.3)
    
    plot_path = "binary_grating_diffraction.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Log the diffraction plot
    client.log_file("diffraction_plot", plot_path, "image/png",
                   meta={"description": "Binary grating diffraction order transmittance", 
                         "x_axis": "wavelength_um", 
                         "y_axis": "diffraction_angle_degrees",
                         "colorbar": "transmittance"})

    # Save diffraction data
    diffraction_data = np.column_stack([mode_wvl, mode_angle, mode_tran])
    csv_path = "binary_grating_data.csv"
    np.savetxt(csv_path, diffraction_data, 
               header="wavelength_um,diffraction_angle_degrees,transmittance", 
               delimiter=",", fmt="%.6f")

    client.log_file("diffraction_data", csv_path, "text/csv",
                   meta={"description": "Binary grating diffraction data", 
                         "columns": ["wavelength_um", "diffraction_angle_degrees", "transmittance"]})

    print("✅ Binary grating analysis completed successfully!")
    print(f"📈 Logged {3} metric steps and {2} artifacts to OptixLog")
    print(f"📊 Peak transmittance: {max(mode_tran):.3f}")
    print(f"📊 Diffraction angle range: {min(mode_angle):.1f}° to {max(mode_angle):.1f}°")

except ValueError as e:
    print(f"\n❌ OptixLog Error: {e}")
    print("Please ensure your API key and URL are correct.")
except Exception as e:
    print(f"\n❌ Simulation Error: {e}")

finally:
    # Clean up generated files
    for file_path in ["binary_grating_diffraction.png", "binary_grating_data.csv"]:
        if os.path.exists(file_path):
            os.remove(file_path)
