"""
Waveguide Bend Transmission Analysis with OptixLog Integration

This example demonstrates transmission analysis around a 90-degree waveguide bend,
comparing straight and bent waveguide configurations with comprehensive logging
via OptixLog.

Based on the Meep tutorial: transmission around a 90-degree waveguide bend in 2D
"""

import os
import optixlog
import meep as mp
import matplotlib.pyplot as plt
import numpy as np

# OptixLog Configuration
api_key = os.getenv("OPTIX_API_KEY", "proj_YOUR_API_KEY_HERE")
api_url = os.getenv("OPTIX_API_URL", "https://optixlog.com")
project_name = os.getenv("OPTIX_PROJECT", "MeepExamples")

print(f"🚀 Initializing OptixLog client for project: {project_name}")

try:
    # Initialize OptixLog client
    client = optixlog.init(
        api_key=api_key,
        api_url=api_url,
        project=project_name,
        run_name="waveguide_bend_transmission",
        config={
            "simulation_type": "waveguide_bend",
            "description": "90-degree waveguide bend transmission analysis",
            "framework": "meep"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")

    # Simulation parameters
    resolution = 10  # pixels/um
    sx = 16  # size of cell in X direction
    sy = 32  # size of cell in Y direction
    dpml = 1.0
    pad = 4  # padding distance between waveguide and cell edge
    w = 1  # width of waveguide
    epsilon_wg = 12.0
    
    fcen = 0.15  # pulse center frequency
    df = 0.1  # pulse width (in frequency)
    nfreq = 100  # number of frequencies at which to compute flux

    # Calculate waveguide positions
    wvg_xcen = 0.5 * (sx - w - 2 * pad)  # x center of vert. wvg
    wvg_ycen = -0.5 * (sy - w - 2 * pad)  # y center of horiz. wvg

    # Log simulation configuration
    client.log(step=0,
               resolution=resolution,
               cell_size_x=sx, cell_size_y=sy,
               pml_thickness=dpml,
               waveguide_width=w,
               waveguide_epsilon=epsilon_wg,
               center_frequency=fcen,
               frequency_width=df,
               num_frequencies=nfreq,
               waveguide_x_center=wvg_xcen,
               waveguide_y_center=wvg_ycen)

    cell = mp.Vector3(sx, sy, 0)
    pml_layers = [mp.PML(dpml)]

    # Straight waveguide geometry
    straight_geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf, w, mp.inf),
            center=mp.Vector3(0, wvg_ycen, 0),
            material=mp.Medium(epsilon=epsilon_wg),
        )
    ]

    # Source definition
    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(-0.5 * sx + dpml, wvg_ycen, 0),
            size=mp.Vector3(0, w, 0),
        )
    ]

    # Create straight waveguide simulation
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=straight_geometry,
        sources=sources,
        resolution=resolution,
    )

    # Flux regions for straight waveguide
    refl_fr = mp.FluxRegion(
        center=mp.Vector3(-0.5 * sx + dpml + 0.5, wvg_ycen, 0), 
        size=mp.Vector3(0, 2 * w, 0)
    )
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)

    tran_fr = mp.FluxRegion(
        center=mp.Vector3(0.5 * sx - dpml, wvg_ycen, 0), 
        size=mp.Vector3(0, 2 * w, 0)
    )
    tran = sim.add_flux(fcen, df, nfreq, tran_fr)

    pt = mp.Vector3(0.5 * sx - dpml - 0.5, wvg_ycen)

    print("🏃 Running straight waveguide simulation...")
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))

    # Get reference data for normalization
    straight_refl_data = sim.get_flux_data(refl)
    straight_tran_flux = mp.get_fluxes(tran)

    client.log(step=1,
               simulation_phase="straight_waveguide",
               max_transmission_flux=max(straight_tran_flux),
               min_transmission_flux=min(straight_tran_flux),
               mean_transmission_flux=np.mean(straight_tran_flux))

    sim.reset_meep()

    # Bent waveguide geometry
    bent_geometry = [
        mp.Block(
            mp.Vector3(sx - pad, w, mp.inf),
            center=mp.Vector3(-0.5 * pad, wvg_ycen),
            material=mp.Medium(epsilon=epsilon_wg),
        ),
        mp.Block(
            mp.Vector3(w, sy - pad, mp.inf),
            center=mp.Vector3(wvg_xcen, 0.5 * pad),
            material=mp.Medium(epsilon=epsilon_wg),
        ),
    ]

    # Create bent waveguide simulation
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=bent_geometry,
        sources=sources,
        resolution=resolution,
    )

    # Flux regions for bent waveguide
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)

    tran_fr = mp.FluxRegion(
        center=mp.Vector3(wvg_xcen, 0.5 * sy - dpml - 0.5, 0), 
        size=mp.Vector3(2 * w, 0, 0)
    )
    tran = sim.add_flux(fcen, df, nfreq, tran_fr)

    # Load reference data to subtract incident field
    sim.load_minus_flux_data(refl, straight_refl_data)

    pt = mp.Vector3(wvg_xcen, 0.5 * sy - dpml - 0.5)

    print("🏃 Running bent waveguide simulation...")
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))

    # Get transmission results
    bend_refl_flux = mp.get_fluxes(refl)
    bend_tran_flux = mp.get_fluxes(tran)
    flux_freqs = mp.get_flux_freqs(refl)

    # Calculate transmission and reflection coefficients
    wavelengths = []
    reflectances = []
    transmittances = []
    losses = []

    for i in range(nfreq):
        wl = 1 / flux_freqs[i]
        R = -bend_refl_flux[i] / straight_tran_flux[i]
        T = bend_tran_flux[i] / straight_tran_flux[i]
        L = 1 - R - T
        
        wavelengths.append(wl)
        reflectances.append(R)
        transmittances.append(T)
        losses.append(L)

    # Log transmission results
    client.log(step=2,
               simulation_phase="bent_waveguide",
               max_transmittance=max(transmittances),
               min_transmittance=min(transmittances),
               mean_transmittance=np.mean(transmittances),
               max_reflectance=max(reflectances),
               max_loss=max(losses),
               wavelength_range_min=min(wavelengths),
               wavelength_range_max=max(wavelengths))

    # Generate transmission plot
    print("📊 Generating transmission plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, reflectances, "bo-", label="Reflectance", linewidth=2)
    plt.plot(wavelengths, transmittances, "ro-", label="Transmittance", linewidth=2)
    plt.plot(wavelengths, losses, "go-", label="Loss", linewidth=2)
    plt.axis([5.0, 10.0, 0, 1])
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Transmission/Reflection Coefficient")
    plt.title("Waveguide Bend Transmission Analysis")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    plot_path = "bend_transmission_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Log the transmission plot
    client.log_file("transmission_plot", plot_path, "image/png",
                   meta={"description": "Waveguide bend transmission vs wavelength", 
                         "x_axis": "wavelength_um", 
                         "y_axis": "transmission_coefficient"})

    # Save transmission data as CSV
    transmission_data = np.column_stack([wavelengths, transmittances, reflectances, losses])
    csv_path = "bend_transmission_data.csv"
    np.savetxt(csv_path, transmission_data, 
               header="wavelength_um,transmittance,reflectance,loss", 
               delimiter=",", fmt="%.6f")

    client.log_file("transmission_data", csv_path, "text/csv",
                   meta={"description": "Waveguide bend transmission data", 
                         "columns": ["wavelength_um", "transmittance", "reflectance", "loss"]})

    print("✅ Waveguide bend analysis completed successfully!")
    print(f"📈 Logged {3} metric steps and {2} artifacts to OptixLog")
    print(f"📊 Peak transmittance: {max(transmittances):.3f}")
    print(f"📊 Peak reflectance: {max(reflectances):.3f}")

except ValueError as e:
    print(f"\n❌ OptixLog Error: {e}")
    print("Please ensure your API key and URL are correct.")
except Exception as e:
    print(f"\n❌ Simulation Error: {e}")

finally:
    # Clean up generated files
    for file_path in ["bend_transmission_analysis.png", "bend_transmission_data.csv"]:
        if os.path.exists(file_path):
            os.remove(file_path)
