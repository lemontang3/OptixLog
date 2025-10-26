"""
Straight Waveguide Simulation with OptixLog Integration

This example demonstrates a simple straight waveguide simulation using Meep,
with comprehensive logging of simulation parameters, field data, and results
via OptixLog.

Based on the Meep tutorial: plotting permittivity and fields of a straight waveguide
"""

import os
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
        run_name="straight_waveguide_simulation",
        config={
            "simulation_type": "straight_waveguide",
            "description": "Basic straight waveguide with field visualization",
            "framework": "meep"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")

    # Simulation parameters
    cell = mp.Vector3(16, 8, 0)
    epsilon_wg = 12.0
    frequency = 0.15
    resolution = 10
    simulation_time = 200

    # Log simulation configuration
    client.log(step=0, 
               cell_x=cell.x, cell_y=cell.y, cell_z=cell.z,
               epsilon_waveguide=epsilon_wg,
               frequency=frequency,
               resolution=resolution,
               simulation_time=simulation_time)

    # Geometry definition
    geometry = [
        mp.Block(
            mp.Vector3(mp.inf, 1, mp.inf),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=epsilon_wg),
        )
    ]

    # Source definition
    sources = [
        mp.Source(
            mp.ContinuousSource(frequency=frequency), 
            component=mp.Ez, 
            center=mp.Vector3(-7, 0)
        )
    ]

    # PML layers
    pml_layers = [mp.PML(1.0)]

    # Create simulation
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
    )

    print("🏃 Running simulation...")
    sim.run(until=simulation_time)

    # Get field data
    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

    # Calculate field statistics
    max_ez = float(np.max(np.abs(ez_data)))
    mean_ez = float(np.mean(np.abs(ez_data)))
    field_energy = float(np.sum(np.abs(ez_data)**2))

    # Log field statistics
    client.log(step=1,
               max_field_amplitude=max_ez,
               mean_field_amplitude=mean_ez,
               total_field_energy=field_energy,
               simulation_completed=True)

    # Generate and save plots
    print("📊 Generating field visualizations...")
    
    # Permittivity plot
    plt.figure(figsize=(10, 5))
    plt.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary")
    plt.title("Waveguide Permittivity Distribution")
    plt.colorbar(label="Relative Permittivity")
    plt.axis("off")
    eps_plot_path = "waveguide_permittivity.png"
    plt.savefig(eps_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Field plot
    plt.figure(figsize=(10, 5))
    plt.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary")
    plt.imshow(ez_data.transpose(), interpolation="spline36", cmap="RdBu", alpha=0.9)
    plt.title("Electric Field Ez in Waveguide")
    plt.colorbar(label="Field Amplitude")
    plt.axis("off")
    field_plot_path = "waveguide_field.png"
    plt.savefig(field_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Log the generated plots
    client.log_file("permittivity_plot", eps_plot_path, "image/png", 
                   meta={"description": "Waveguide permittivity distribution", "component": "permittivity"})
    client.log_file("field_plot", field_plot_path, "image/png",
                   meta={"description": "Electric field Ez distribution", "component": "Ez"})

    print("✅ Simulation completed successfully!")
    print(f"📈 Logged {2} metrics and {2} artifacts to OptixLog")

except ValueError as e:
    print(f"\n❌ OptixLog Error: {e}")
    print("Please ensure your API key and URL are correct.")
except Exception as e:
    print(f"\n❌ Simulation Error: {e}")

finally:
    # Clean up generated files
    for file_path in ["waveguide_permittivity.png", "waveguide_field.png"]:
        if os.path.exists(file_path):
            os.remove(file_path)
