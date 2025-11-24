"""
Straight Waveguide Simulation with OptixLog

This example demonstrates a simple straight waveguide simulation using Meep,
with comprehensive logging of simulation parameters, field data, and results.

Features demonstrated:
- Context managers for automatic cleanup
- log_matplotlib() for one-line plot logging
- log_array_as_image() for direct array visualization
- Return values with URLs
- Colored console output

Based on the Meep tutorial: plotting permittivity and fields of a straight waveguide
"""

import os
import optixlog
import meep as mp
import matplotlib.pyplot as plt
import numpy as np

# OptixLog Configuration
api_key = os.getenv("OPTIX_API_KEY", "proj_YOUR_API_KEY_HERE")
project_name = os.getenv("OPTIX_PROJECT", "MeepExamples")

print(f"🚀 Starting straight waveguide simulation with OptixLog")

try:
    # Use context manager for automatic cleanup
    with optixlog.run(
        run_name="straight_waveguide_simulation",
        project=project_name,
        config={
            "simulation_type": "straight_waveguide",
            "description": "Basic straight waveguide with field visualization",
            "framework": "meep"
        },
        create_project_if_not_exists=True
    ) as client:
        
        print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
        
        # Simulation parameters
        cell = mp.Vector3(16, 8, 0)
        epsilon_wg = 12.0
        frequency = 0.15
        resolution = 10
        simulation_time = 200
        
        # Log with return values
        result = client.log(step=0, 
                   cell_x=cell.x, cell_y=cell.y, cell_z=cell.z,
                   epsilon_waveguide=epsilon_wg,
                   frequency=frequency,
                   resolution=resolution,
                   simulation_time=simulation_time)
        
        if result and result.success:
            print(f"✓ Simulation config logged")
        
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
        stats_result = client.log(step=1,
                   max_field_amplitude=max_ez,
                   mean_field_amplitude=mean_ez,
                   total_field_energy=field_energy,
                   simulation_completed=True)
        
        if stats_result and stats_result.success:
            print(f"✓ Field statistics logged")
        
        # Generate and log plots
        print("📊 Generating field visualizations...")
        
        # Permittivity plot
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        im1 = ax1.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary")
        ax1.set_title("Waveguide Permittivity Distribution")
        plt.colorbar(im1, ax=ax1, label="Relative Permittivity")
        ax1.axis("off")
        
        # One line to log - no manual save/upload/cleanup!
        eps_result = client.log_matplotlib("permittivity_plot", fig1, 
                                          meta={"description": "Waveguide permittivity distribution", 
                                                "component": "permittivity"})
        
        if eps_result and eps_result.success:
            print(f"  ✓ Permittivity plot logged: {eps_result.url}")
        
        plt.close(fig1)
        
        # Field plot
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary")
        im2 = ax2.imshow(ez_data.transpose(), interpolation="spline36", cmap="RdBu", alpha=0.9)
        ax2.set_title("Electric Field Ez in Waveguide")
        plt.colorbar(im2, ax=ax2, label="Field Amplitude")
        ax2.axis("off")
        
        # One line again!
        field_result = client.log_matplotlib("field_plot", fig2,
                                            meta={"description": "Electric field Ez distribution", 
                                                  "component": "Ez"})
        
        if field_result and field_result.success:
            print(f"  ✓ Field plot logged: {field_result.url}")
        
        plt.close(fig2)
        
        # Direct array visualization
        array_result = client.log_array_as_image("field_heatmap", 
                                                  np.abs(ez_data), 
                                                  cmap='hot',
                                                  meta={"description": "Field magnitude heatmap"})
        
        if array_result and array_result.success:
            print(f"  ✓ Field heatmap logged: {array_result.url}")
        
        print("\n" + "="*70)
        print("✅ Simulation completed successfully!")
        print(f"📈 Logged 2 metric steps and 3 visualizations to OptixLog")
        print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")
        print("="*70)

except ValueError as e:
    print(f"\n❌ OptixLog Error: {e}")
    print("Please ensure your API key is set correctly.")
except Exception as e:
    print(f"\n❌ Simulation Error: {e}")
    import traceback
    traceback.print_exc()

# No manual cleanup needed - context manager handles everything!
