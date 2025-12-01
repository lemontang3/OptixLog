"""
Cherenkov Radiation.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: cherenkov-radiation.py
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
        run_name="cherenkov-radiation_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "cherenkov-radiation.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=10,
        dpml=1.0
    )

## moving point charge with superluminal phase velocity in dielectric media emitting Cherenkov radiation
sx = 60
sy = 60
cell_size = mp.Vector3(sx, sy, 0)

dpml = 1.0
pml_layers = [mp.PML(thickness=dpml)]

v = 0.7  # velocity of point charge

symmetries = [mp.Mirror(direction=mp.Y)]

sim = mp.Simulation(
    resolution=10,
    cell_size=cell_size,
    default_material=mp.Medium(index=1.5),
    symmetries=symmetries,
    boundary_layers=pml_layers,
)


def move_source(sim):
    sim.change_sources(
        [
            mp.Source(
                mp.ContinuousSource(frequency=1e-10),
                component=mp.Ex,
                center=mp.Vector3(-0.5 * sx + dpml + v * sim.meep_time()),
            )
        ]
    )


sim.run(
    move_source,
    mp.at_every(2, mp.output_png(mp.Hz, "-vZc dkbluered -M 1")
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True)),
    until=sx / v,
)
except ValueError as e:
    print(f"\n❌ OptixLog Error: {{e}}")
    print("Please ensure your API key and URL are correct.")
except Exception as e:
    print(f"\n❌ Simulation Error: {{e}}")

finally:
    # Clean up generated files
    import glob
    for file_path in glob.glob("*.png") + glob.glob("*.csv") + glob.glob("*.txt"):
        if os.path.exists(file_path):
            os.remove(file_path)
