"""
Phase In Material.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: phase_in_material.py
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
        run_name="phase_in_material_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "phase_in_material.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=20
    )

cell_size = mp.Vector3(6, 6, 0)

geometry1 = [
    mp.Cylinder(center=mp.Vector3(), radius=1.0, material=mp.Medium(index=3.5))
]

sim1 = mp.Simulation(cell_size=cell_size, geometry=geometry1, resolution=20)

sim1.init_sim()

geometry2 = [
    mp.Cylinder(center=mp.Vector3(1, 1), radius=1.0, material=mp.Medium(index=3.5))
]

sim2 = mp.Simulation(cell_size=cell_size, geometry=geometry2, resolution=20)

sim2.init_sim()

sim1.fields.phase_in_material(sim2.structure, 10.0)

sim1.run(
    mp.at_beginning(mp.output_epsilon), mp.at_every(0.5, mp.output_epsilon), until=10
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
