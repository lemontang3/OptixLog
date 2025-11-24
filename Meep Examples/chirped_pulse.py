"""
Chirped Pulse.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: chirped_pulse.py
"""

import os
import optixlog
import meep as mp
import matplotlib.pyplot as plt
import numpy as np

# OptixLog Configuration
api_key = os.getenv("OPTIX_API_KEY", "proj_YOUR_API_KEY_HERE")
api_url = os.getenv("OPTIX_API_URL", "https://backend.optixlog.com")
project_name = os.getenv("OPTIX_PROJECT", "MeepExamples")

print(f"🚀 Initializing OptixLog client for project: {project_name}")

try:
    # Initialize OptixLog client
    client = optixlog.init(
        api_key=api_key,
        api_url=api_url,
        project=project_name,
        run_name="chirped_pulse_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "chirped_pulse.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=40,
        dpml=2
    )

## linear-chirped pulse planewave with higher frequencies at the front (down-chirp)
resolution = 40

dpml = 2
pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

sx = 40
sy = 6
cell_size = mp.Vector3(sx + 2 * dpml, sy)

v0 = 1.0  # pulse center frequency
a = 0.2  # Gaussian envelope half-width
b = -0.5  # linear chirp rate (positive: up-chirp, negative: down-chirp)
t0 = 15  # peak time

chirp = lambda t: np.exp(1j * 2 * np.pi * v0 * (t - t0)) * np.exp(
    -a * (t - t0) ** 2 + 1j * b * (t - t0) ** 2
)

sources = [
    mp.Source(
        src=mp.CustomSource(src_func=chirp),
        center=mp.Vector3(-0.5 * sx),
        size=mp.Vector3(y=sy),
        component=mp.Ez,
    )
]

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    resolution=resolution,
    k_point=mp.Vector3(),
    sources=sources,
    symmetries=[mp.Mirror(mp.Y)],
)

sim.run(
    mp.in_volume(
        mp.Volume(center=mp.Vector3()
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True), size=mp.Vector3(sx, sy)),
        mp.at_every(2.7, mp.output_efield_z),
    ),
    until=t0 + 50,
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
