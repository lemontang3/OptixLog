"""
Gaussian Beam.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: gaussian-beam.py
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
        run_name="gaussian-beam_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "gaussian-beam.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=50,
        fcen=1,
        dpml=2
    )

## launch a Gaussian beam
matplotlib.use("agg")
s = 14
resolution = 50
dpml = 2

cell_size = mp.Vector3(s, s)

boundary_layers = [mp.PML(thickness=dpml)]

beam_x0 = mp.Vector3(0, 3.0)  # beam focus (relative to source center)
rot_angle = 0  # CCW rotation angle about z axis (0: +y axis)
beam_kdir = mp.Vector3(0, 1, 0).rotate(
    mp.Vector3(0, 0, 1), math.radians(rot_angle)
)  # beam propagation direction
beam_w0 = 0.8  # beam waist radius
beam_E0 = mp.Vector3(0, 0, 1)
fcen = 1
sources = [
    mp.GaussianBeamSource(
        src=mp.ContinuousSource(fcen),
        center=mp.Vector3(0, -0.5 * s + dpml + 1.0),
        size=mp.Vector3(s),
        beam_x0=beam_x0,
        beam_kdir=beam_kdir,
        beam_w0=beam_w0,
        beam_E0=beam_E0,
    )
]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=boundary_layers,
    sources=sources,
)

sim.run(until=20)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True)

sim.plot2D(
    fields=mp.Ez,
    output_plane=mp.Volume(
        center=mp.Vector3(), size=mp.Vector3(s - 2 * dpml, s - 2 * dpml)
    ),
)

plt.savefig(f"Ez_angle{rot_angle}.png", bbox_inches="tight", pad_inches=0)
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
