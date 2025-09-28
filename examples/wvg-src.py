"""
Wvg Src.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: wvg-src.py
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
        run_name="wvg-src_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "wvg-src.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=10
    )

cell = mp.Vector3(16, 8)

# an asymmetrical dielectric waveguide:
geometry = [
    mp.Block(
        center=mp.Vector3(),
        size=mp.Vector3(mp.inf, 1, mp.inf),
        material=mp.Medium(epsilon=12),
    ),
    mp.Block(
        center=mp.Vector3(y=0.3),
        size=mp.Vector3(mp.inf, 0.1, mp.inf),
        material=mp.Medium(),
    ),
]

# create a transparent source that excites a right-going waveguide mode
sources = [
    mp.EigenModeSource(
        src=mp.ContinuousSource(0.15),
        size=mp.Vector3(y=6),
        center=mp.Vector3(x=-5),
        component=mp.Dielectric,
        eig_parity=mp.ODD_Z,
    )
]

pml_layers = [mp.PML(1.0)]

force_complex_fields = True  # so we can get time-average flux

resolution = 10

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    sources=sources,
    boundary_layers=pml_layers,
    force_complex_fields=force_complex_fields,
    resolution=resolution,
)

sim.run(
    mp.at_beginning(mp.output_epsilon)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True),
    mp.at_end(mp.output_png(mp.Ez, "-a yarg -A $EPS -S3 -Zc dkbluered", rm_h5=False)),
    until=200,
)

flux1 = sim.flux_in_box(
    mp.X, mp.Volume(center=mp.Vector3(-6.0), size=mp.Vector3(1.8, 6))
)
flux2 = sim.flux_in_box(
    mp.X, mp.Volume(center=mp.Vector3(6.0), size=mp.Vector3(1.8, 6))
)

# averaged over y region of width 1.8
print(f"left-going flux = {flux1 / -1.8}")

# averaged over y region of width 1.8
print(f"right-going flux = {flux2 / 1.8}")
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
