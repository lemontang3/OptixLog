"""
Material Dispersion.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: material-dispersion.py
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
        run_name="material-dispersion_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "material-dispersion.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=20,
        fcen=1.0,
        df=2.0
    )

# Material dispersion example, from the Meep tutorial.  Here, we simply
# simulate homogenous space filled with a dispersive material, and compute
# its modes as a function of wavevector k.  Since omega/c = k/n, we can
# extract the dielectric function epsilon(omega) = (ck/omega)^2.
cell = mp.Vector3()
resolution = 20

# We'll use a dispersive material with two polarization terms, just for
# illustration.  The first one is a strong resonance at omega=1.1,
# which leads to a polaritonic gap in the dispersion relation.  The second
# one is a weak resonance at omega=0.5, whose main effect is to add a
# small absorption loss around that frequency.

susceptibilities = [
    mp.LorentzianSusceptibility(frequency=1.1, gamma=1e-5, sigma=0.5),
    mp.LorentzianSusceptibility(frequency=0.5, gamma=0.1, sigma=2e-5),
]

default_material = mp.Medium(epsilon=2.25, E_susceptibilities=susceptibilities)

fcen = 1.0
df = 2.0

sources = [
    mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=mp.Vector3())
]

kmin = 0.3
kmax = 2.2
k_interp = 99

kpts = mp.interpolate(k_interp, [mp.Vector3(kmin), mp.Vector3(kmax)])

sim = mp.Simulation(
    cell_size=cell,
    geometry=[],
    sources=sources,
    default_material=default_material,
    resolution=resolution,
)

all_freqs = sim.run_k_points(200, kpts)  # a list of lists of frequencies

for fs, kx in zip(all_freqs, [v.x for v in kpts]):
    for f in fs:
        print(f"eps:, {f.real:.6g}, {f.imag:.6g}, {(kx / f) ** 2:.6g}")
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
