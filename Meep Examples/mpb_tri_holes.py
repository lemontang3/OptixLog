"""
Mpb Tri Holes.Py with OptixLog Integration

MPB eigenmode analysis

Based on the Meep tutorial: mpb_tri_holes.py
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
        run_name="mpb_tri_holes_simulation",
        config={
            "simulation_type": "mpb_eigenmode",
            "description": "MPB eigenmode analysis",
            "framework": "meep",
            "original_file": "mpb_tri_holes.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=32,
        r=0.45
    )

from meep import mpb

# 2d system: triangular lattice of air holes in dielectric
# This structure has a complete band gap (i.e. a gap in both TE and TM
# simultaneously) for a hole radius of 0.45a and a dielectric constant of
# 12.   (See, e.g., the book "Photonic Crystals" by Joannopoulos et al.)

# first, define the lattice vectors and k-points for a triangular lattice:

geometry_lattice = mp.Lattice(
    size=mp.Vector3(1, 1),
    basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
    basis2=mp.Vector3(math.sqrt(3) / 2, -0.5),
)

kz = 0  # use non-zero kz to consider vertical propagation

k_points = [
    mp.Vector3(z=kz),  # Gamma
    mp.Vector3(0, 0.5, kz),  # M
    mp.Vector3(1 / -3, 1 / 3, kz),  # K
    mp.Vector3(z=kz),  # Gamma
]

k_interp = 4
k_points = mp.interpolate(k_interp, k_points)

# Now, define the geometry, etcetera:

eps = 12  # the dielectric constant of the background
r = 0.45  # the hole radius

default_material = mp.Medium(epsilon=eps)
geometry = [mp.Cylinder(r, material=mp.air)]

resolution = 32
num_bands = 8

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    k_points=k_points,
    default_material=default_material,
    resolution=resolution,
    num_bands=num_bands,
)


def main():
    if kz == 0:
        ms.run_te()
        ms.run_tm()
    else:
        ms.run()  # if kz != 0 there are no purely te and tm bands


if __name__ == "__main__":
    main()
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
