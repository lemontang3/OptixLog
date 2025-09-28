"""
Mpb Sq Rods.Py with OptixLog Integration

MPB eigenmode analysis

Based on the Meep tutorial: mpb_sq_rods.py
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
        run_name="mpb_sq_rods_simulation",
        config={
            "simulation_type": "mpb_eigenmode",
            "description": "MPB eigenmode analysis",
            "framework": "meep",
            "original_file": "mpb_sq_rods.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=32,
        r=0.2
    )

import time

from meep import mpb

# Compute band structure for a square lattice of dielectric rods
# in air.

# Define various parameters with define_param so that they are
# settable from the command_line (with mpb <param>=<value>):
r = 0.2  # radius of the rods
eps = 11.56  # dielectric constant
k_interp = 4  # number of k points to interpolate

GaAs = mp.Medium(epsilon=eps)

geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))  # 2d cell

geometry = [mp.Cylinder(r, material=GaAs)]

Gamma = mp.Vector3()
X = mp.Vector3(0.5, 0)
M = mp.Vector3(0.5, 0.5)
k_points = mp.interpolate(k_interp, [Gamma, X, M, Gamma])

resolution = 32
num_bands = 8

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    k_points=k_points,
    resolution=resolution,
    num_bands=num_bands,
)


def main():
    # Compute the TE and TM bands and report the total elapsed time:
    t0 = time.time()
    ms.run_te()
    ms.run_tm()
    print(f"total time for both TE and TM bands: {time.time() - t0:.2f} seconds")

    ms.display_eigensolver_stats()


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
