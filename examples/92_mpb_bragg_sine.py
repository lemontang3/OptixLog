"""
Mpb Bragg Sine.Py with OptixLog Integration

MPB eigenmode analysis

Based on the Meep tutorial: mpb_bragg_sine.py
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
        run_name="mpb_bragg_sine_simulation",
        config={
            "simulation_type": "mpb_eigenmode",
            "description": "MPB eigenmode analysis",
            "framework": "meep",
            "original_file": "mpb_bragg_sine.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=32
    )

from meep import mpb

# Compute the band structure for a Bragg mirror consisting of a
# sinusoidally-varying dielectric index.

# The index will vary sinusoidally between index-min and index-max:
index_min = 1
index_max = 3


# Define a function of position p (in the lattice basis) that returns
# the material at that position.  In this case, we use the function:
#        index-min + 0.5 * (index-max - index-min)
#                        * (1 + cos(2*pi*x))
# This is periodic, and also has inversion symmetry.
def eps_func(p):
    return mp.Medium(
        index=index_min
        + 0.5 * (index_max - index_min) * (1 + math.cos(2 * math.pi * p.x))
    )


geometry_lattice = mp.Lattice(size=mp.Vector3(1))  # 1d cell

# We'll just make it the default material, so that it goes everywhere.
default_material = eps_func

k_points = mp.interpolate(9, [mp.Vector3(), mp.Vector3(x=0.5)])

resolution = 32
num_bands = 8

ms = mpb.ModeSolver(
    num_bands=num_bands,
    k_points=k_points,
    geometry_lattice=geometry_lattice,
    resolution=resolution,
    default_material=default_material,
)


def main():
    # the TM and TE bands are degenerate, so we only need TM:
    ms.run_tm()


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
