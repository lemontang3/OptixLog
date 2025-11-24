"""
Mpb Bragg.Py with OptixLog Integration

MPB eigenmode analysis

Based on the Meep tutorial: mpb_bragg.py
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
        run_name="mpb_bragg_simulation",
        config={
            "simulation_type": "mpb_eigenmode",
            "description": "MPB eigenmode analysis",
            "framework": "meep",
            "original_file": "mpb_bragg.py"
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

# Compute the bands at the X point for a quarter-wave stack Bragg
# mirror (this is the point that defines the band gap edges).

# the high and low indices:
n_lo = 1.0
n_hi = 3.0

w_hi = n_lo / (n_hi + n_lo)  # a quarter_wave stack

geometry_lattice = mp.Lattice(size=mp.Vector3(1))  # 1d cell
default_material = mp.Medium(index=n_lo)
geometry = mp.Cylinder(
    material=mp.Medium(index=n_hi),
    center=mp.Vector3(),
    axis=mp.Vector3(1),
    radius=mp.inf,
    height=w_hi,
)

kx = 0.5
k_points = [mp.Vector3(kx)]

resolution = 32
num_bands = 8

ms = mpb.ModeSolver(
    num_bands=num_bands,
    k_points=k_points,
    geometry_lattice=geometry_lattice,
    geometry=[geometry],
    resolution=resolution,
    default_material=default_material,
)


def main():
    ms.run_tm(
        mpb.output_hfield_y
    )  # note that TM and TE bands are degenerate, so we only need TM


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
