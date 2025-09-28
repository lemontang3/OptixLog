"""
Mpb Hole Slab.Py with OptixLog Integration

MPB eigenmode analysis

Based on the Meep tutorial: mpb_hole_slab.py
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
        run_name="mpb_hole_slab_simulation",
        config={
            "simulation_type": "mpb_eigenmode",
            "description": "MPB eigenmode analysis",
            "framework": "meep",
            "original_file": "mpb_hole_slab.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        r=0.3
    )

from meep import mpb

# Photonic crystal slab consisting of a triangular lattice of air
# holes in a finite_thickness dielectric slab, optionally with a
# substrate on one side of the slab.  See the paper: S. G. Johnson,
# S. Fan, P. R. Villeneuve, J. D. Joannopoulos, L. A. Kolodziejski,
# "Guided modes in photonic crystal slabs," PRB 60, 5751 (August
# 1999).

# Note that this structure has mirror symmetry throught the z=0 plane,
# and we are looking at k_vectors in the xy plane only.  Thus, we can
# break up the modes into even and odd (analogous to TE and TM), using
# the run_zeven and run_zodd functions.

h = 0.5  # the thickness of the slab
eps = 12.0  # the dielectric constant of the slab
loweps = 1.0  # the dielectric constant of the substrate
r = 0.3  # the radius of the holes
supercell_h = 4  # height of the supercell

# triangular lattice with vertical supercell:
geometry_lattice = mp.Lattice(
    size=mp.Vector3(1, 1, supercell_h),
    basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
    basis2=mp.Vector3(math.sqrt(3) / 2, -0.5),
)

geometry = [
    mp.Block(
        material=mp.Medium(epsilon=loweps),
        center=mp.Vector3(z=0.25 * supercell_h),
        size=mp.Vector3(mp.inf, mp.inf, 0.5 * supercell_h),
    ),
    mp.Block(material=mp.Medium(epsilon=eps), size=mp.Vector3(mp.inf, mp.inf, h)),
    mp.Cylinder(r, material=mp.air, height=supercell_h),
]

# 1st Brillouin zone of a triangular lattice:
Gamma = mp.Vector3()
M = mp.Vector3(y=0.5)
K = mp.Vector3(1 / -3, 1 / 3)

only_K = False  # run with only_K=true to only do this k_point
k_interp = 4  # the number of k points to interpolate
k_points = [K] if only_K else mp.interpolate(k_interp, [Gamma, M, K, Gamma])
resolution = mp.Vector3(32, 32, 16)
num_bands = 9

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    resolution=resolution,
    num_bands=num_bands,
    k_points=k_points,
)


def main():
    # Run even and odd bands, outputting fields only at the K point:
    if loweps == 1.0:
        # we only have even/odd classification for symmetric structure
        ms.run_zeven(mpb.output_at_kpoint(K, mpb.output_hfield_z))
        ms.run_zodd(mpb.output_at_kpoint(K, mpb.output_dfield_z))
    else:
        ms.run(mpb.output_at_kpoint(K, mpb.output_hfield_z), mpb.display_zparities)

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
