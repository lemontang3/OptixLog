"""
Mpb Data Analysis.Py with OptixLog Integration

MPB eigenmode analysis

Based on the Meep tutorial: mpb_data_analysis.py
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
        run_name="mpb_data_analysis_simulation",
        config={
            "simulation_type": "mpb_eigenmode",
            "description": "MPB eigenmode analysis",
            "framework": "meep",
            "original_file": "mpb_data_analysis.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=32
    )

import os
import sys

from meep import mpb

examples_dir = os.path.realpath(os.path.dirname(__file__))
sys.path.insert(0, examples_dir)


def tri_rods():
    # Import the ModeSolver defined in the mpb_tri_rods.py example
    from mpb_tri_rods import ms as tr_ms

    efields = []

    # Band function to collect the efields
    def get_efields(tr_ms, band):
        efields.append(tr_ms.get_efield(band))

    tr_ms.run_tm(
        mpb.output_at_kpoint(
            mp.Vector3(1 / -3, 1 / 3), mpb.fix_efield_phase, get_efields
        )
    )

    # Create an MPBData instance to transform the efields
    md = mpb.MPBData(rectify=True, resolution=32, periods=3)

    converted = []
    for f in efields:
        # Get just the z component of the efields
        f = f[..., 0, 2]
        converted.append(md.convert(f))

    tr_ms.run_te()

    eps = tr_ms.get_epsilon()
    plt.imshow(eps.T, interpolation="spline36", cmap="binary")
    plt.axis("off")
    plt.show()

    md = mpb.MPBData(rectify=True, resolution=32, periods=3)
    rectangular_data = md.convert(eps)
    plt.imshow(rectangular_data.T, interpolation="spline36", cmap="binary")
    plt.axis("off")
    plt.show()

    for i, f in enumerate(converted):
        plt.subplot(331 + i)
        plt.contour(rectangular_data.T, cmap="binary")
        plt.imshow(np.real(f).T, interpolation="spline36", cmap="RdBu", alpha=0.9)
        plt.axis("off")

    plt.show()


def diamond():
    # Import the ModeSolver from the mpb_diamond.py example
    from mpb_diamond import ms as d_ms

    dpwr = []

    def get_dpwr(ms, band):
        dpwr.append(ms.get_dpwr(band))

    d_ms.run(mpb.output_at_kpoint(mp.Vector3(0, 0.625, 0.375), get_dpwr))

    md = mpb.MPBData(rectify=True, periods=2, resolution=32)
    converted_dpwr = [md.convert(d) for d in dpwr]

    # TODO: Plot


if __name__ == "__main__":
    tri_rods()
    diamond()
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
