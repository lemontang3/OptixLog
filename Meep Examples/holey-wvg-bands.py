"""
Holey Wvg Bands.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: holey-wvg-bands.py
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
        run_name="holey-wvg-bands_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "holey-wvg-bands.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=20,
        fcen=0.25,
        df=1.5,
        w=1.2,
        r=0.36,
        dpml=1
    )

# Meep Tutorial: Hz-polarized transmission and reflection through a cavity
# formed by a periodic sequence of holes in a dielectric waveguide,
# with a defect formed by a larger spacing between one pair of holes.

# This structure is based on one analyzed in:
#    S. Fan, J. N. Winn, A. Devenyi, J. C. Chen, R. D. Meade, and
#    J. D. Joannopoulos, "Guided and defect modes in periodic dielectric
#    waveguides," J. Opt. Soc. Am. B, 12 (7), 1267-1272 (1995).


def main():
    # Some parameters to describe the geometry:
    eps = 13  # dielectric constant of waveguide
    w = 1.2  # width of waveguide
    r = 0.36  # radius of holes

    # The cell dimensions
    sy = 12  # size of cell in y direction (perpendicular to wvg.)
    dpml = 1  # PML thickness (y direction only!)

    cell = mp.Vector3(1, sy)

    b = mp.Block(size=mp.Vector3(mp.inf, w, mp.inf), material=mp.Medium(epsilon=eps))
    c = mp.Cylinder(radius=r)

    fcen = 0.25  # pulse center frequency
    df = 1.5  # pulse freq. width: large df = short impulse

    s = mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df),
        component=mp.Hz,
        center=mp.Vector3(0.1234),
    )

    sym = mp.Mirror(direction=mp.Y, phase=-1)

    sim = mp.Simulation(
        cell_size=cell,
        geometry=[b, c],
        sources=[s],
        symmetries=[sym],
        boundary_layers=[mp.PML(dpml, direction=mp.Y)],
        resolution=20,
    )

    kx = False  # if true, do run at specified kx and get fields
    if kx:
        sim.k_point = mp.Vector3(kx)

        sim.run(
            mp.at_beginning(mp.output_epsilon)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True),
            mp.after_sources(mp.Harminv(mp.Hz, mp.Vector3(0.1234), fcen, df)),
            until_after_sources=300,
        )

        sim.run(mp.at_every(1 / fcen / 20, mp.output_hfield_z)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True), until=1 / fcen)

    else:
        k_interp = 19  # # k-points to interpolate, otherwise

        sim.run_k_points(300, mp.interpolate(k_interp, [mp.Vector3(), mp.Vector3(0.5)]))


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
