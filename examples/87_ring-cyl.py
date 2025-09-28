"""
Ring Cyl.Py with OptixLog Integration

Ring resonator analysis

Based on the Meep tutorial: ring-cyl.py
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
        run_name="ring-cyl_simulation",
        config={
            "simulation_type": "ring_resonator",
            "description": "Ring resonator analysis",
            "framework": "meep",
            "original_file": "ring-cyl.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=20,
        w=1,
        r=1,
        pad=4,
        dpml=32
    )

# Calculating 2d ring-resonator modes using cylindrical coordinates,
# from the Meep tutorial.
import argparse

def main(args):

    n = 3.4  # index of waveguide
    w = 1  # width of waveguide
    r = 1  # inner radius of ring
    pad = 4  # padding between waveguide and edge of PML
    dpml = 32  # thickness of PML

    sr = r + w + pad + dpml  # radial size (cell is from 0 to sr)
    dimensions = mp.CYLINDRICAL
    cell = mp.Vector3(sr, 0, 0)

    # in cylindrical coordinates, the phi (angular) dependence of the fields
    # is given by exp(i m phi), where m is given by:
    m = args.m

    geometry = [
        mp.Block(
            center=mp.Vector3(r + (w / 2)),
            size=mp.Vector3(w, mp.inf, mp.inf),
            material=mp.Medium(index=n),
        )
    ]

    pml_layers = [mp.PML(dpml)]
    resolution = 20

    # If we don't want to excite a specific mode symmetry, we can just
    # put a single point source at some arbitrary place, pointing in some
    # arbitrary direction.  We will only look for Ez-polarized modes.

    fcen = args.fcen  # pulse center frequency
    df = args.df  # pulse frequency width
    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(r + 0.1),
        )
    ]

    # note that the r -> -r mirror symmetry is exploited automatically

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        boundary_layers=pml_layers,
        resolution=resolution,
        sources=sources,
        dimensions=dimensions,
        m=m,
    )

    sim.run(
        mp.after_sources(mp.Harminv(mp.Ez, mp.Vector3(r + 0.1)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True), fcen, df)),
        until_after_sources=200,
    )

    # Output fields for one period at the end.  (If we output
    # at a single time, we might accidentally catch the Ez field when it is
    # almost zero and get a distorted view.)  We'll append the fields
    # to a file to get an r-by-t picture.  We'll also output from -sr to -sr
    # instead of from 0 to sr.
    sim.run(
        mp.in_volume(
            mp.Volume(center=mp.Vector3()
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True), size=mp.Vector3(2 * sr)),
            mp.at_beginning(mp.output_epsilon),
            mp.to_appended("ez", mp.at_every(1 / fcen / 20, mp.output_efield_z)),
        ),
        until=1 / fcen,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fcen", type=float, default=0.15, help="pulse center frequency"
    )
    parser.add_argument("-df", type=float, default=0.1, help="pulse frequency width")
    parser.add_argument(
        "-m",
        type=int,
        default=3,
        help="phi (angular) dependence of the fields given by exp(i m phi)",
    )
    args = parser.parse_args()
    main(args)
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
