"""
Absorber 1D.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: absorber-1d.py
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
        run_name="absorber-1d_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "absorber-1d.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=40
    )

import argparse

from meep.materials import Al

def main(args):

    resolution = 40
    cell_size = mp.Vector3(z=10)

    boundary_layers = [
        mp.PML(1, direction=mp.Z) if args.pml else mp.Absorber(1, direction=mp.Z)
    ]

    sources = [
        mp.Source(
            src=mp.GaussianSource(1 / 0.803, fwidth=0.1),
            center=mp.Vector3(),
            component=mp.Ex,
        )
    ]

    def print_stuff(sim):
        p = sim.get_field_point(mp.Ex, mp.Vector3())
        print(f"ex:, {sim.meep_time()}, {p.real}")

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        dimensions=1,
        default_material=Al,
        boundary_layers=boundary_layers,
        sources=sources,
    )

    sim.run(
        mp.at_every(10, print_stuff)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True),
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(), 1e-6),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pml", action="store_true", default=False, help="Use PML as boundary layer"
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
