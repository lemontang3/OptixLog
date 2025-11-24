"""
Cyl Ellipsoid.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: cyl-ellipsoid.py
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
        run_name="cyl-ellipsoid_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "cyl-ellipsoid.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=100
    )

def main():

    c = mp.Cylinder(radius=3, material=mp.Medium(index=3.5))
    e = mp.Ellipsoid(size=mp.Vector3(1, 2, mp.inf))

    src_cmpt = mp.Hz
    sources = mp.Source(
        src=mp.GaussianSource(1, fwidth=0.1), component=src_cmpt, center=mp.Vector3()
    )

    if src_cmpt == mp.Ez:
        symmetries = [mp.Mirror(mp.X), mp.Mirror(mp.Y)]

    if src_cmpt == mp.Hz:
        symmetries = [mp.Mirror(mp.X, -1), mp.Mirror(mp.Y, -1)]

    sim = mp.Simulation(
        cell_size=mp.Vector3(10, 10),
        geometry=[c, e],
        boundary_layers=[mp.PML(1.0)],
        sources=[sources],
        symmetries=symmetries,
        resolution=100,
    )

    def print_stuff(sim_obj):
        v = mp.Vector3(4.13, 3.75, 0)
        p = sim.get_field_point(src_cmpt, v)
        print(f"t, Ez: {sim.round_time()} {p.real}+{p.imag}i")

    sim.run(
        mp.at_beginning(mp.output_epsilon)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True),
        mp.at_every(0.25, print_stuff),
        mp.at_end(print_stuff),
        mp.at_end(mp.output_efield_z),
        until=23,
    )

    print(f"stopped at meep time = {sim.round_time()}")


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
