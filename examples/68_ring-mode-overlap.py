"""
Ring Mode Overlap.Py with OptixLog Integration

Ring resonator analysis

Based on the Meep tutorial: ring-mode-overlap.py
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
        run_name="ring-mode-overlap_simulation",
        config={
            "simulation_type": "ring_resonator",
            "description": "Ring resonator analysis",
            "framework": "meep",
            "original_file": "ring-mode-overlap.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=20,
        fcen=0.118,
        df=0.010,
        w=1,
        r=1,
        pad=4,
        dpml=2
    )

# Calculating 2d ring-resonator modes, from the Meep tutorial.
n = 3.4  # index of waveguide
w = 1  # width of waveguide
r = 1  # inner radius of ring

pad = 4  # padding between waveguide and edge of PML
dpml = 2  # thickness of PML

sxy = 2 * (r + w + pad + dpml)  # cell size
cell = mp.Vector3(sxy, sxy)

# Create a ring waveguide by two overlapping cylinders - later objects
# take precedence over earlier objects, so we put the outer cylinder first.
# and the inner (air) cylinder second.
geometry = [
    mp.Cylinder(radius=r + w, height=mp.inf, material=mp.Medium(index=n)),
    mp.Cylinder(radius=r, height=mp.inf, material=mp.air),
]

pml_layers = [mp.PML(dpml)]
resolution = 20

# If we don't want to excite a specific mode symmetry, we can just
# put a single point source at some arbitrary place, pointing in some
# arbitrary direction. We will only look for Ez-polarized modes.

fcen = 0.118  # pulse center frequency
df = 0.010  # pulse width (in frequency)
sources = [
    mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(r + 0.1),
    )
]

# exploit the mirror symmetry in structure+source:
symmetries = [mp.Mirror(mp.Y)]

sim = mp.Simulation(
    cell_size=cell,
    resolution=resolution,
    geometry=geometry,
    boundary_layers=pml_layers,
    sources=sources,
    symmetries=symmetries,
)

h1 = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)
sim.run(mp.after_sources(h1)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True), until_after_sources=300)

fields2 = sim.fields
sim.reset_meep()

fcen = 0.236
h2 = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)
sim.run(mp.after_sources(h2)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True), until_after_sources=300)


def overlap_integral(r, ez1, ez2):
    return ez1.conjugate() * ez2


res = sim.integrate2_field_function(fields2, [mp.Ez], [mp.Ez], overlap_integral)
print(f"overlap integral of mode at w and 2w: {abs(res)}")
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
