"""
Oblique Planewave.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: oblique-planewave.py
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
        run_name="oblique-planewave_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "oblique-planewave.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation start
    client.log(step=0, simulation_started=True)

"""Demonstration of launching a planewave source at oblique incidence.

tutorial reference:
https://meep.readthedocs.io/en/latest/Python_Tutorials/Eigenmode_Source/#planewaves-in-homogeneous-media
"""

mp.verbosity(2)

resolution_um = 50
pml_um = 2.0
size_um = 10.0
cell_size = mp.Vector3(size_um + 2 * pml_um, size_um, 0)
pml_layers = [mp.PML(thickness=pml_um, direction=mp.X)]

# Incident angle of planewave. 0 is +x with rotation in
# counter clockwise (CCW) direction around z axis.
incident_angle = np.radians(40.0)

wavelength_um = 1.0
frequency = 1 / wavelength_um

n_mat = 1.5  # refractive index of homogeneous material
default_material = mp.Medium(index=n_mat)

k_point = mp.Vector3(n_mat * frequency, 0, 0).rotate(
    mp.Vector3(0, 0, 1), incident_angle
)

if incident_angle == 0:
    direction = mp.AUTOMATIC
    eig_parity = mp.EVEN_Y + mp.ODD_Z
    symmetries = [mp.Mirror(mp.Y)]
    eig_vol = None
else:
    direction = mp.NO_DIRECTION
    eig_parity = mp.ODD_Z
    symmetries = []
    eig_vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, 1 / resolution_um, 0))

sources = [
    mp.EigenModeSource(
        src=mp.ContinuousSource(frequency),
        center=mp.Vector3(),
        size=mp.Vector3(0, size_um, 0),
        direction=direction,
        eig_kpoint=k_point,
        eig_band=1,
        eig_parity=eig_parity,
        eig_vol=eig_vol,
    )
]

sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution_um,
    boundary_layers=pml_layers,
    sources=sources,
    k_point=k_point,
    default_material=default_material,
    symmetries=symmetries,
)

sim.run(until=23.56)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True)

output_plane = mp.Volume(center=mp.Vector3(), size=mp.Vector3(size_um, size_um, 0))

fig, ax = plt.subplots()
sim.plot2D(fields=mp.Ez, output_plane=output_plane, ax=ax)
fig.savefig("planewave_source.png", bbox_inches="tight")
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
