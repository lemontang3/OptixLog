"""
Differential Cross Section.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: differential_cross_section.py
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
        run_name="differential_cross_section_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "differential_cross_section.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=20,
        r=1.0,
        dpml=0.5
    )

import PyMieScatt as ps

r = 1.0  # radius of sphere

frq_cen = 1.0

resolution = 20  # pixels/um

dpml = 0.5
dair = 1.5  # at least 0.5/frq_cen padding between source and near-field monitor

pml_layers = [mp.PML(thickness=dpml)]

s = 2 * (dpml + dair + r)
cell_size = mp.Vector3(s, s, s)

# circularly-polarized source with propagation axis along x
# is_integrated=True necessary for any planewave source extending into PML
sources = [
    mp.Source(
        mp.GaussianSource(frq_cen, fwidth=0.2 * frq_cen, is_integrated=True),
        center=mp.Vector3(-0.5 * s + dpml),
        size=mp.Vector3(0, s, s),
        component=mp.Ez,
    ),
    mp.Source(
        mp.GaussianSource(frq_cen, fwidth=0.2 * frq_cen, is_integrated=True),
        center=mp.Vector3(-0.5 * s + dpml),
        size=mp.Vector3(0, s, s),
        component=mp.Ey,
        amplitude=1j,
    ),
]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    k_point=mp.Vector3(),
)

box_flux = sim.add_flux(
    frq_cen,
    0,
    1,
    mp.FluxRegion(center=mp.Vector3(x=-2 * r), size=mp.Vector3(0, 4 * r, 4 * r)),
)

nearfield_box = sim.add_near2far(
    frq_cen,
    0,
    1,
    mp.Near2FarRegion(
        center=mp.Vector3(x=-2 * r), size=mp.Vector3(0, 4 * r, 4 * r), weight=+1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(x=+2 * r), size=mp.Vector3(0, 4 * r, 4 * r), weight=-1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(y=-2 * r), size=mp.Vector3(4 * r, 0, 4 * r), weight=+1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(y=+2 * r), size=mp.Vector3(4 * r, 0, 4 * r), weight=-1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(z=-2 * r), size=mp.Vector3(4 * r, 4 * r, 0), weight=+1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(z=+2 * r), size=mp.Vector3(4 * r, 4 * r, 0), weight=-1
    ),
)

sim.run(until_after_sources=10)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True)

input_flux = mp.get_fluxes(box_flux)[0]
nearfield_box_data = sim.get_near2far_data(nearfield_box)

sim.reset_meep()

n_sphere = 2.0
geometry = [
    mp.Sphere(material=mp.Medium(index=n_sphere), center=mp.Vector3(), radius=r)
]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    k_point=mp.Vector3(),
    geometry=geometry,
)

nearfield_box = sim.add_near2far(
    frq_cen,
    0,
    1,
    mp.Near2FarRegion(
        center=mp.Vector3(x=-2 * r), size=mp.Vector3(0, 4 * r, 4 * r), weight=+1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(x=+2 * r), size=mp.Vector3(0, 4 * r, 4 * r), weight=-1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(y=-2 * r), size=mp.Vector3(4 * r, 0, 4 * r), weight=+1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(y=+2 * r), size=mp.Vector3(4 * r, 0, 4 * r), weight=-1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(z=-2 * r), size=mp.Vector3(4 * r, 4 * r, 0), weight=+1
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(z=+2 * r), size=mp.Vector3(4 * r, 4 * r, 0), weight=-1
    ),
)

sim.load_minus_near2far_data(nearfield_box, nearfield_box_data)

sim.run(until_after_sources=100)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True)

npts = 100  # number of points in [0,pi) range of polar angles to sample far fields along semi-circle
angles = np.pi / npts * np.arange(npts)

ff_r = 10000 * r  # radius of far-field semi-circle

E = np.zeros((npts, 3), dtype=np.complex128)
H = np.zeros((npts, 3), dtype=np.complex128)
for n in range(npts):
    ff = sim.get_farfield(
        nearfield_box, ff_r * mp.Vector3(np.cos(angles[n]), 0, np.sin(angles[n]))
    )
    E[n, :] = [np.conj(ff[j]) for j in range(3)]
    H[n, :] = [ff[j + 3] for j in range(3)]

Px = np.real(np.multiply(E[:, 1], H[:, 2]) - np.multiply(E[:, 2], H[:, 1]))
Py = np.real(np.multiply(E[:, 2], H[:, 0]) - np.multiply(E[:, 0], H[:, 2]))
Pz = np.real(np.multiply(E[:, 0], H[:, 1]) - np.multiply(E[:, 1], H[:, 0]))
Pr = np.sqrt(np.square(Px) + np.square(Py) + np.square(Pz))

intensity = input_flux / (4 * r) ** 2
diff_cross_section = ff_r**2 * Pr / intensity
scatt_cross_section_meep = (
    2 * np.pi * np.sum(np.multiply(diff_cross_section, np.sin(angles))) * np.pi / npts
)
scatt_cross_section_theory = (
    ps.MieQ(n_sphere, 1000 / frq_cen, 2 * r * 1000, asDict=True, asCrossSection=True)[
        "Csca"
    ]
    * 1e-6
)  # units of um^2

print(
    "scatt:, {:.16f} (meep), {:.16f} (theory)".format(
        scatt_cross_section_meep, scatt_cross_section_theory
    )
)
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
