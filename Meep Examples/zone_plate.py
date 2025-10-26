"""
Zone Plate.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: zone_plate.py
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
        run_name="zone_plate_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "zone_plate.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation start
    client.log(step=0, simulation_started=True)

"""Computes the diffraction spectra of a zone plate in cylindrical coords."""

resolution_um = 25

pml_um = 1.0
substrate_um = 2.0
padding_um = 2.0
height_um = 0.5
focal_length_um = 200
scan_length_z_um = 100
farfield_resolution_um = 10

pml_layers = [mp.PML(thickness=pml_um)]

wavelength_um = 0.5
frequency = 1 / wavelength_um
frequench_width = 0.2 * frequency

# The number of zones in the zone plate.
# Odd-numbered zones impart a π phase shift and
# even-numbered zones impart no phase shift.
num_zones = 25

# Specify the radius of each zone using the equation
# from https://en.wikipedia.org/wiki/Zone_plate.
zone_radius_um = np.zeros(num_zones)
for n in range(1, num_zones + 1):
    zone_radius_um[n - 1] = math.sqrt(
        n * wavelength_um * (focal_length_um + n * wavelength_um / 4)
    )

size_r_um = zone_radius_um[-1] + padding_um + pml_um
size_z_um = pml_um + substrate_um + height_um + padding_um + pml_um
cell_size = mp.Vector3(size_r_um, 0, size_z_um)

# Specify a (linearly polarized) planewave at normal incidence.
sources = [
    mp.Source(
        mp.GaussianSource(frequency, fwidth=frequench_width, is_integrated=True),
        component=mp.Er,
        center=mp.Vector3(0.5 * size_r_um, 0, -0.5 * size_z_um + pml_um),
        size=mp.Vector3(size_r_um),
    ),
    mp.Source(
        mp.GaussianSource(frequency, fwidth=frequench_width, is_integrated=True),
        component=mp.Ep,
        center=mp.Vector3(0.5 * size_r_um, 0, -0.5 * size_z_um + pml_um),
        size=mp.Vector3(size_r_um),
        amplitude=-1j,
    ),
]

glass = mp.Medium(index=1.5)

# Add the substrate.
geometry = [
    mp.Block(
        material=glass,
        size=mp.Vector3(size_r_um, 0, pml_um + substrate_um),
        center=mp.Vector3(
            0.5 * size_r_um, 0, -0.5 * size_z_um + 0.5 * (pml_um + substrate_um)
        ),
    )
]

# Add the zone plates starting with the ones with largest radius.
for n in range(num_zones - 1, -1, -1):
    geometry.append(
        mp.Block(
            material=glass if n % 2 == 0 else mp.vacuum,
            size=mp.Vector3(zone_radius_um[n], 0, height_um),
            center=mp.Vector3(
                0.5 * zone_radius_um[n],
                0,
                -0.5 * size_z_um + pml_um + substrate_um + 0.5 * height_um,
            ),
        )
    )

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    resolution=resolution_um,
    sources=sources,
    geometry=geometry,
    dimensions=mp.CYLINDRICAL,
    m=-1,
)

# Add the near-field monitor (must be entirely in air).
n2f_monitor = sim.add_near2far(
    frequency,
    0,
    1,
    mp.Near2FarRegion(
        center=mp.Vector3(0.5 * (size_r_um - pml_um), 0, 0.5 * size_z_um - pml_um),
        size=mp.Vector3(size_r_um - pml_um, 0, 0),
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(
            size_r_um - pml_um,
            0,
            0.5 * size_z_um - pml_um - 0.5 * (height_um + padding_um),
        ),
        size=mp.Vector3(0, 0, height_um + padding_um),
    ),
)

fig, ax = plt.subplots()
sim.plot2D(ax=ax)
if mp.am_master():
    fig.savefig("zone_plate_layout.png", bbox_inches="tight", dpi=150)

# Timestep the fields until they have sufficiently decayed away.
sim.run(
    until_after_sources=mp.stop_when_fields_decayed(
        50.0, mp.Er, mp.Vector3(0.5 * size_r_um, 0, 0)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True), 1e-6
    )
)

farfields_r = sim.get_farfields(
    n2f_monitor,
    farfield_resolution_um,
    center=mp.Vector3(
        0.5 * (size_r_um - pml_um),
        0,
        -0.5 * size_z_um + pml_um + substrate_um + height_um + focal_length_um,
    ),
    size=mp.Vector3(size_r_um - pml_um, 0, 0),
)

farfields_z = sim.get_farfields(
    n2f_monitor,
    farfield_resolution_um,
    center=mp.Vector3(
        0, 0, -0.5 * size_z_um + pml_um + substrate_um + height_um + focal_length_um
    ),
    size=mp.Vector3(0, 0, scan_length_z_um),
)

intensity_r = (
    np.absolute(farfields_r["Ex"]) ** 2
    + np.absolute(farfields_r["Ey"]) ** 2
    + np.absolute(farfields_r["Ez"]) ** 2
)
intensity_z = (
    np.absolute(farfields_z["Ex"]) ** 2
    + np.absolute(farfields_z["Ey"]) ** 2
    + np.absolute(farfields_z["Ez"]) ** 2
)

# Plot the intensity data and save the result to disk.
fig, ax = plt.subplots(ncols=2)

ax[0].semilogy(np.linspace(0, size_r_um - pml_um, intensity_r.size), intensity_r, "bo-")
ax[0].set_xlim(-2, 20)
ax[0].set_xticks(np.arange(0, 25, 5))
ax[0].grid(True, axis="y", which="both", ls="-")
ax[0].set_xlabel(r"$r$ coordinate (μm)")
ax[0].set_ylabel(r"energy density of far fields, |E|$^2$")

ax[1].semilogy(
    np.linspace(
        focal_length_um - 0.5 * scan_length_z_um,
        focal_length_um + 0.5 * scan_length_z_um,
        intensity_z.size,
    ),
    intensity_z,
    "bo-",
)
ax[1].grid(True, axis="y", which="both", ls="-")
ax[1].set_xlabel(r"$z$ coordinate (μm)")
ax[1].set_ylabel(r"energy density of far fields, |E|$^2$")

fig.suptitle(f"binary-phase zone plate with focal length $z$ = {focal_length_um} μm")

if mp.am_master():
    fig.savefig("zone_plate_farfields.png", dpi=200, bbox_inches="tight")
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
