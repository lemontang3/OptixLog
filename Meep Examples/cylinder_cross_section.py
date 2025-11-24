"""
Cylinder Cross Section.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: cylinder_cross_section.py
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
        run_name="cylinder_cross_section_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "cylinder_cross_section.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=25,
        r=0.7,
        dpml=0.5
    )

r = 0.7  # radius of cylinder
h = 2.3  # height of cylinder

wvl_min = 2 * np.pi * r / 10
wvl_max = 2 * np.pi * r / 2

frq_min = 1 / wvl_max
frq_max = 1 / wvl_min
frq_cen = 0.5 * (frq_min + frq_max)
dfrq = frq_max - frq_min
nfrq = 100

## at least 8 pixels per smallest wavelength, i.e. np.floor(8/wvl_min)
resolution = 25

dpml = 0.5 * wvl_max
dair = 1.0 * wvl_max

pml_layers = [mp.PML(thickness=dpml)]

sr = r + dair + dpml
sz = dpml + dair + h + dair + dpml
cell_size = mp.Vector3(sr, 0, sz)

sources = [
    mp.Source(
        mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
        component=mp.Er,
        center=mp.Vector3(0.5 * sr, 0, -0.5 * sz + dpml),
        size=mp.Vector3(sr),
    ),
    mp.Source(
        mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
        component=mp.Ep,
        center=mp.Vector3(0.5 * sr, 0, -0.5 * sz + dpml),
        size=mp.Vector3(sr),
        amplitude=-1j,
    ),
]

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    resolution=resolution,
    sources=sources,
    dimensions=mp.CYLINDRICAL,
    m=-1,
)

box_z1 = sim.add_flux(
    frq_cen,
    dfrq,
    nfrq,
    mp.FluxRegion(center=mp.Vector3(0.5 * r, 0, -0.5 * h), size=mp.Vector3(r)),
)
box_z2 = sim.add_flux(
    frq_cen,
    dfrq,
    nfrq,
    mp.FluxRegion(center=mp.Vector3(0.5 * r, 0, +0.5 * h), size=mp.Vector3(r)),
)
box_r = sim.add_flux(
    frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(r), size=mp.Vector3(z=h))
)

sim.run(until_after_sources=10)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True)

freqs = mp.get_flux_freqs(box_z1)
box_z1_data = sim.get_flux_data(box_z1)
box_z2_data = sim.get_flux_data(box_z2)
box_r_data = sim.get_flux_data(box_r)

box_z1_flux0 = mp.get_fluxes(box_z1)

sim.reset_meep()

n_cyl = 2.0
geometry = [
    mp.Block(
        material=mp.Medium(index=n_cyl),
        center=mp.Vector3(0.5 * r),
        size=mp.Vector3(r, 0, h),
    )
]

sim = mp.Simulation(
    cell_size=cell_size,
    geometry=geometry,
    boundary_layers=pml_layers,
    resolution=resolution,
    sources=sources,
    dimensions=mp.CYLINDRICAL,
    m=-1,
)

box_z1 = sim.add_flux(
    frq_cen,
    dfrq,
    nfrq,
    mp.FluxRegion(center=mp.Vector3(0.5 * r, 0, -0.5 * h), size=mp.Vector3(r)),
)
box_z2 = sim.add_flux(
    frq_cen,
    dfrq,
    nfrq,
    mp.FluxRegion(center=mp.Vector3(0.5 * r, 0, +0.5 * h), size=mp.Vector3(r)),
)
box_r = sim.add_flux(
    frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(r), size=mp.Vector3(z=h))
)

sim.load_minus_flux_data(box_z1, box_z1_data)
sim.load_minus_flux_data(box_z2, box_z2_data)
sim.load_minus_flux_data(box_r, box_r_data)

sim.run(until_after_sources=100)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True)

box_z1_flux = mp.get_fluxes(box_z1)
box_z2_flux = mp.get_fluxes(box_z2)
box_r_flux = mp.get_fluxes(box_r)

scatt_flux = np.asarray(box_z1_flux) - np.asarray(box_z2_flux) - np.asarray(box_r_flux)
intensity = np.asarray(box_z1_flux0) / (np.pi * r**2)
scatt_cross_section = np.divide(-scatt_flux, intensity)

if mp.am_master():
    plt.figure(dpi=150)
    plt.loglog(2 * np.pi * r * np.asarray(freqs), scatt_cross_section, "bo-")
    plt.grid(True, which="both", ls="-")
    plt.xlabel("(cylinder circumference)/wavelength, 2πr/λ")
    plt.ylabel("scattering cross section, σ")
    plt.title("Scattering Cross Section of a Lossless Dielectric Cylinder")
    plt.tight_layout()
    plt.savefig("cylinder_cross_section.png")
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
