"""
Refl Quartz.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: refl-quartz.py
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
        run_name="refl-quartz_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "refl-quartz.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=200,
        fcen=0.5,
        dpml=1.0
    )

from meep.materials import fused_quartz

resolution = 200  # pixels/μm

dpml = 1.0
sz = 10 + 2 * dpml
cell_size = mp.Vector3(z=sz)
pml_layers = [mp.PML(dpml)]

wvl_min = 0.4
wvl_max = 0.8
fmin = 1 / wvl_max
fmax = 1 / wvl_min
fcen = 0.5 * (fmax + fmin)
df = fmax - fmin
nfreq = 50

sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ex,
        center=mp.Vector3(z=-0.5 * sz + dpml),
    )
]

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    dimensions=1,
    resolution=resolution,
)

refl_fr = mp.FluxRegion(center=mp.Vector3(z=-0.25 * sz))
refl = sim.add_flux(fcen, df, nfreq, refl_fr)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3()
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True), 1e-9))

empty_flux = mp.get_fluxes(refl)
empty_data = sim.get_flux_data(refl)
sim.reset_meep()

geometry = [
    mp.Block(
        mp.Vector3(mp.inf, mp.inf, 0.5 * sz),
        center=mp.Vector3(z=0.25 * sz),
        material=fused_quartz,
    )
]

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    dimensions=1,
    resolution=resolution,
)

refl = sim.add_flux(fcen, df, nfreq, refl_fr)
sim.load_minus_flux_data(refl, empty_data)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3()
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True), 1e-9))

refl_flux = mp.get_fluxes(refl)
R_meep = -1 * np.divide(refl_flux, empty_flux)

freqs = mp.get_flux_freqs(refl)
wvls = np.divide(1, freqs)

eps_quartz = (
    lambda l: 1
    + 0.6961663 * math.pow(l, 2) / (pow(l, 2) - pow(0.0684043, 2))
    + 0.4079426 * pow(l, 2) / (pow(l, 2) - pow(0.1162414, 2))
    + 0.8974794 * pow(l, 2) / (pow(l, 2) - pow(9.896161, 2))
)
R_fresnel = lambda l: math.pow(
    math.fabs(1 - math.sqrt(eps_quartz(l))) / (1 + math.sqrt(eps_quartz(l))), 2
)
R_analytic = [R_fresnel(i) for i in wvls]

plt.figure()
plt.plot(wvls, R_meep, "bo-", label="meep")
plt.plot(wvls, R_analytic, "rs-", label="analytic")
plt.xlabel("wavelength (μm)")
plt.ylabel("reflectance")
plt.axis([0.4, 0.8, 0.0340, 0.0365])
plt.xticks(list(np.arange(0.4, 0.9, 0.1)))
plt.legend(loc="upper right")
plt.show()
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
