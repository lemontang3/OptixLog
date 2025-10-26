"""
Metal Cavity Ldos.Py with OptixLog Integration

Cavity mode analysis

Based on the Meep tutorial: metal-cavity-ldos.py
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
        run_name="metal-cavity-ldos_simulation",
        config={
            "simulation_type": "cavity",
            "description": "Cavity mode analysis",
            "framework": "meep",
            "original_file": "metal-cavity-ldos.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=50,
        df=0.2,
        dpml=1
    )

def metal_cavity(w):
    resolution = 50
    sxy = 2
    dpml = 1
    sxy += 2 * dpml
    cell = mp.Vector3(sxy, sxy)

    pml_layers = [mp.PML(dpml)]
    a = 1
    t = 0.1
    geometry = [
        mp.Block(mp.Vector3(a + 2 * t, a + 2 * t, mp.inf), material=mp.metal),
        mp.Block(mp.Vector3(a, a, mp.inf), material=mp.air),
    ]

    geometry.append(
        mp.Block(
            center=mp.Vector3(a / 2), size=mp.Vector3(2 * t, w, mp.inf), material=mp.air
        )
    )

    fcen = math.sqrt(0.5) / a
    df = 0.2
    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=mp.Vector3()
        )
    ]

    symmetries = [mp.Mirror(mp.Y)]

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        boundary_layers=pml_layers,
        sources=sources,
        symmetries=symmetries,
        resolution=resolution,
    )

    h = mp.Harminv(mp.Ez, mp.Vector3(), fcen, df)
    sim.run(mp.after_sources(h)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True), until_after_sources=500)

    m = h.modes[0]
    f = m.freq
    Q = m.Q
    Vmode = 0.25 * a * a
    ldos_1 = Q / Vmode / (2 * math.pi * f * math.pi * 0.5)

    sim.reset_meep()

    T = 2 * Q * (1 / f)
    sim.run(mp.dft_ldos(f, 0, 1)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True), until_after_sources=T)
    ldos_2 = sim.ldos_data[0]

    return ldos_1, ldos_2


ws = np.arange(0.2, 0.5, 0.1)
ldos_1 = np.zeros(len(ws))
ldos_2 = np.zeros(len(ws))

for j in range(len(ws)):
    ldos_1[j], ldos_2[j] = metal_cavity(ws[j])
    print(f"ldos:, {ldos_1[j]}, {ldos_2[2]}")

plt.figure(dpi=150)
plt.semilogy(1 / ws, ldos_1, "bo-", label="2Q/(πωV)")
plt.semilogy(1 / ws, ldos_2, "rs-", label="LDOS")
plt.xlabel("a/w")
plt.ylabel("2Q/(πωW) or LDOS")
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
