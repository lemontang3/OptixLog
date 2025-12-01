"""
Faraday Rotation.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: faraday-rotation.py
"""

import os
import optixlog
import meep as mp
import matplotlib.pyplot as plt
import numpy as np

# OptixLog Configuration
api_key = os.getenv("OPTIX_API_KEY", "proj_YOUR_API_KEY_HERE")
api_url = os.getenv("OPTIX_API_URL", "https://optixlog.com")
project_name = os.getenv("OPTIX_PROJECT", "MeepExamples")

print(f"🚀 Initializing OptixLog client for project: {project_name}")

try:
    # Initialize OptixLog client
    client = optixlog.init(
        api_key=api_key,
        api_url=api_url,
        project=project_name,
        run_name="faraday-rotation_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "faraday-rotation.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=50
    )

# From the Meep tutorial: plotting Faraday rotation of a linearly polarized plane wave
epsn = 1.5  # background permittivity
f0 = 1.0  # natural frequency
gamma = 1e-6  # damping rate
sn = 0.1  # sigma parameter
b0 = 0.15  # magnitude of bias vector

susc = [
    mp.GyrotropicLorentzianSusceptibility(
        frequency=f0, gamma=gamma, sigma=sn, bias=mp.Vector3(0, 0, b0)
    )
]
mat = mp.Medium(epsilon=epsn, mu=1, E_susceptibilities=susc)

## Set up and run the Meep simulation:
tmax = 100
L = 20.0
cell = mp.Vector3(0, 0, L)
fsrc, src_z = 0.8, -8.5
pml_layers = [mp.PML(thickness=1.0, direction=mp.Z)]

sources = [
    mp.Source(
        mp.ContinuousSource(frequency=fsrc),
        component=mp.Ex,
        center=mp.Vector3(0, 0, src_z),
    )
]

sim = mp.Simulation(
    cell_size=cell,
    geometry=[],
    sources=sources,
    boundary_layers=pml_layers,
    default_material=mat,
    resolution=50,
)
sim.run(until=tmax)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True)

ex_data = sim.get_efield_x().real
ey_data = sim.get_efield_y().real

z = np.linspace(-L / 2, L / 2, len(ex_data))
plt.figure(1)
plt.plot(z, ex_data, label="Ex")
plt.plot(z, ey_data, label="Ey")
plt.xlim(-L / 2, L / 2)
plt.xlabel("z")
plt.legend()

## Comparison with analytic result:
dfsq = f0**2 - 1j * fsrc * gamma - fsrc**2
eperp = epsn + sn * f0**2 * dfsq / (dfsq**2 - (fsrc * b0) ** 2)
eta = sn * f0**2 * fsrc * b0 / (dfsq**2 - (fsrc * b0) ** 2)

k_gyro = 2 * np.pi * fsrc * np.sqrt(0.5 * (eperp - np.sqrt(eperp**2 - eta**2)))
Ex_theory = 0.37 * np.cos(k_gyro * (z - src_z)).real
Ey_theory = 0.37 * np.sin(k_gyro * (z - src_z)).real

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(z, ex_data, label="Ex (MEEP)")
plt.plot(z, Ex_theory, "k--")
plt.plot(z, -Ex_theory, "k--", label="Ex envelope (theory)")
plt.xlim(-L / 2, L / 2)
plt.xlabel("z")
plt.legend(loc="lower right")

plt.subplot(2, 1, 2)
plt.plot(z, ey_data, label="Ey (MEEP)")
plt.plot(z, Ey_theory, "k--")
plt.plot(z, -Ey_theory, "k--", label="Ey envelope (theory)")
plt.xlim(-L / 2, L / 2)
plt.xlabel("z")
plt.legend(loc="lower right")
plt.tight_layout()
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
