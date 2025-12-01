"""
Oblique Source.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: oblique-source.py
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
        run_name="oblique-source_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "oblique-source.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=50,
        w=1.0
    )

resolution = 50  # pixels/μm

cell_size = mp.Vector3(14, 14)

pml_layers = [mp.PML(thickness=2)]

# rotation angle (in degrees) of waveguide, counter clockwise (CCW) around z-axis
rot_angle = np.radians(20)

w = 1.0  # width of waveguide

geometry = [
    mp.Block(
        center=mp.Vector3(),
        size=mp.Vector3(mp.inf, w, mp.inf),
        e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), rot_angle),
        e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), rot_angle),
        material=mp.Medium(epsilon=12),
    )
]

fsrc = 0.15  # frequency of eigenmode or constant-amplitude source
bnum = 1  # band number of eigenmode

kpoint = mp.Vector3(x=1).rotate(mp.Vector3(z=1), rot_angle)

compute_flux = True  # compute flux (True) or plot the field profile (False)

eig_src = True  # eigenmode (True) or constant-amplitude (False) source

if eig_src:
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fsrc, fwidth=0.2 * fsrc)
            if compute_flux
            else mp.ContinuousSource(fsrc),
            center=mp.Vector3(),
            size=mp.Vector3(y=3 * w),
            direction=mp.NO_DIRECTION,
            eig_kpoint=kpoint,
            eig_band=bnum,
            eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
            eig_match_freq=True,
        )
    ]
else:
    sources = [
        mp.Source(
            src=mp.GaussianSource(fsrc, fwidth=0.2 * fsrc)
            if compute_flux
            else mp.ContinuousSource(fsrc),
            center=mp.Vector3(),
            size=mp.Vector3(y=3 * w),
            component=mp.Ez,
        )
    ]

sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution,
    boundary_layers=pml_layers,
    sources=sources,
    geometry=geometry,
    symmetries=[mp.Mirror(mp.Y)] if rot_angle == 0 else [],
)

if compute_flux:
    tran = sim.add_flux(
        fsrc, 0, 1, mp.FluxRegion(center=mp.Vector3(x=5), size=mp.Vector3(y=14))
    )
    sim.run(until_after_sources=50)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True)
    res = sim.get_eigenmode_coefficients(
        tran,
        [1],
        eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
        direction=mp.NO_DIRECTION,
        kpoint_func=lambda f, n: kpoint,
    )
    print(
        "flux:, {:.6f}, {:.6f}".format(
            mp.get_fluxes(tran)[0], abs(res.alpha[0, 0, 0]) ** 2
        )
    )
else:
    sim.run(until=100)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True)
    sim.plot2D(
        output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(10, 10)),
        fields=mp.Ez,
        field_parameters={"alpha": 0.9},
    )
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
