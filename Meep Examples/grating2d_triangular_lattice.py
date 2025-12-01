"""
Grating2D Triangular Lattice.Py with OptixLog Integration

Grating diffraction analysis

Based on the Meep tutorial: grating2d_triangular_lattice.py
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
        run_name="grating2d_triangular_lattice_simulation",
        config={
            "simulation_type": "grating",
            "description": "Grating diffraction analysis",
            "framework": "meep",
            "original_file": "grating2d_triangular_lattice.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=100,
        fcen=1,
        r=2.0,
        dpml=1.0
    )

# Computes the diffraction orders of a 2D binary grating with
# triangular lattice using a rectangular supercell and verifies
# that only the diffraction orders of the actual unit cell
# produce non-zero power (up to discretization error)
resolution = 100  # pixels/μm

ng = 1.5
glass = mp.Medium(index=ng)

wvl = 0.5  # wavelength
fcen = 1 / wvl

# rectangular supercell
sx = 1.0
sy = np.sqrt(3)

dpml = 1.0  # PML thickness
dsub = 2.0  # substrate thickness
dair = 2.0  # air padding
hcyl = 0.5  # cylinder height
rcyl = 0.1  # cylinder radius

sz = dpml + dsub + hcyl + dair + dpml

cell_size = mp.Vector3(sx, sy, sz)

boundary_layers = [mp.PML(thickness=dpml, direction=mp.Z)]

# periodic boundary conditions
k_point = mp.Vector3()

src_pt = mp.Vector3(0, 0, -0.5 * sz + dpml)
sources = [
    mp.Source(
        src=mp.GaussianSource(fcen, fwidth=0.1 * fcen),
        size=mp.Vector3(sx, sy, 0),
        center=src_pt,
        component=mp.Ex,
    )
]

substrate = [
    mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, dpml + dsub),
        center=mp.Vector3(0, 0, -0.5 * sz + 0.5 * (dpml + dsub)),
        material=glass,
    )
]

cyl_grating = [
    mp.Cylinder(
        center=mp.Vector3(0, 0, -0.5 * sz + dpml + dsub + 0.5 * hcyl),
        radius=rcyl,
        height=hcyl,
        material=glass,
    ),
    mp.Cylinder(
        center=mp.Vector3(0.5 * sx, 0.5 * sy, -0.5 * sz + dpml + dsub + 0.5 * hcyl),
        radius=rcyl,
        height=hcyl,
        material=glass,
    ),
    mp.Cylinder(
        center=mp.Vector3(-0.5 * sx, 0.5 * sy, -0.5 * sz + dpml + dsub + 0.5 * hcyl),
        radius=rcyl,
        height=hcyl,
        material=glass,
    ),
    mp.Cylinder(
        center=mp.Vector3(-0.5 * sx, -0.5 * sy, -0.5 * sz + dpml + dsub + 0.5 * hcyl),
        radius=rcyl,
        height=hcyl,
        material=glass,
    ),
    mp.Cylinder(
        center=mp.Vector3(0.5 * sx, -0.5 * sy, -0.5 * sz + dpml + dsub + 0.5 * hcyl),
        radius=rcyl,
        height=hcyl,
        material=glass,
    ),
]

geometry = substrate + cyl_grating

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    sources=sources,
    geometry=geometry,
    boundary_layers=boundary_layers,
    k_point=k_point,
)

tran_pt = mp.Vector3(0, 0, 0.5 * sz - dpml)
tran_flux = sim.add_mode_monitor(
    fcen, 0, 1, mp.ModeRegion(center=tran_pt, size=mp.Vector3(sx, sy, 0))
)

sim.run(until_after_sources=mp.stop_when_fields_decayed(20, mp.Ex, src_pt, 1e-6)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True))

# diffraction order of unit cell (triangular lattice)
mx = 0
my = 1

# check: for diffraction orders of supercell for which
#        nx = mx and ny = -mx + 2*my and thus
#        only even orders should produce nonzero power
nx = mx
for ny in range(4):
    kz2 = fcen**2 - (nx / sx) ** 2 - (ny / sy) ** 2
    if kz2 > 0:
        res = sim.get_eigenmode_coefficients(
            tran_flux, mp.DiffractedPlanewave((nx, ny, 0), mp.Vector3(0, 1, 0), 1, 0)
        )
        t_coeffs = res.alpha
        tran = abs(t_coeffs[0, 0, 0]) ** 2

        print(f"order:, {nx}, {ny}, {tran:.5f}")
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
