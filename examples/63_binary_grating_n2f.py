"""
Binary Grating N2F.Py with OptixLog Integration

Grating diffraction analysis

Based on the Meep tutorial: binary_grating_n2f.py
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
        run_name="binary_grating_n2f_simulation",
        config={
            "simulation_type": "grating",
            "description": "Grating diffraction analysis",
            "framework": "meep",
            "original_file": "binary_grating_n2f.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=25,
        fcen=0.5,
        pad=3.0,
        dpml=1.0
    )

from numpy import linalg as LA

resolution = 25  # pixels/μm

dpml = 1.0  # PML thickness
dsub = 3.0  # substrate thickness
dpad = 3.0  # padding between grating and PML
gp = 10.0  # grating period
gh = 0.5  # grating height
gdc = 0.5  # grating duty cycle

nperiods = 10  # number of unit cells in finite periodic grating

ff_distance = 1e8  # far-field distance from near-field monitor
ff_angle = 20  # far-field cone angle
ff_npts = 500  # number of far-field points

ff_length = ff_distance * math.tan(math.radians(ff_angle))
ff_res = ff_npts / ff_length

sx = dpml + dsub + gh + dpad + dpml
cell_size = mp.Vector3(sx)

pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

symmetries = [mp.Mirror(mp.Y)]

wvl_min = 0.4  # min wavelength
wvl_max = 0.6  # max wavelength
fmin = 1 / wvl_max  # min frequency
fmax = 1 / wvl_min  # max frequency
fcen = 0.5 * (fmin + fmax)  # center frequency
df = fmax - fmin  # frequency width

src_pt = mp.Vector3(-0.5 * sx + dpml + 0.5 * dsub)
sources = [
    mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=src_pt)
]

k_point = mp.Vector3()

glass = mp.Medium(index=1.5)

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    k_point=k_point,
    default_material=glass,
    sources=sources,
)

nfreq = 21
n2f_pt = mp.Vector3(0.5 * sx - dpml - 0.5 * dpad)
n2f_obj = sim.add_near2far(fcen, df, nfreq, mp.Near2FarRegion(center=n2f_pt))

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, n2f_pt, 1e-9)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True))

ff_source = sim.get_farfields(
    n2f_obj,
    ff_res,
    center=mp.Vector3(ff_distance, 0.5 * ff_length),
    size=mp.Vector3(y=ff_length),
)

sim.reset_meep()

### unit cell with periodic boundaries

sy = gp
cell_size = mp.Vector3(sx, sy)

sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
        component=mp.Ez,
        center=src_pt,
        size=mp.Vector3(y=sy),
    )
]

geometry = [
    mp.Block(
        material=glass,
        size=mp.Vector3(dpml + dsub, mp.inf, mp.inf),
        center=mp.Vector3(-0.5 * sx + 0.5 * (dpml + dsub)),
    ),
    mp.Block(
        material=glass,
        size=mp.Vector3(gh, gdc * gp, mp.inf),
        center=mp.Vector3(-0.5 * sx + dpml + dsub + 0.5 * gh),
    ),
]

sim = mp.Simulation(
    resolution=resolution,
    split_chunks_evenly=True,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    k_point=k_point,
    sources=sources,
    symmetries=symmetries,
)

n2f_obj = sim.add_near2far(
    fcen,
    df,
    nfreq,
    mp.Near2FarRegion(center=n2f_pt, size=mp.Vector3(y=sy)),
    nperiods=nperiods,
)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, n2f_pt, 1e-9)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True))

ff_unitcell = sim.get_farfields(
    n2f_obj,
    ff_res,
    center=mp.Vector3(ff_distance, 0.5 * ff_length),
    size=mp.Vector3(y=ff_length),
)

sim.reset_meep()

### finite periodic grating with flat surface termination extending into PML

num_cells = 2 * nperiods + 1
sy = dpml + num_cells * gp + dpml
cell_size = mp.Vector3(sx, sy)

pml_layers = [mp.PML(thickness=dpml)]

sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
        component=mp.Ez,
        center=src_pt,
        size=mp.Vector3(y=sy),
    )
]

geometry = [
    mp.Block(
        material=glass,
        size=mp.Vector3(dpml + dsub, mp.inf, mp.inf),
        center=mp.Vector3(-0.5 * sx + 0.5 * (dpml + dsub)),
    )
]

for j in range(num_cells):
    geometry.append(
        mp.Block(
            material=glass,
            size=mp.Vector3(gh, gdc * gp, mp.inf),
            center=mp.Vector3(
                -0.5 * sx + dpml + dsub + 0.5 * gh, -0.5 * sy + dpml + (j + 0.5) * gp
            ),
        )
    )

sim = mp.Simulation(
    resolution=resolution,
    split_chunks_evenly=True,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    k_point=k_point,
    sources=sources,
    symmetries=symmetries,
)

n2f_obj = sim.add_near2far(
    fcen, df, nfreq, mp.Near2FarRegion(center=n2f_pt, size=mp.Vector3(y=sy - 2 * dpml))
)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, n2f_pt, 1e-9)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True))

ff_supercell = sim.get_farfields(
    n2f_obj,
    ff_res,
    center=mp.Vector3(ff_distance, 0.5 * ff_length),
    size=mp.Vector3(y=ff_length),
)

norm_err = LA.norm(ff_unitcell["Ez"] - ff_supercell["Ez"]) / nperiods
print(f"error:, {nperiods}, {norm_err}")

freqs = mp.get_near2far_freqs(n2f_obj)
wvl = np.divide(1, freqs)
ff_lengths = np.linspace(0, ff_length, ff_npts)
angles = [math.degrees(math.atan(f)) for f in ff_lengths / ff_distance]

wvl_slice = 0.5
idx_slice = np.where(np.asarray(freqs) == 1 / wvl_slice)[0][0]

rel_enh = np.absolute(ff_unitcell["Ez"]) ** 2 / np.absolute(ff_source["Ez"]) ** 2

plt.figure(dpi=150)

plt.subplot(1, 2, 1)
plt.pcolormesh(wvl, angles, rel_enh, cmap="Blues", shading="flat")
plt.axis([wvl_min, wvl_max, 0, ff_angle])
plt.xlabel("wavelength (μm)")
plt.ylabel("angle (degrees)")
plt.grid(linewidth=0.5, linestyle="--")
plt.xticks([t for t in np.arange(wvl_min, wvl_max + 0.1, 0.1)])
plt.yticks([t for t in range(0, ff_angle + 1, 10)])
plt.title("far-field spectra")

plt.subplot(1, 2, 2)
plt.plot(angles, rel_enh[:, idx_slice], "bo-")
plt.xlim(0, ff_angle)
plt.ylim(0)
plt.xticks([t for t in range(0, ff_angle + 1, 10)])
plt.xlabel("angle (degrees)")
plt.ylabel("relative enhancement")
plt.grid(axis="x", linewidth=0.5, linestyle="--")
plt.title(f"f.-f. spectra @  λ = {wvl_slice:.1} μm")

plt.tight_layout(pad=0.5)
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
