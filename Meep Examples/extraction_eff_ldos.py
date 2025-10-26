"""
Extraction Eff Ldos.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: extraction_eff_ldos.py
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
        run_name="extraction_eff_ldos_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "extraction_eff_ldos.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        resolution=80,
        fcen=1,
        r=1.0,
        dpml=0.5
    )

"""Computes the extraction efficiency in 3D and cylindrical coordinates.

Verifies that the extraction efficiency of a point dipole in a dielectric
layer above a lossless metallic ground plane computed in two different
coordinate systems agree.
"""

resolution = 80  # pixels/μm
dpml = 0.5  # thickness of PML
dair = 1.0  # thickness of air padding
L = 6.0  # length of non-PML region
n = 2.4  # refractive index of surrounding medium
wvl = 1.0  # wavelength (in vacuum)

fcen = 1 / wvl  # center frequency of source/monitor

# runtime termination criteria
tol = 1e-8


def extraction_eff_cyl(dmat: float, h: float) -> float:
    """Computes the extraction efficiency in cylindrical coordinates.

    Args:
      dmat: thickness of dielectric layer.
      h: height of dipole above ground plane as fraction of dmat.

    Returns:
      The extraction efficiency of the dipole within the dielecric layer.
    """
    sr = L + dpml
    sz = dmat + dair + dpml
    cell_size = mp.Vector3(sr, 0, sz)

    boundary_layers = [
        mp.PML(dpml, direction=mp.R),
        mp.PML(dpml, direction=mp.Z, side=mp.High),
    ]

    src_cmpt = mp.Er

    # Because (1) Er is not defined at r=0 on the Yee grid, and (2) there
    # seems to be a bug in the interpolation of an Er point source at r=0,
    # the source is placed at r=~Δr (just outside the first voxel).
    # This incurs a small error which decreases linearly with resolution.
    # Ref: https://github.com/NanoComp/meep/issues/2704
    src_pt = mp.Vector3(1.5 / resolution, 0, -0.5 * sz + h * dmat)

    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=0.1 * fcen),
            component=src_cmpt,
            center=src_pt,
        )
    ]

    geometry = [
        mp.Block(
            material=mp.Medium(index=n),
            center=mp.Vector3(0, 0, -0.5 * sz + 0.5 * dmat),
            size=mp.Vector3(mp.inf, mp.inf, dmat),
        )
    ]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        dimensions=mp.CYLINDRICAL,
        m=-1,
        boundary_layers=boundary_layers,
        sources=sources,
        geometry=geometry,
    )

    flux_air = sim.add_flux(
        fcen,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(0.5 * L, 0, 0.5 * sz - dpml),
            size=mp.Vector3(L, 0, 0),
        ),
        mp.FluxRegion(
            center=mp.Vector3(L, 0, 0.5 * sz - dpml - 0.5 * dair),
            size=mp.Vector3(0, 0, dair),
        ),
    )

    sim.run(
        mp.dft_ldos(fcen, 0, 1)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True),
        until_after_sources=mp.stop_when_fields_decayed(20, src_cmpt, src_pt, tol),
    )

    out_flux = mp.get_fluxes(flux_air)[0]
    if src_pt.x == 0:
        dV = np.pi / (resolution**3)
    else:
        dV = 2 * np.pi * src_pt.x / (resolution**2)
    total_flux = -np.real(sim.ldos_Fdata[0] * np.conj(sim.ldos_Jdata[0])) * dV
    ext_eff = out_flux / total_flux
    print(f"extraction efficiency (cyl):, " f"{dmat:.4f}, {h:.4f}, {ext_eff:.6f}")

    return ext_eff


def extraction_eff_3D(dmat: float, h: float) -> float:
    """Computes the extraction efficiency in 3D Cartesian coordinates.

    Args:
      dmat: thickness of dielectric layer.
      h: height of dipole above ground plane as fraction of dmat.

    Returns:
      The extraction efficiency of the dipole within the dielecric layer.
    """
    sxy = L + 2 * dpml
    sz = dmat + dair + dpml
    cell_size = mp.Vector3(sxy, sxy, sz)

    symmetries = [
        mp.Mirror(direction=mp.X, phase=-1),
        mp.Mirror(direction=mp.Y),
    ]

    boundary_layers = [
        mp.PML(dpml, direction=mp.X),
        mp.PML(dpml, direction=mp.Y),
        mp.PML(dpml, direction=mp.Z, side=mp.High),
    ]

    src_cmpt = mp.Ex
    src_pt = mp.Vector3(0, 0, -0.5 * sz + h * dmat)
    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=0.1 * fcen),
            component=src_cmpt,
            center=src_pt,
        )
    ]

    geometry = [
        mp.Block(
            material=mp.Medium(index=n),
            center=mp.Vector3(0, 0, -0.5 * sz + 0.5 * dmat),
            size=mp.Vector3(mp.inf, mp.inf, dmat),
        )
    ]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        sources=sources,
        geometry=geometry,
        symmetries=symmetries,
    )

    flux_air = sim.add_flux(
        fcen,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(0, 0, 0.5 * sz - dpml),
            size=mp.Vector3(L, L, 0),
        ),
        mp.FluxRegion(
            center=mp.Vector3(0.5 * L, 0, 0.5 * sz - dpml - 0.5 * dair),
            size=mp.Vector3(0, L, dair),
        ),
        mp.FluxRegion(
            center=mp.Vector3(-0.5 * L, 0, 0.5 * sz - dpml - 0.5 * dair),
            size=mp.Vector3(0, L, dair),
            weight=-1.0,
        ),
        mp.FluxRegion(
            center=mp.Vector3(0, 0.5 * L, 0.5 * sz - dpml - 0.5 * dair),
            size=mp.Vector3(L, 0, dair),
        ),
        mp.FluxRegion(
            center=mp.Vector3(0, -0.5 * L, 0.5 * sz - dpml - 0.5 * dair),
            size=mp.Vector3(L, 0, dair),
            weight=-1.0,
        ),
    )

    sim.run(
        mp.dft_ldos(fcen, 0, 1)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True),
        until_after_sources=mp.stop_when_fields_decayed(20, src_cmpt, src_pt, tol),
    )

    out_flux = mp.get_fluxes(flux_air)[0]
    dV = 1 / (resolution**3)
    total_flux = -np.real(sim.ldos_Fdata[0] * np.conj(sim.ldos_Jdata[0])) * dV
    ext_eff = out_flux / total_flux
    print(f"extraction efficiency (3D):, {dmat:.4f}, {h:.4f}, {ext_eff:.6f}")

    return ext_eff


if __name__ == "__main__":
    layer_thickness = 0.7 * wvl / n
    dipole_height = np.linspace(0.1, 0.9, 21)

    exteff_cyl = np.zeros(len(dipole_height))
    exteff_3D = np.zeros(len(dipole_height))
    for j in range(len(dipole_height)):
        exteff_cyl[j] = extraction_eff_cyl(layer_thickness, dipole_height[j])
        exteff_3D[j] = extraction_eff_3D(layer_thickness, dipole_height[j])

    plt.plot(dipole_height, exteff_cyl, "bo-", label="cylindrical")
    plt.plot(dipole_height, exteff_3D, "ro-", label="3D Cartesian")
    plt.xlabel("height of dipole above ground plane (fraction of layer thickness)")
    plt.ylabel("extraction efficiency")
    plt.legend()

    if mp.am_master():
        plt.savefig("extraction_eff_vs_dipole_height.png", dpi=150, bbox_inches="tight")
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
