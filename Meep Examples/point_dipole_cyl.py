"""
Point Dipole Cyl.Py with OptixLog Integration

Dipole radiation analysis

Based on the Meep tutorial: point_dipole_cyl.py
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
        run_name="point_dipole_cyl_simulation",
        config={
            "simulation_type": "dipole",
            "description": "Dipole radiation analysis",
            "framework": "meep",
            "original_file": "point_dipole_cyl.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        r=0
    )

"""Tutorial example for dipole current sources in cylindrical coordinates.

tutorial reference:
https://meep.readthedocs.io/en/latest/Python_Tutorials/Cylindrical_Coordinates/#nonaxisymmetric-dipole-sources
"""

from typing import Tuple

RESOLUTION_UM = 50
WAVELENGTH_UM = 1.0
N_SLAB = 2.4
SLAB_THICKNESS_UM = 0.7 * WAVELENGTH_UM / N_SLAB


def dipole_in_slab(zpos: float, rpos_um: float, m: int) -> Tuple[float, float]:
    """Computes the flux from a dipole in a slab.

    Args:
      zpos: position of dipole as a fraction of layer thickness.
      rpos_um: position of source in radial direction.
      m: angular φ dependence of the fields exp(imφ).

    Returns:
      A 2-tuple of the radiated and total flux.
    """
    pml_um = 1.0  # thickness of PML
    padding_um = 1.0  # thickness of air padding
    r_um = 20.0  # length of cell in r

    frequency = 1 / WAVELENGTH_UM  # center frequency of source/monitor

    # runtime termination criteria
    flux_decay_threshold = 1e-4

    size_r = r_um + pml_um
    size_z = SLAB_THICKNESS_UM + padding_um + pml_um
    cell_size = mp.Vector3(size_r, 0, size_z)

    boundary_layers = [
        mp.PML(pml_um, direction=mp.R),
        mp.PML(pml_um, direction=mp.Z, side=mp.High),
    ]

    src_pt = mp.Vector3(rpos_um, 0, -0.5 * size_z + zpos * SLAB_THICKNESS_UM)
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.05 * frequency),
            component=mp.Er,
            center=src_pt,
        ),
    ]

    geometry = [
        mp.Block(
            material=mp.Medium(index=N_SLAB),
            center=mp.Vector3(0, 0, -0.5 * size_z + 0.5 * SLAB_THICKNESS_UM),
            size=mp.Vector3(mp.inf, mp.inf, SLAB_THICKNESS_UM),
        )
    ]

    sim = mp.Simulation(
        resolution=RESOLUTION_UM,
        cell_size=cell_size,
        dimensions=mp.CYLINDRICAL,
        m=m,
        boundary_layers=boundary_layers,
        sources=sources,
        geometry=geometry,
        force_complex_fields=True,
    )

    flux_mon = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(0.5 * r_um, 0, 0.5 * size_z - pml_um),
            size=mp.Vector3(r_um, 0, 0),
        ),
        mp.FluxRegion(
            center=mp.Vector3(r_um, 0, 0.5 * size_z - pml_um - 0.5 * padding_um),
            size=mp.Vector3(0, 0, padding_um),
        ),
    )

    sim.run(
        mp.dft_ldos(frequency, 0, 1)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True),
        until_after_sources=mp.stop_when_dft_decayed(tol=flux_decay_threshold),
    )

    radiated_flux = mp.get_fluxes(flux_mon)[0]

    # volume of the ring current source
    delta_vol = 2 * np.pi * rpos_um / (RESOLUTION_UM**2)

    # total flux from point source via LDOS
    source_flux = -np.real(sim.ldos_Fdata[0] * np.conj(sim.ldos_Jdata[0])) * delta_vol

    print(
        f"flux-cyl:, {rpos_um:.2f}, {m:3d}, " f"{source_flux:.6f}, {radiated_flux:.6f}"
    )

    return radiated_flux, source_flux


if __name__ == "__main__":
    dipole_height = 0.5

    # An Er source at r = 0 needs to be slightly offset.
    # https://github.com/NanoComp/meep/issues/2704
    dipole_rpos_um = 1.5 / RESOLUTION_UM

    # Er source at r = 0 requires a single simulation with m = ±1.
    m = 1
    radiated_flux, source_flux = dipole_in_slab(
        dipole_height,
        dipole_rpos_um,
        m,
    )
    extraction_efficiency = radiated_flux / source_flux
    print(f"exteff:, {dipole_rpos_um}, {extraction_efficiency:.6f}")

    # Er source at r > 0 requires Fourier-series expansion of φ.

    # Threshold flux to determine when to truncate expansion.
    flux_decay_threshold = 1e-2

    dipole_rpos_um = [3.5, 6.7, 9.5]
    for rpos_um in dipole_rpos_um:
        source_flux_total = 0
        radiated_flux_total = 0
        radiated_flux_max = 0
        m = 0
        while True:
            radiated_flux, source_flux = dipole_in_slab(
                dipole_height,
                rpos_um,
                m,
            )
            radiated_flux_total += radiated_flux * (1 if m == 0 else 2)
            source_flux_total += source_flux * (1 if m == 0 else 2)

            if radiated_flux > radiated_flux_max:
                radiated_flux_max = radiated_flux

            if m > 0 and (radiated_flux / radiated_flux_max) < flux_decay_threshold:
                break
            else:
                m += 1

        extraction_efficiency = radiated_flux_total / source_flux_total
        print(f"exteff:, {rpos_um}, {extraction_efficiency:.6f}")
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
