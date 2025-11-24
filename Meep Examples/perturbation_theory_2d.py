"""
Perturbation Theory 2D.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: perturbation_theory_2d.py
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
        run_name="perturbation_theory_2d_simulation",
        config={
            "simulation_type": "general",
            "description": "Electromagnetic simulation",
            "framework": "meep",
            "original_file": "perturbation_theory_2d.py"
        },
        create_project_if_not_exists=True
    )
    print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
    print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")


    # Log simulation parameters
    client.log(step=0,
        fcen=0.21,
        df=0.2,
        w=1,
        r=1,
        pad=4,
        dpml=2
    )

import argparse

def main(args):
    if args.perpendicular:
        src_cmpt = mp.Hz
        fcen = 0.21  # pulse center frequency
    else:
        src_cmpt = mp.Ez
        fcen = 0.17  # pulse center frequency

    n = 3.4  # index of waveguide
    w = 1  # ring width
    r = 1  # inner radius of ring
    pad = 4  # padding between waveguide and edge of PML
    dpml = 2  # thickness of PML

    pml_layers = [mp.PML(dpml)]

    sxy = 2 * (r + w + pad + dpml)
    cell_size = mp.Vector3(sxy, sxy)

    symmetries = [
        mp.Mirror(mp.X, phase=+1 if args.perpendicular else -1),
        mp.Mirror(mp.Y, phase=-1 if args.perpendicular else +1),
    ]

    geometry = [
        mp.Cylinder(
            material=mp.Medium(index=n),
            radius=r + w,
            height=mp.inf,
            center=mp.Vector3(),
        ),
        mp.Cylinder(material=mp.vacuum, radius=r, height=mp.inf, center=mp.Vector3()),
    ]

    # find resonant frequency of unperturbed geometry using broadband source

    df = 0.2 * fcen  # pulse width (in frequency)

    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=src_cmpt,
            center=mp.Vector3(r + 0.1),
        ),
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=src_cmpt,
            center=mp.Vector3(-(r + 0.1)),
            amplitude=-1,
        ),
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        boundary_layers=pml_layers,
        resolution=args.res,
        sources=sources,
        symmetries=symmetries,
    )

    h = mp.Harminv(src_cmpt, mp.Vector3(r + 0.1), fcen, df)
    sim.run(mp.after_sources(h)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True), until_after_sources=100)

    frq_unperturbed = h.modes[0].freq

    sim.reset_meep()

    # unperturbed geometry with narrowband source centered at resonant frequency

    fcen = frq_unperturbed
    df = 0.05 * fcen

    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=src_cmpt,
            center=mp.Vector3(r + 0.1),
        ),
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=src_cmpt,
            center=mp.Vector3(-(r + 0.1)),
            amplitude=-1,
        ),
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        boundary_layers=pml_layers,
        resolution=args.res,
        sources=sources,
        symmetries=symmetries,
    )

    sim.run(until_after_sources=100)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True)

    deps = 1 - n**2
    deps_inv = 1 - 1 / n**2

    if args.perpendicular:
        para_integral = (
            deps
            * 2
            * np.pi
            * (
                r * abs(sim.get_field_point(mp.Ey, mp.Vector3(r))) ** 2
                - (r + w) * abs(sim.get_field_point(mp.Ey, mp.Vector3(r + w))) ** 2
            )
        )
        perp_integral = (
            deps_inv
            * 2
            * np.pi
            * (
                -r * abs(sim.get_field_point(mp.Dy, mp.Vector3(y=r))) ** 2
                + (r + w) * abs(sim.get_field_point(mp.Dy, mp.Vector3(y=r + w))) ** 2
            )
        )
        numerator_integral = para_integral + perp_integral
    else:
        numerator_integral = (
            deps
            * 2
            * np.pi
            * (
                r * abs(sim.get_field_point(mp.Ez, mp.Vector3(r))) ** 2
                - (r + w) * abs(sim.get_field_point(mp.Ez, mp.Vector3(r + w))) ** 2
            )
        )

    denominator_integral = sim.electric_energy_in_box(
        center=mp.Vector3(), size=mp.Vector3(sxy - 2 * dpml, sxy - 2 * dpml)
    )
    perturb_theory_dw_dR = (
        -frq_unperturbed * numerator_integral / (8 * denominator_integral)
    )

    # perturbed geometry with narrowband source

    dr = 0.04

    sim.reset_meep()

    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=src_cmpt,
            center=mp.Vector3(r + dr + 0.1),
        ),
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=src_cmpt,
            center=mp.Vector3(-(r + dr + 0.1)),
            amplitude=-1,
        ),
    ]

    geometry = [
        mp.Cylinder(
            material=mp.Medium(index=n),
            radius=r + dr + w,
            height=mp.inf,
            center=mp.Vector3(),
        ),
        mp.Cylinder(
            material=mp.vacuum, radius=r + dr, height=mp.inf, center=mp.Vector3()
        ),
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        boundary_layers=pml_layers,
        resolution=args.res,
        sources=sources,
        symmetries=symmetries,
    )

    h = mp.Harminv(src_cmpt, mp.Vector3(r + dr + 0.1), fcen, df)
    sim.run(mp.after_sources(h)
    
    # Log simulation completion
    client.log(step=1, simulation_completed=True), until_after_sources=100)

    frq_perturbed = h.modes[0].freq

    finite_diff_dw_dR = (frq_perturbed - frq_unperturbed) / dr

    print(
        f"dwdR:, {perturb_theory_dw_dR} (pert. theory), {finite_diff_dw_dR} (finite diff.)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-perpendicular",
        action="store_true",
        help="use perpendicular field source (default: parallel field source)",
    )
    parser.add_argument(
        "-res", type=int, default=30, help="resolution (default: 30 pixels/um)"
    )
    args = parser.parse_args()
    main(args)
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
