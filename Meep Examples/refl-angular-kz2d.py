"""
Refl Angular Kz2D.Py with OptixLog Integration

Electromagnetic simulation

Based on the Meep tutorial: refl-angular-kz2d.py
"""

import os
import math
import optixlog
import meep as mp
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

# OptixLog Configuration
api_key = os.getenv("OPTIX_API_KEY", "proj_YOUR_API_KEY_HERE")
api_url = os.getenv("OPTIX_API_URL", "https://backend.optixlog.com")
project_name = os.getenv("OPTIX_PROJECT", "MeepExamples")

def main():
    """Main simulation function with OptixLog integration"""
    
    # Check if this is the master process
    if not optixlog.is_master_process():
        mpi_info = optixlog.get_mpi_info()
        print(f"Worker process (rank {mpi_info[1]}/{mpi_info[2]}) - skipping simulation")
        return
    
    print(f"🚀 Initializing OptixLog client for project: {project_name}")
    
    try:
        # Initialize OptixLog client
        client = optixlog.init(
            api_key=api_key,
            api_url=api_url,
            project=project_name,
            run_name="refl-angular-kz2d_simulation",
            config={
                "simulation_type": "general",
                "description": "Electromagnetic simulation",
                "framework": "meep",
                "original_file": "refl-angular-kz2d.py"
            },
            create_project_if_not_exists=True
        )
        print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
        print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")
        
        # Log simulation parameters
        client.log(step=0,
            resolution=100,
            fcen=1.0,
            dpml=1.0
        )
    except Exception as e:
        print(f"❌ Failed to initialize OptixLog: {e}")
        client = None

    def refl_planar(theta, kz_2d):
        resolution = 100

        dpml = 1.0
        sx = 10
        sx = 10 + 2 * dpml
        cell_size = mp.Vector3(sx)
        pml_layers = [mp.PML(dpml)]

        fcen = 1.0

        # plane of incidence is XZ
        k = mp.Vector3(z=math.sin(theta)).scale(fcen)

        sources = [
            mp.Source(
                mp.GaussianSource(fcen, fwidth=0.2 * fcen),
                component=mp.Ey,
                center=mp.Vector3(-0.5 * sx + dpml),
            )
        ]

        sim = mp.Simulation(
            cell_size=cell_size,
            boundary_layers=pml_layers,
            sources=sources,
            k_point=k,
            kz_2d=kz_2d,
            resolution=resolution,
        )

        refl_fr = mp.FluxRegion(center=mp.Vector3(-0.25 * sx))
        refl = sim.add_flux(fcen, 0, 1, refl_fr)

        sim.run(
            until_after_sources=mp.stop_when_fields_decayed(
                50, mp.Ey, mp.Vector3(-0.5 * sx + dpml), 1e-9
            )
        )

        input_flux = mp.get_fluxes(refl)
        input_data = sim.get_flux_data(refl)
        sim.reset_meep()

        # add a block with n=3.5 for the air-dielectric interface
        geometry = [
            mp.Block(
                size=mp.Vector3(0.5 * sx, mp.inf, mp.inf),
                center=mp.Vector3(0.25 * sx),
                material=mp.Medium(index=3.5),
            )
        ]

        sim = mp.Simulation(
            cell_size=cell_size,
            geometry=geometry,
            boundary_layers=pml_layers,
            sources=sources,
            k_point=k,
            kz_2d=kz_2d,
            resolution=resolution,
        )

        refl = sim.add_flux(fcen, 0, 1, refl_fr)
        sim.load_minus_flux_data(refl, input_data)

        sim.run(
            until_after_sources=mp.stop_when_fields_decayed(
                50, mp.Ey, mp.Vector3(-0.5 * sx + dpml), 1e-9
            )
        )

        refl_flux = mp.get_fluxes(refl)
        freqs = mp.get_flux_freqs(refl)

        return -refl_flux[0] / input_flux[0]

    # rotation angle of source: CCW around Y axis, 0 degrees along +X axis
    theta_r = math.radians(19.4)

    Rmeep_real_imag = refl_planar(theta_r, "real/imag")
    Rmeep_complex = refl_planar(theta_r, "complex")
    Rmeep_3d = refl_planar(theta_r, "3d")

    n1 = 1
    n2 = 3.5

    # compute angle of refracted planewave in medium n2
    # for incident planewave in medium n1 at angle theta_in
    theta_out = lambda theta_in: math.asin(n1 * math.sin(theta_in) / n2)

    # compute Fresnel reflectance for S-polarization in medium n2
    # for incident planewave in medium n1 at angle theta_in
    Rfresnel = (
        lambda theta_in: math.fabs(
            (n2 * math.cos(theta_out(theta_in)) - n1 * math.cos(theta_in))
            / (n2 * math.cos(theta_out(theta_in)) + n1 * math.cos(theta_in))
        )
        ** 2
    )

    print(
        f"refl:, {Rmeep_real_imag} (real/imag), {Rmeep_complex} (complex), {Rmeep_3d} (3d), {Rfresnel(theta_r)} (analytic)"
    )
    
    # Log simulation completion
    if client:
        client.log(step=1, simulation_completed=True)

if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"\n❌ OptixLog Error: {e}")
        print("Please ensure your API key and URL are correct.")
    except Exception as e:
        print(f"\n❌ Simulation Error: {e}")
    finally:
        # Clean up generated files
        import glob
        for file_path in glob.glob("*.png") + glob.glob("*.csv") + glob.glob("*.txt"):
            if os.path.exists(file_path):
                os.remove(file_path)
