"""
Antenna Pec Ground Plane.Py with OptixLog Integration

Antenna radiation analysis

Based on the Meep tutorial: antenna_pec_ground_plane.py
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
            run_name="antenna_pec_ground_plane_simulation",
            config={
                "simulation_type": "antenna",
                "description": "Antenna radiation analysis",
                "framework": "meep",
                "original_file": "antenna_pec_ground_plane.py"
            },
            create_project_if_not_exists=True
        )
        print(f"✅ OptixLog client initialized. Run ID: {client.run_id}")
        print(f"🔗 View run at: https://optixlog.com/runs/{client.run_id}")
        
        # Log simulation parameters
        client.log(step=0,
            resolution=200,
            fcen=1,
            r=1000,
            dpml=1
        )
    except Exception as e:
        print(f"❌ Failed to initialize OptixLog: {e}")
        client = None

    # Computes the radiation pattern of a dipole antenna
    # positioned a given height above a perfect-electric
    # conductor (PEC) ground plane and compares the result
    # to analytic theory.
    
    resolution = 200  # pixels/um
    n = 1.2  # refractive index of surrounding medium
    h = 1.25  # height of antenna (point dipole source) above ground plane
    wvl = 0.65  # vacuum wavelength
    r = 1000 * wvl  # radius of far-field circle
    npts = 50  # number of points in [0,pi/2) range of angles

    angles = 0.5 * math.pi / npts * np.arange(npts)

    def radial_flux(sim, nearfield_box, r):
        E = np.zeros((npts, 3), dtype=np.complex128)
        H = np.zeros((npts, 3), dtype=np.complex128)

        for n in range(npts):
            ff = sim.get_farfield(
                nearfield_box, mp.Vector3(r * math.sin(angles[n]), r * math.cos(angles[n]))
            )
            E[n, :] = [np.conj(ff[j]) for j in range(3)]
            H[n, :] = [ff[j + 3] for j in range(3)]

        Px = np.real(E[:, 1] * H[:, 2] - E[:, 2] * H[:, 1])  # Ey*Hz-Ez*Hy
        Py = np.real(E[:, 2] * H[:, 0] - E[:, 0] * H[:, 2])  # Ez*Hx-Ex*Hz
        return np.sqrt(np.square(Px) + np.square(Py))

    def free_space_radiation(src_cmpt):
        sxy = 4
        dpml = 1
        cell_size = mp.Vector3(sxy + 2 * dpml, sxy + 2 * dpml)
        pml_layers = [mp.PML(dpml)]

        fcen = 1 / wvl
        sources = [
            mp.Source(
                src=mp.GaussianSource(fcen, fwidth=0.2 * fcen),
                center=mp.Vector3(),
                component=src_cmpt,
            )
        ]

        if src_cmpt == mp.Hz:
            symmetries = [mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=-1)]
        elif src_cmpt == mp.Ez:
            symmetries = [mp.Mirror(mp.X, phase=+1), mp.Mirror(mp.Y, phase=+1)]
        else:
            symmetries = []

        sim = mp.Simulation(
            cell_size=cell_size,
            resolution=resolution,
            sources=sources,
            symmetries=symmetries,
            boundary_layers=pml_layers,
            default_material=mp.Medium(index=n),
        )

        nearfield_box = sim.add_near2far(
            fcen,
            0,
            1,
            mp.Near2FarRegion(center=mp.Vector3(0, +0.5 * sxy), size=mp.Vector3(sxy, 0)),
            mp.Near2FarRegion(
                center=mp.Vector3(0, -0.5 * sxy), size=mp.Vector3(sxy, 0), weight=-1
            ),
            mp.Near2FarRegion(center=mp.Vector3(+0.5 * sxy, 0), size=mp.Vector3(0, sxy)),
            mp.Near2FarRegion(
                center=mp.Vector3(-0.5 * sxy, 0), size=mp.Vector3(0, sxy), weight=-1
            ),
        )

        sim.run(until_after_sources=mp.stop_when_dft_decayed())

        return radial_flux(sim, nearfield_box, r)

    def pec_ground_plane_radiation(src_cmpt=mp.Hz):
        L = 8.0  # length of non-PML region
        dpml = 1.0  # thickness of PML
        sxy = dpml + L + dpml
        cell_size = mp.Vector3(sxy, sxy, 0)
        boundary_layers = [mp.PML(dpml)]

        fcen = 1 / wvl

        # The near-to-far field transformation feature only supports
        # homogeneous media which means it cannot explicitly take into
        # account the ground plane. As a workaround, we use two antennas
        # of opposite sign surrounded by a single near2far box which
        # encloses both antennas. We then use an odd mirror symmetry to
        # divide the computational cell in half which is effectively
        # equivalent to a PEC boundary condition on one side.
        # Note: This setup means that the radiation pattern can only
        # be measured in the top half above the dipole.
        sources = [
            mp.Source(
                src=mp.GaussianSource(fcen, fwidth=0.2 * fcen),
                component=src_cmpt,
                center=mp.Vector3(0, +h),
            ),
            mp.Source(
                src=mp.GaussianSource(fcen, fwidth=0.2 * fcen),
                component=src_cmpt,
                center=mp.Vector3(0, -h),
                amplitude=-1,
            ),
        ]

        if src_cmpt == mp.Hz:
            symmetries = [mp.Mirror(mp.X, phase=-1)]
        elif src_cmpt == mp.Ez:
            symmetries = [mp.Mirror(mp.X, phase=+1)]
        else:
            symmetries = []

        sim = mp.Simulation(
            cell_size=cell_size,
            resolution=resolution,
            sources=sources,
            symmetries=symmetries,
            boundary_layers=boundary_layers,
            default_material=mp.Medium(index=n),
        )

        nearfield_box = sim.add_near2far(
            fcen,
            0,
            1,
            mp.Near2FarRegion(
                center=mp.Vector3(0, h), size=mp.Vector3(4 * h, 0), weight=+1
            ),
            mp.Near2FarRegion(
                center=mp.Vector3(0, -h), size=mp.Vector3(4 * h, 0), weight=-1
            ),
            mp.Near2FarRegion(
                center=mp.Vector3(h, 0), size=mp.Vector3(0, 4 * h), weight=+1
            ),
            mp.Near2FarRegion(
                center=mp.Vector3(-h, 0), size=mp.Vector3(0, 4 * h), weight=-1
            ),
        )

        sim.plot2D()
        plt.savefig("antenna_pec_ground_plane.png", bbox_inches="tight")

        sim.run(until_after_sources=mp.stop_when_dft_decayed())

        return radial_flux(sim, nearfield_box, r)

    # Run the simulations
    src_cmpt = mp.Ez  # TM/P: Hz or TE/S: Ez
    Pr_fsp = free_space_radiation(src_cmpt)
    Pr_pec = pec_ground_plane_radiation(src_cmpt)

    # The radiation pattern of a two-element antenna
    # array is equivalent to the radiation pattern of
    # a single antenna multiplied by its array factor
    # as derived in Section 6.2 "Two-Element Array" of
    # Antenna Theory: Analysis and Design, Fourth Edition
    # (2016) by C.A. Balanis.
    k = 2 * np.pi / (wvl / n)  # wavevector in free space
    Pr_theory = np.zeros(npts)
    for i, ang in enumerate(angles):
        Pr_theory[i] = Pr_fsp[i] * 2 * np.sin(k * h * np.cos(ang))

    Pr_pec_norm = Pr_pec / np.max(Pr_pec)
    Pr_theory_norm = (Pr_theory / max(Pr_theory)) ** 2

    plt.figure()
    plt.plot(np.degrees(angles), Pr_pec_norm, "b-", label="Meep")
    plt.plot(np.degrees(angles), Pr_theory_norm, "r-", label="theory")
    plt.xlabel("angle (degrees)")
    plt.ylabel("radial flux (normalized by maximum flux)")
    plt.title(
        f"antenna with {'E' if src_cmpt==mp.Ez else 'H'}$_z$ polarization above PEC ground plane"
    )

    plt.axis([0, 90, 0, 1.0])
    plt.legend()
    plt.savefig("radiation_pattern.png", bbox_inches="tight")

    print(f"norm:, {np.linalg.norm(Pr_pec_norm - Pr_theory_norm):.6f}")
    
    # Calculate additional metrics for comprehensive analysis
    if client:
        # Calculate radiation pattern statistics
        max_pec_angle = np.degrees(angles[np.argmax(Pr_pec_norm)])
        max_theory_angle = np.degrees(angles[np.argmax(Pr_theory_norm)])
        
        # Calculate beamwidth (3dB points)
        pec_3db_level = np.max(Pr_pec_norm) / 2
        theory_3db_level = np.max(Pr_theory_norm) / 2
        
        pec_3db_indices = np.where(Pr_pec_norm >= pec_3db_level)[0]
        theory_3db_indices = np.where(Pr_theory_norm >= theory_3db_level)[0]
        
        pec_beamwidth = np.degrees(angles[pec_3db_indices[-1]] - angles[pec_3db_indices[0]]) if len(pec_3db_indices) > 1 else 0
        theory_beamwidth = np.degrees(angles[theory_3db_indices[-1]] - angles[theory_3db_indices[0]]) if len(theory_3db_indices) > 1 else 0
        
        # Calculate directivity (approximate)
        pec_directivity = 2 * np.max(Pr_pec_norm) / np.mean(Pr_pec_norm)
        theory_directivity = 2 * np.max(Pr_theory_norm) / np.mean(Pr_theory_norm)
        
        # Calculate efficiency metrics
        correlation_coeff = np.corrcoef(Pr_pec_norm, Pr_theory_norm)[0, 1]
        relative_error = np.mean(np.abs(Pr_pec_norm - Pr_theory_norm) / Pr_theory_norm) * 100
        
        # Log simulation completion
        client.log(step=1, simulation_completed=True)
        
        # Log the radiation pattern plot
        from PIL import Image
        radiation_img = Image.open("radiation_pattern.png")
        client.log_image("radiation_pattern", radiation_img, 
                        {"description": "Antenna radiation pattern comparison with PEC ground plane"})
        
        # Log the antenna setup plot
        setup_img = Image.open("antenna_pec_ground_plane.png")
        client.log_image("antenna_setup", setup_img, 
                        {"description": "Antenna setup with PEC ground plane"})
        
        # Log comprehensive radiation pattern metrics
        client.log(step=2,
                  # Basic comparison metrics
                  norm_difference=np.linalg.norm(Pr_pec_norm - Pr_theory_norm),
                  correlation_coefficient=correlation_coeff,
                  relative_error_percent=relative_error,
                  
                  # Peak radiation metrics
                  max_radiation_pec=np.max(Pr_pec_norm),
                  max_radiation_theory=np.max(Pr_theory_norm),
                  max_pec_angle_degrees=max_pec_angle,
                  max_theory_angle_degrees=max_theory_angle,
                  
                  # Beamwidth analysis
                  pec_beamwidth_degrees=pec_beamwidth,
                  theory_beamwidth_degrees=theory_beamwidth,
                  beamwidth_difference=abs(pec_beamwidth - theory_beamwidth),
                  
                  # Directivity analysis
                  pec_directivity=pec_directivity,
                  theory_directivity=theory_directivity,
                  directivity_error_percent=abs(pec_directivity - theory_directivity) / theory_directivity * 100,
                  
                  # Simulation parameters
                  antenna_height=h,
                  frequency_ghz=1/wvl,
                  wavelength_um=wvl,
                  refractive_index=n,
                  resolution=resolution,
                  dpml_thickness=1.0,
                  polarization="Ez" if src_cmpt==mp.Ez else "Hz",
                  
                  # Array analysis
                  wavevector=k,
                  array_factor_max=np.max(2 * np.sin(k * h * np.cos(angles))),
                  array_factor_min=np.min(2 * np.sin(k * h * np.cos(angles))),
                  
                  # Pattern statistics
                  pec_pattern_std=np.std(Pr_pec_norm),
                  theory_pattern_std=np.std(Pr_theory_norm),
                  pec_pattern_mean=np.mean(Pr_pec_norm),
                  theory_pattern_mean=np.mean(Pr_theory_norm))
        
        # Log detailed angle-by-angle analysis
        client.log(step=3,
                  angles_degrees=np.degrees(angles).tolist(),
                  pec_radiation_pattern=Pr_pec_norm.tolist(),
                  theory_radiation_pattern=Pr_theory_norm.tolist(),
                  free_space_pattern=Pr_fsp.tolist(),
                  array_factor_pattern=(2 * np.sin(k * h * np.cos(angles))).tolist(),
                  pattern_differences=(Pr_pec_norm - Pr_theory_norm).tolist())
        
        # Print comprehensive analysis summary
        print(f"\n📊 Antenna Radiation Analysis:")
        print(f"   Correlation coefficient: {correlation_coeff:.6f}")
        print(f"   Relative error: {relative_error:.2f}%")
        print(f"   Max PEC radiation at: {max_pec_angle:.1f}°")
        print(f"   Max theory radiation at: {max_theory_angle:.1f}°")
        print(f"   PEC beamwidth: {pec_beamwidth:.1f}°")
        print(f"   Theory beamwidth: {theory_beamwidth:.1f}°")
        print(f"   PEC directivity: {pec_directivity:.2f}")
        print(f"   Theory directivity: {theory_directivity:.2f}")
        print(f"   Directivity error: {abs(pec_directivity - theory_directivity) / theory_directivity * 100:.2f}%")
        print(f"   Array factor range: {np.min(2 * np.sin(k * h * np.cos(angles))):.3f} to {np.max(2 * np.sin(k * h * np.cos(angles))):.3f}")
    else:
        print("⚠️ No OptixLog client available for detailed logging")

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