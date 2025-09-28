import meep as mp
import csv
import numpy as np
import matplotlib.pyplot as plt
from Mie_theory import Mie_Solver
from meep.materials import Ag
import optixlog
import os
client = optixlog.init(
    # api_url / api_key / project are read from env by default in your client;
    # but passing explicitly is fine too:
    api_url=os.getenv("OPTIX_API_URL", "https://coupler.onrender.com"),
    api_key=os.getenv("OPTIX_API_KEY", "proj_rLe5i6YI6Ozgl8W8Y5G9"),
    project=os.getenv("OPTIX_PROJECT", "Examples"),
    run_name="mie_sphere_3d_0.3",
    config={
        "size_xyz": [2, 2, 2],
        "sphere_radius": 0.30,
        "n_sphere": 2.0,
        "lambda_range": [0.4, 0.7],
        "lambda_center": 0.55,
    }, create_project_if_not_exists=True
)
Size_x = 2
Size_y = 2
Size_z = 2
cell_size = mp.Vector3(Size_x,Size_y,Size_z)

# # set up geometry (nanosphere)
sphere_radius = 0.35
sphere_refractive_index = 2
sphere_material = mp.Medium(index = sphere_refractive_index)
# sphere_material = Ag

geometry = [mp.Sphere(material = sphere_material, center = mp.Vector3(), radius = sphere_radius)]

# set up the source (plane wave)
center_wavelength = 0.55
min_wavelength = 0.4
max_wavelength = 0.7
center_freq = 1.0 / center_wavelength

freq_width = 1.0/min_wavelength - 1.0/max_wavelength
source_position = - Size_y * 0.5 + 0.3
gaussian_pulse = mp.GaussianSource(frequency = center_freq, fwidth = 2*freq_width, is_integrated=True)
source = [mp.Source(gaussian_pulse,
                    component = mp.Ez,
                    center = mp.Vector3(0,source_position),
                    size = mp.Vector3(Size_x,0,Size_z))]

# # set up simulation object
step_size = min([sphere_radius, min_wavelength]) / 20
resolution = 1.0 / step_size # number of cells per micron
pmls = [mp.PML(thickness = 10 * step_size)]

sim = mp.Simulation(resolution = resolution,
                    cell_size = cell_size,
                    sources = source,
                    boundary_layers = pmls,
                    geometry = [])


# # add monitors
wavelengths = np.linspace(min_wavelength, max_wavelength, 41) # wavelengths for monitors
frequencies = 1.0 / wavelengths

# # Domain DFT monitor
dft_freqs = [1./min_wavelength, center_freq, 1./max_wavelength]
dft_fields = sim.add_dft_fields([mp.Ez],
                                dft_freqs,
                                center = mp.Vector3(),
                                size = mp.Vector3(Size_x,Size_y))

# Flux box monitors
box_size = sphere_radius * 4
box_mx = sim.add_flux(frequencies, mp.FluxRegion(center = mp.Vector3(x = -box_size / 2), size = mp.Vector3(0,box_size,box_size)))
box_px = sim.add_flux(frequencies, mp.FluxRegion(center = mp.Vector3(x = box_size / 2), size = mp.Vector3(0,box_size,box_size)))
box_my = sim.add_flux(frequencies,mp.FluxRegion(center=mp.Vector3(y=-box_size/2),size=mp.Vector3(x = box_size, z = box_size)))
box_py = sim.add_flux(frequencies,mp.FluxRegion(center=mp.Vector3(y=box_size/2),size=mp.Vector3(x = box_size, z = box_size)))
box_mz = sim.add_flux(frequencies,mp.FluxRegion(center=mp.Vector3(z=-box_size/2),size=mp.Vector3(x = box_size, y = box_size)))
box_pz = sim.add_flux(frequencies,mp.FluxRegion(center=mp.Vector3(z=box_size/2),size=mp.Vector3(x = box_size, y = box_size)))


# visualize the simulation domain
sim.plot2D(output_plane = mp.Volume(center = mp.Vector3(), size = mp.Vector3(Size_x,Size_y)))
plt.savefig("simXY_empty.png")
plt.close()

sim.plot2D(output_plane = mp.Volume(center = mp.Vector3(), size = mp.Vector3(Size_x,0,Size_z)))
plt.savefig("simXZ_empty.png")
plt.close()
client.log_file("simXY_empty.png", "simXY_empty.png", "image/png", meta={"stage":"empty"})
client.log_file("simXZ_empty.png", "simXZ_empty.png", "image/png", meta={"stage":"empty"})
# run the empty-domain simulation
# sim.run(until_after_sources = 10)
sim.run(until_after_sources = mp.stop_when_fields_decayed(10, mp.Ez, mp.Vector3(0,box_size/2,0), 1e-4))

# loop through frequencies, and plot DFT fields
for f in range(len(dft_freqs)):
    Ez_f = sim.get_dft_array(dft_fields,mp.Ez,f)
    plt.imshow(np.real(Ez_f),extent = [-Size_y/2, Size_y/2, -Size_x/2, Size_x/2])
    plt.savefig(f"Ez_f_{1./dft_freqs[f]}.png")
    fn = f"Ez_f_{1./dft_freqs[f]}.png"
    plt.close()
    client.log_file(fn, fn, "image/png", meta={"stage":"empty","wavelength":float(1./dft_freqs[f])})

# Store flux monitor data for empty simulation
box_mx_data_empty = sim.get_flux_data(box_mx)
box_px_data_empty = sim.get_flux_data(box_px)
box_my_data_empty = sim.get_flux_data(box_my)
box_py_data_empty = sim.get_flux_data(box_py)
box_mz_data_empty = sim.get_flux_data(box_mz)
box_pz_data_empty = sim.get_flux_data(box_pz)

incident_intensity = np.array(mp.get_fluxes(box_my)) / (box_size ** 2) # calculate incident intensity for normalization later

plt.plot(wavelengths,incident_intensity)
plt.savefig('Inc_Intensity.png')
client.log_file("Inc_Intensity.png", "Inc_Intensity.png", "image/png", meta={"stage":"empty"})
# also log summary scalar for quick glance
client.log(step=0, incident_peak=float(np.max(incident_intensity)))
plt.close()

# Re-setup simulation but with the scatterer this time
sim.reset_meep()

sim = mp.Simulation(resolution = resolution,
                    cell_size = cell_size,
                    sources = source,
                    boundary_layers = pmls,
                    geometry = geometry,
                    Courant = 0.3)

# DFT Monitor
dft_freqs = [1./min_wavelength, center_freq, 1./max_wavelength]
dft_fields = sim.add_dft_fields([mp.Ez],
                                dft_freqs,
                                center = mp.Vector3(),
                                size = mp.Vector3(Size_x,Size_y))

# Scattering flux monitors
box_mx = sim.add_flux(frequencies, mp.FluxRegion(center = mp.Vector3(x = -box_size / 2), size = mp.Vector3(0,box_size,box_size)))
box_px = sim.add_flux(frequencies, mp.FluxRegion(center = mp.Vector3(x = box_size / 2), size = mp.Vector3(0,box_size,box_size)))
box_my = sim.add_flux(frequencies,mp.FluxRegion(center=mp.Vector3(y=-box_size/2),size=mp.Vector3(x = box_size, z = box_size)))
box_py = sim.add_flux(frequencies,mp.FluxRegion(center=mp.Vector3(y=box_size/2),size=mp.Vector3(x = box_size, z = box_size)))
box_mz = sim.add_flux(frequencies,mp.FluxRegion(center=mp.Vector3(z=-box_size/2),size=mp.Vector3(x = box_size, y = box_size)))
box_pz = sim.add_flux(frequencies,mp.FluxRegion(center=mp.Vector3(z=box_size/2),size=mp.Vector3(x = box_size, y = box_size)))

# Absorption flux monitors
box_mx_abs = sim.add_flux(frequencies, mp.FluxRegion(center = mp.Vector3(x = -box_size / 2), size = mp.Vector3(0,box_size,box_size)))
box_px_abs = sim.add_flux(frequencies, mp.FluxRegion(center = mp.Vector3(x = box_size / 2), size = mp.Vector3(0,box_size,box_size)))
box_my_abs = sim.add_flux(frequencies,mp.FluxRegion(center=mp.Vector3(y=-box_size/2),size=mp.Vector3(x = box_size, z = box_size)))
box_py_abs = sim.add_flux(frequencies,mp.FluxRegion(center=mp.Vector3(y=box_size/2),size=mp.Vector3(x = box_size, z = box_size)))
box_mz_abs = sim.add_flux(frequencies,mp.FluxRegion(center=mp.Vector3(z=-box_size/2),size=mp.Vector3(x = box_size, y = box_size)))
box_pz_abs = sim.add_flux(frequencies,mp.FluxRegion(center=mp.Vector3(z=box_size/2),size=mp.Vector3(x = box_size, y = box_size)))

# Load empty-domain simulation data into scattering monitor
sim.load_minus_flux_data(box_mx, box_mx_data_empty)
sim.load_minus_flux_data(box_px, box_px_data_empty)
sim.load_minus_flux_data(box_my, box_my_data_empty)
sim.load_minus_flux_data(box_py, box_py_data_empty)
sim.load_minus_flux_data(box_mz, box_mz_data_empty)
sim.load_minus_flux_data(box_pz, box_pz_data_empty)

# add movie monitors
animate = mp.Animate2D(fields=mp.Ez,
                       normalize = True,
                       # field_parameters={'alpha':0.8, 'cmap':'RdBu', 'interpolation':'none'},
                       # boundary_parameters={'hatch':'o', 'linewidth':1.5, 'facecolor':'y', 'edgecolor':'b', 'alpha':0.3},
                       output_plane = mp.Volume(center = mp.Vector3(), size = mp.Vector3(Size_x,Size_y)))


# visualize the simulation domain
sim.plot2D(output_plane = mp.Volume(center = mp.Vector3(), size = mp.Vector3(Size_x,Size_y)))
plt.savefig("simXY.png")
plt.close()

sim.plot2D(output_plane = mp.Volume(center = mp.Vector3(), size = mp.Vector3(Size_x,0,Size_z)))
plt.savefig("simXZ.png")
plt.close()
client.log_file("simXY.png", "simXY.png", "image/png", meta={"stage":"scatterer"})
client.log_file("simXZ.png", "simXZ.png", "image/png", meta={"stage":"scatterer"})
# --------------------------------
# Run simulation with scatterer + movie monitor
sim.run(mp.at_every(0.1,animate),until_after_sources = mp.stop_when_fields_decayed(10, mp.Ez, mp.Vector3(0,box_size/2,0), 1e-4))

# Create movie - requires ffmpeg
animate.to_mp4(fps = 10, filename = 'Nanosphere_Simulation.mp4')
plt.close()
client.log_file("Nanosphere_Simulation.mp4", "Nanosphere_Simulation.mp4", "video/mp4",
                meta={"stage":"scatterer"})
# Plot Fourier Transform fields
for f in range(len(dft_freqs)):
    Ez_f = sim.get_dft_array(dft_fields,mp.Ez,f)
    plt.imshow(np.real(Ez_f),extent = [-Size_y/2, Size_y/2, -Size_x/2, Size_x/2])
    plt.savefig(f"Ez_f_{1./dft_freqs[f]}.png")
    fn = f"Ez_f_{1./dft_freqs[f]}.png"
    plt.close()
    client.log_file(fn, fn, "image/png", meta={"stage":"scatterer","wavelength":float(1./dft_freqs[f])})
    # --------------------------------


# get scattering fluxes
box_mx_flux = np.array(mp.get_fluxes(box_mx))
box_px_flux = np.array(mp.get_fluxes(box_px))
box_my_flux = np.array(mp.get_fluxes(box_my))
box_py_flux = np.array(mp.get_fluxes(box_py))
box_pz_flux = np.array(mp.get_fluxes(box_pz))
box_mz_flux = np.array(mp.get_fluxes(box_mz))
# get absorption fluxes
box_mx_flux_abs = np.array(mp.get_fluxes(box_mx_abs))
box_px_flux_abs = np.array(mp.get_fluxes(box_px_abs))
box_my_flux_abs = np.array(mp.get_fluxes(box_my_abs))
box_py_flux_abs = np.array(mp.get_fluxes(box_py_abs))
box_pz_flux_abs = np.array(mp.get_fluxes(box_pz_abs))
box_mz_flux_abs = np.array(mp.get_fluxes(box_mz_abs))

# Calculate scattering cross-section data and compare with Mie theory
scatt_flux = -box_mx_flux + box_px_flux -box_my_flux + box_py_flux -box_mz_flux + box_pz_flux
abs_flux = -box_mx_flux_abs + box_px_flux_abs -box_my_flux_abs + box_py_flux_abs -box_mz_flux_abs + box_pz_flux_abs

scatt_cross_section = scatt_flux / incident_intensity
abs_cross_section = -abs_flux / incident_intensity

mie_wavelengths = np.linspace(min_wavelength,max_wavelength,100)
permittivities = np.array([Ag.epsilon(1./wl) for wl in mie_wavelengths])
sphere_refractive_index = np.sqrt(permittivities)[:,0,0]
print(sphere_refractive_index)

Q_sca, Q_abs, _ = Mie_Solver(Sphere_Radius = sphere_radius,
                                    Sphere_refractive_index = sphere_refractive_index,
                                    Wavelengths = mie_wavelengths)


plt.plot(wavelengths,scatt_cross_section / (np.pi * sphere_radius ** 2),'x',color = 'red')
plt.plot(mie_wavelengths,Q_sca,'blue')
plt.savefig("scattering_spectrum.png")
plt.close()

plt.plot(wavelengths,abs_cross_section / (np.pi * sphere_radius ** 2),'x',color = 'red')
plt.plot(mie_wavelengths,Q_abs,'blue')
plt.savefig("absorption_spectrum.png")
plt.close()
client.log_file("scattering_spectrum.png", "scattering_spectrum.png", "image/png")
client.log_file("absorption_spectrum.png", "absorption_spectrum.png", "image/png")
try:
    sca_peak = float(np.max(scatt_cross_section / (np.pi * sphere_radius ** 2)))
    abs_peak = float(np.max(abs_cross_section   / (np.pi * sphere_radius ** 2)))
    client.log(step=1, sca_peak=sca_peak, abs_peak=abs_peak)
except Exception:
    pass
# --------------------------------

# ---------- OptixLog: log full spectra as metrics + CSV ----------
# (1) per-wavelength metrics (step = index)
for i, wl in enumerate(wavelengths):
    client.log(step=i,
               wavelength=float(wl),
               sca_cs=float(scatt_cross_section[i]),
               abs_cs=float(abs_cross_section[i]))

# (2) CSV artifacts for download
def write_csv(fname, xs, ys, header=("wavelength","value")):
    with open(fname, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for x, y in zip(xs, ys):
            w.writerow([float(x), float(y)])

write_csv("scattering_spectrum.csv",
          wavelengths, scatt_cross_section / (np.pi * sphere_radius ** 2),
          header=("wavelength","Q_scat"))

write_csv("absorption_spectrum.csv",
          wavelengths, abs_cross_section / (np.pi * sphere_radius ** 2),
          header=("wavelength","Q_abs"))

client.log_file("scattering_spectrum.csv","scattering_spectrum.csv","text/csv")
client.log_file("absorption_spectrum.csv","absorption_spectrum.csv","text/csv")
# ---------------------------------------------------------------

print("✅ Finished + uploaded to OptixLog.")