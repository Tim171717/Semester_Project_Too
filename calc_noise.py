# Imports
import numpy as np
from astropy import constants as const
from matplotlib import pyplot as plt
from phringe.core.scene import Scene
from phringe.core.sources.exozodi import Exozodi
from phringe.core.sources.local_zodi import LocalZodi
from phringe.core.sources.planet import Planet
from phringe.core.sources.star import Star
from phringe.io.input_spectrum import InputSpectrum
from phringe.lib.beam_combiner import DoubleBracewell
from phringe.util.baseline import OptimalNullingBaseline
from spectres import spectres

from lifesimmc.core.modules.generating.data_generation_module import DataGenerationModule
from lifesimmc.core.modules.generating.template_generation_module import TemplateGenerationModule
from lifesimmc.core.modules.loading.setup_module import SetupModule
from lifesimmc.core.modules.processing.calibration_star_zca_whitening_module import CalibrationStarZCAWhiteningModule
from lifesimmc.core.modules.processing.ml_parameter_estimation_module import MLParameterEstimationModule
from lifesimmc.core.pipeline import Pipeline
from lifesimmc.lib.instrument import LIFEReferenceDesign, InstrumentalNoise
from lifesimmc.lib.observation import LIFEReferenceObservation

# Settings
# path to input spectrum .txt
input_spec_path = '/mnt/c/Users/timlf/PycharmProjects/Semester_Project_Too/VH2O_sn50.txt'
output_spec_path = '/mnt/c/Users/timlf/PycharmProjects/Semester_Project_Too/VH2O_sn10_noise.txt'

reference_SNR = 10  # target SNR at reference wavelength
reference_wl = 11.25  # wavelength (in um) where SNR should be set to target

spec_resolution = 100  # spectral resolving power R of input spectrum
min_wavelength = 4.0  # short wavelength cutoff for output spectrum (in um)
max_wavelength = 18.5  # long wavelength cutoff for output spectrum (in um)


# Helper functions
def Scale_SNR(wl, SNR_lam, SNR_ref, lam_ref):
    # takes the value of the SNR at the wl closest to the reference
    SNR_calc = SNR_lam[np.argmin(np.abs(wl - lam_ref))]
    # scale SNR at all wl to match reference definition
    return SNR_lam * SNR_ref / SNR_calc


def fnu_to_flambda(spectrum):
    """
    Convert a spectrum from erg/s/Hz/m^2 to W/m^2/µm/sr.

    Parameters:
        spectrum (numpy.ndarray): 2D array where:
            - column 0 = wavelength in microns (µm)
            - column 1 = flux density in erg/s/Hz/m^2

    Returns:
        numpy.ndarray: 2D array where:
            - column 0 = wavelength in microns (µm)
            - column 1 = flux density in W/m^2/µm/sr
    """
    lam_m = spectrum[:, 0] * 1e-6  # µm -> m
    fnu_watt = spectrum[:, 1] * 1e-7  # erg/s -> W

    flambda = fnu_watt * 1e-6 * const.c.value / (lam_m ** 2)  # W/m^2/µm

    flambda /= np.pi * (const.R_earth.value / (10 * const.pc.value)) ** 2  # W/m^2/µm/sr

    return np.column_stack((spectrum[:, 0], flambda))


def flambda_to_fnu_with_error(wavelength_um, flux_photon, flux_err_photon):
    """
    Convert photon flux density (ph/s/m^2/um) to energy flux density (erg/s/m^2/Hz).

    Parameters:
        wavelength_um (numpy.ndarray): Wavelength array in microns (µm).
        flux_photon (numpy.ndarray): Flux density array in ph/s/m^2/µm.
        flux_err_photon (numpy.ndarray): Flux density error array in same units.

    Returns:
        numpy.ndarray: 2D array with columns:
            - wavelength in µm
            - flux density in erg/s/m^2/Hz
            - flux error in erg/s/m^2/Hz
    """
    # Conversion factor
    factor = const.h.value * 1e7 * wavelength_um

    fnu = np.array(flux_photon) * factor
    fnu_err = np.array(flux_err_photon) * factor

    return np.column_stack((wavelength_um, fnu, fnu_err))


# Convert input spectrum
input_spec = np.genfromtxt(input_spec_path)
input_spec_flam = fnu_to_flambda(input_spec)
np.savetxt('temp_spec.txt', input_spec_flam)

# Use the predefined ideal LIFE baseline instrument, i.e. without any instrumental noise
inst = LIFEReferenceDesign(instrumental_noise=InstrumentalNoise.NONE)

# For this example, manually update the spectral resolving power and aperture diameter
inst.spectral_resolving_power = spec_resolution
inst.wavelength_max = input_spec[-1, 0] * 1e-6
inst.wavelength_min = input_spec[0, 0] * 1e-6
inst.aperture_diameter = 3.5

# User the predefined observation for the LIFE baseline design
obs = LIFEReferenceObservation(
    total_integration_time='10 d',
    detector_integration_time='0.05 d',
    nulling_baseline=OptimalNullingBaseline(
        angular_star_separation='habitable-zone',
        wavelength='15 um',
        sep_at_max_mod_eff=DoubleBracewell.sep_at_max_mod_eff[0]
    ),
)

scene = Scene()

sun_twin = Star(
    name='Sun Twin',
    distance='10 pc',
    mass='1 Msun',
    radius='1 Rsun',
    temperature='5700 K',
    right_ascension='10 hourangle',
    declination='45 deg',
)

local_zodi = LocalZodi()

exozodi = Exozodi(level=3)

earth_twin = Planet(
    name='Earth Twin',
    has_orbital_motion=False,
    mass='1 Mearth',
    radius='1 Rearth',
    temperature='254 K',
    semi_major_axis='1 au',
    eccentricity='0',
    inclination='0 deg',
    raan='90 deg',
    argument_of_periapsis='0 deg',
    true_anomaly='45 deg',
    input_spectrum=InputSpectrum(path_to_file='temp_spec.txt', sed_units='W/m2/um/sr', wavelength_units='um'),
)

scene.add_source(sun_twin)
scene.add_source(local_zodi)
scene.add_source(exozodi)
scene.add_source(earth_twin)

# Create the pipeline
pipeline = Pipeline(gpu_index=2, seed=42, grid_size=40)

# Setup the simulation
module = SetupModule(
    n_setup_out='setup',
    n_planet_params_out='params_init',
    instrument=inst,
    observation=obs,
    scene=scene
)
pipeline.add_module(module)

module = DataGenerationModule(n_setup_in='setup', n_data_out='data')
pipeline.add_module(module)

module = TemplateGenerationModule(n_setup_in='setup', n_template_out='temp', fov=1e-6)
pipeline.add_module(module)

module = CalibrationStarZCAWhiteningModule(
    n_setup_in='setup',
    n_data_in='data',
    n_template_in='temp',  # Optional
    # n_planet_params_in='params_init',
    n_data_out='data_norm',
    n_template_out='temp_norm',  # Optional
    n_transformation_out='norm',
    diagonal_only=True
)
pipeline.add_module(module)

# Run pipeline with all modules we have added so far
pipeline.run()

module = MLParameterEstimationModule(
    n_setup_in='setup',
    n_data_in='data_norm',
    n_template_in='temp_norm',
    n_transformation_in='norm',
    n_planet_params_in='params_init',
    n_planet_params_out='params_ml',
    bounds=True
)
pipeline.add_module(module)
pipeline.run()

# Get the initial (input) parameters so we can plot the input spectrum (spectral energy distribution; SED) as a reference
params_init = pipeline.get_resource('params_init')
sed_init = params_init.params[0].sed.cpu().numpy()[:-1]  # Convert to numpy array from a torch Tensor
sed_init /= 1e6  # Convert to ph s-1 m-2 um-1
wavelengths = params_init.params[0].sed_wavelength_bin_centers.cpu().numpy()  # Convert to Torch tensor
wavelengths *= 1e6  # Convert from m to um

# Get the estimated parameters
params_ml = pipeline.get_resource('params_ml')
sed_estimated = params_ml.params[0].sed.cpu().numpy()
sed_estimated /= 1e6  # Convert to ph s-1 m-2 um-1
sed_err_low = params_ml.params[0].sed_err_low / 1e6
sed_err_high = params_ml.params[0].sed_err_high / 1e6
estimated_wavelengths = params_ml.params[0].sed_wavelength_bin_centers.cpu().numpy() * 1e6

# Rebin spectra
input_spec_rebin = input_spec[(input_spec[:, 0] > min_wavelength) & (input_spec[:, 0] < max_wavelength)]
sed_init_rebin = spectres(input_spec_rebin[:, 0], wavelengths, sed_init)
sed_estimated_rebin = np.array(spectres(input_spec_rebin[:, 0], estimated_wavelengths,
                                        sed_estimated, spec_errs=np.array(sed_err_low))).T

# calculate SNR and rescale
snr = sed_init_rebin / sed_estimated_rebin[:, 1]
snr_scaled = Scale_SNR(input_spec_rebin[:, 0], snr, reference_SNR, reference_wl)

error = (input_spec_rebin[:, 1] / snr_scaled).reshape((len(snr), 1))
input_spec_rebin = np.hstack((input_spec_rebin, error))

np.savetxt(output_spec_path, input_spec_rebin)

# Plot resulting spectrum with errorbars
plt.figure()
plt.errorbar(
    input_spec_rebin[:, 0],
    input_spec_rebin[:, 1],
    yerr=input_spec_rebin[:, 2],
    fmt='o',
    ecolor='gray',
    alpha=0.8,
    zorder=1,
    capsize=1.5,
    capthick=0.5,
    linewidth=0.5
)
plt.xlabel(r'$\lambda$ [$\mu$m]')
plt.ylabel(r'Flux [erg s$^{-1}$ Hz$^{-1}$ m$^{-2}$]')
plt.xlim((min_wavelength, max_wavelength))
plt.ylim(bottom=0.)

plt.fill_between(
    wavelengths[:-1],
    np.array(sed_init) - np.array(sed_err_low)[:-1],
    np.array(sed_init) + np.array(sed_err_high)[:-1],
    color='dodgerblue',
    edgecolor=None,
    lw=0,
    alpha=0.2,
    label='1$\sigma$',
    zorder=0
)
plt.scatter(wavelengths, sed_estimated, label='Data', color="xkcd:sapphire", zorder=2, marker='.')
plt.plot(wavelengths[:-1], sed_init, label='True', linestyle='dashed', color='black', alpha=0.6, zorder=1)
plt.title('True vs. Estimated Spectrum')
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('SED (ph s$^{-1}$ m$^{-2}$ $\mu$m$^{-1}$)')
plt.ylim(-0.2, 0.6)
plt.legend()
plt.show()