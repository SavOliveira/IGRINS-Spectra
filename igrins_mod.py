import numpy as np
import matplotlib.pyplot as plt
# import mplcursors
import pandas as pd

from astropy.io import fits

from lmfit import Model

from scipy.special import wofz
from scipy.integrate import trapz
from scipy.optimize import curve_fit

# Pandas Column IDs
# SNR = SNR per resolution element
igrins_cols = ['Wavelength', 'Flux', 'SNR','zero']
# IGRINS rpectral resolution element
spec_res = 1e-5 # micron per pixel

# change the centers to something e.g. c1+offset instead of c1+c5
# use lab wavelengths -> velocity -> lambda = lam_lab (1+v/c)
# maybe force std to be the same
def gaussian(x, amp, c, std):
    # Gaussian Distribution
    return ((amp/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - c)**2/std**2))) + 1

def gaussian_area(amp, std):
    # area of a guassian using parameters (just amp??)
    return np.abs(amp*std)*np.sqrt(2*np.pi)

# /(std*np.sqrt(2*np.pi))
# could use same sigma for multi gauss fits
def two_gaussian(x, amp1, c1, std1, amp2, c2, std2):
    return (gaussian(x, amp1,c1,std1) + gaussian(x, amp2,c2,std2) - 1)

def three_gaussian(x, amp1, c1, std1, amp2, c2, std2, amp3, c3, std3):
    return (gaussian(x, amp1,c1,std1) + gaussian(x,amp2,c2,std2) + gaussian(x, amp3, c3, std3) - 2)

def three_gaussian_beta(x, amp1, c1, std1, amp2, c2, std2, amp3, c3, std3, beta):
    return (gaussian(x, amp1, c1*beta, std1) + 
            gaussian(x, amp2, c2*beta, std2) + 
            gaussian(x, amp3, c3*beta, std3) - 2)

def four_gaussian(x, amp1, c1, std1, amp2, c2, std2, amp3, c3, std3, amp4, c4, std4):
    # linear combination of four gaussians with an offset to set baseline at 1
    # beta = (1+v/c)
    return (gaussian(x, amp1, c1, std1) +
            gaussian(x, amp2, c2, std2) +
            gaussian(x, amp3, c3, std3) +
            gaussian(x, amp4, c4, std4) - 3)

def four_gaussian_beta(x, amp1, c1, std1, amp2, c2, std2, amp3, c3, std3, amp4, c4, std4, beta):
    # linear combination of four gaussians with an offset to set baseline at 1
    # using the same sigma for each component Gaussian
    # beta = (1+v/c)
    return (gaussian(x, amp1, c1*(beta), std1) +
            gaussian(x, amp2, c2*(beta), std2) +
            gaussian(x, amp3, c3*(beta), std3) +
            gaussian(x, amp4, c4*(beta), std4) - 3)


def five_gaussian(x, amp1, c1, std1, amp2, c2, std2, amp3, c3, std3, amp4, c4, std4, amp5, c5, std5, beta):
    return (gaussian(x, amp1, c1*(beta), std1) +
            gaussian(x, amp2, c2*(beta), std2) +
            gaussian(x, amp3, c3*(beta), std3) +
            gaussian(x, amp4, c4*(beta), std4) +
            gaussian(x, amp5, c5*(beta), std5) - 4)

def five_gaussian_beta(x, amp1, c1, std1, amp2, c2, std2, amp3, c3, std3, amp4, c4, std4, amp5, c5, std5, beta):
    return (gaussian(x, amp1, c1*(beta), std1) +
            gaussian(x, amp3, c3*(beta), std2) +
            gaussian(x, amp4, c4*(beta), std3) +
            gaussian(x, amp2, c2*(beta), std4) +
            gaussian(x, amp5, c5*(beta), std5) - 4)

def voigt(x, amp, center, sigma, gamma):
    """
    Voigt profile function for curve fitting

    Parameters:
    x : array_like
        The x values at which to evaluate the Voigt profile
    amp : float
        Amplitude of the Voigt profile
    center : float
        Center position of the Voigt profile
    sigma : float
        The standard deviation of the Gaussian component
    gamma : float
        The half-width at half-maximum of the Lorentzian component

    Returns:
    y : ndarray
        The Voigt profile values at x
    """
    z = ((x - center) + 1j*gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

def model_fit(func,wavelen,norm_flux,flux_err,init_params,**kwargs):
    '''
    Fit a model to some spectral region using scipy curve_fit
    wavelen
    norm_flux
    flux_error
    init_params
    max_iter
    '''

    # wavelen_mask = (wavelen > wavelen_min) & (wavelen < wavelen_max)
    # wavelen = wavelen[wavelen_mask]
    # norm_flux = norm_flux[wavelen_mask]

    popt, pcov = curve_fit(f = func,
                           xdata = wavelen,
                           ydata = norm_flux,
                           sigma = flux_err,
                           p0 = init_params,
                           nan_policy = 'omit',
                           **kwargs)
    # Calculate errors on each parameter
    param_error = np.sqrt(np.diag(pcov))

    # Give the optimal parameters as caluclated by curve fit to the Gaussian model
    best_model = func(wavelen,*popt)

    return popt, pcov, param_error, best_model

# Define function to calculate errors
def calculate_model_errors(xdata, popt, pcov, model_func):
    """Calculate errors for model predictions."""
    model = model_func(xdata, *popt)
    n_params = len(popt)
    jacobian = np.zeros((len(xdata), n_params))

    for i in range(n_params):
        perturbed_params = np.copy(popt)
        perturbed_params[i] += np.sqrt(np.diag(pcov))[i]
        jacobian[:, i] = model_func(xdata, *perturbed_params) - model

    errors = np.sqrt(np.sum((jacobian @ np.sqrt(np.diag(pcov)))**2, axis=1))
    return errors

from lmfit import Model

def lm_model_fit(func, wavelen, norm_flux, flux_err, init_params, max_iter):
    '''
    Fit a model to some spectral region
    wavelen
    norm_flux
    flux_error
    init_params
    max_iter
    '''
    
    # Create a model based on the provided function
    model = Model(func)
    
    # Create parameters from init_params
    params = model.make_params(**init_params)
    
    # Perform the fit
    result = model.fit(norm_flux, params, x=wavelen, weights=1/flux_err, max_nfev=max_iter)
    
    # Extract the optimized parameters and their covariance
    popt = [result.params[key].value for key in result.params.keys()]
    pcov = result.covar
    
    # Evaluate the best fit model
    best_model = result.best_fit
    
    return popt, pcov, best_model


def get_fitsdata(filepath):
    '''
    Get data from fits file + do some cleaning
    Input: 
    filepath = string

    Output:
    wavelen = np array
    flux = np array
    snr = np array
    '''
    data = fits.getdata(filepath) # type: ignore
    wavelen = data[0] # type: ignore
    flux = data[1] # type: ignore
    snr = data[2] # type: ignore

    # Clean data a bit
    snr_min = 50 # Minimum SNR
    snr_max = 1e4 # Maxmimum SNR
    snr_cut = (snr > snr_min) & (snr < snr_max) # bitwise SNR masking

    flux_min = 0 # minimum flux
    flux_cut = flux > flux_min # bitwise flux masking

    wavelen = wavelen[snr_cut & flux_cut]
    flux = flux[snr_cut & flux_cut]
    snr = snr[snr_cut & flux_cut]

    return wavelen, flux, snr

def txt_to_table(file_list):
    '''
    Input:
    ---
    filelist = path to folder of IGRINS spectra txt files

    Output:
    ---
    table = pandas DataFrame
    '''
    # Create Pandas dataframe
    table = pd.read_csv(file_list, delimiter='\s+', comment='#', names=igrins_cols) # type: ignore
    # Convert SNR column to float
    table['SNR'] = table['SNR'].astype(float)
    
    table = table[(table['Wavelength'].gt(2.)) & table['SNR'].gt(50)]
    wavlen = table['Wavelength']

    return table,wavlen

def local_continuum_fit(wavelen_arr, flux_arr, flux_err_arr, poly_order, line_center, spec_res, regions):
    '''
    Local Continuum Fitting to spectral features with error propagation

    Input:
    ---
    wavelen_arr: 1D numpy array
        Array of wavelengths
    flux_arr: 1D numpy array
        Array of raw flux
    flux_err_arr: 1D numpy array
        Array of flux errors
    poly_order: int
        Order of the polynomial for fitting
    line_center: float
        Center wavelength of the spectral feature
    spec_res: float
        Spectral resolution of the instrument
    regions: list of tuples
        List of (center, width) tuples to define continuum regions. Each width is in units of spectral resolution.

    Output:
    ---
    continuum: 1D numpy array
        Continuum fit to the spectral feature
    continuum_err: 1D numpy array
        Error in the continuum fit
    region_indices: list of tuples
        List of indices defining the regions where continuum is determined
    '''

    contwave_array = []
    contflux_array = []
    contflux_err_array = []
    region_indices = []

    for leftreg, width in regions:
        region_start = line_center + (leftreg * spec_res)
        region_width = width * spec_res

        # Define continuum region boundaries
        cont_reg_lo = region_start
        cont_reg_hi = region_start + region_width

        # Find the indices for the continuum region
        cont_reg_lo_idx = np.nanargmin(np.abs(wavelen_arr - cont_reg_lo))
        cont_reg_hi_idx = np.nanargmin(np.abs(wavelen_arr - cont_reg_hi))

        # Append wavelength, flux, and flux error regions to arrays
        contwave_array.extend(wavelen_arr[cont_reg_lo_idx:cont_reg_hi_idx])
        contflux_array.extend(flux_arr[cont_reg_lo_idx:cont_reg_hi_idx])
        contflux_err_array.extend(flux_err_arr[cont_reg_lo_idx:cont_reg_hi_idx])

        region_indices.append((cont_reg_lo_idx, cont_reg_hi_idx))

    # Convert lists to numpy arrays for polyfit
    contwave_array = np.array(contwave_array)
    contflux_array = np.array(contflux_array)
    contflux_err_array = np.array(contflux_err_array)

    # Estimate continuum using weighted 1D polyfit to points in selected range
    weights = 1 / contflux_err_array
    continuum_fit = np.polyfit(x=contwave_array, y=contflux_array, deg=poly_order, w=weights)
    fitval = np.poly1d(continuum_fit)
    
    # Compute the continuum and its error
    continuum = fitval(wavelen_arr[region_indices[0][0]:region_indices[-1][1]])

    # Error propagation: The variance in the continuum fit
    continuum_err = np.sqrt(np.sum((fitval.deriv()(contwave_array) * contflux_err_array) ** 2))

    # Return the full continuum fit, its error, and the indices of the regions used
    return continuum, continuum_err, region_indices

# def local_continuum_fit(wavelen_arr, flux_arr, poly_order, line_center, spec_res, regions):
#     '''
#     Local Continuum Fitting to spectral features

#     Input:
#     ---
#     wavelen_arr: 1D numpy array
#         Array of wavelengths
#     flux_arr: 1D numpy array
#         Array of raw flux
#     poly_order: int
#         Order of the polynomial for fitting
#     line_center: float
#         Center wavelength of the spectral feature
#     spec_res: float
#         Spectral resolution of the instrument
#     regions: list of tuples
#         List of (center, width) tuples to define continuum regions. Each width is in units of spectral resolution.

#     Output:
#     ---
#     continuum: 1D numpy array
#         Continuum fit to the spectral feature
#     region_indices: list of tuples
#         List of indices defining the regions where continuum is determined
#     '''

#     contwave_array = []
#     contflux_array = []
#     region_indices = []

#     for leftreg, width in regions:
#         region_start = line_center + (leftreg * spec_res)
#         region_width = width * spec_res

#         # Define continuum region boundaries
#         cont_reg_lo = region_start
#         cont_reg_hi = region_start + region_width

#         # Find the indices for the continuum region
#         cont_reg_lo_idx = np.nanargmin(np.abs(wavelen_arr - cont_reg_lo))
#         cont_reg_hi_idx = np.nanargmin(np.abs(wavelen_arr - cont_reg_hi))

#         # Append wavelength and flux regions to arrays
#         contwave_array.extend(wavelen_arr[cont_reg_lo_idx:cont_reg_hi_idx])
#         contflux_array.extend(flux_arr[cont_reg_lo_idx:cont_reg_hi_idx])

#         region_indices.append((cont_reg_lo_idx, cont_reg_hi_idx))

#     # Estimate continuum using 1D polyfit to points in selected range
#     continuum_fit = np.polyfit(x=contwave_array, y=contflux_array, deg=poly_order)
#     fitval = np.poly1d(continuum_fit)
#     continuum = fitval(wavelen_arr[region_indices[0][0]:region_indices[-1][1]])

#     # Return the full continuum fit and the indices of the regions used
#     return continuum, region_indices

def calc_eq_width(norm_flux, dlam):
    '''
    norm_flux = line_flux / continuum_flux
    dlam = wavelength range
    '''
    ew = np.sum((1-norm_flux)*dlam)
    return ew

def normalize_flux(flux):
    return flux/np.median(flux)

def mean_continuum(continuum):
    return np.mean(continuum)

def continuum_subtract(flux,continuum):
    cont_sub = flux-continuum
    return cont_sub

def continuum_normalize(flux,continuum):
    cont_norm = flux/continuum
    return cont_norm

def mean_continuum_subtract(flux,mean_continuum):
    meancont_sub = flux-mean_continuum
    return meancont_sub

def mean_continuum_norm(flux,mean_continuum):
    meancont_norm = flux/mean_continuum
    return meancont_norm