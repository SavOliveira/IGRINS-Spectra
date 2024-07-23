import numpy as np
import matplotlib.pyplot as plt
# import mplcursors
import pandas as pd

from astropy.io import fits
from scipy.special import wofz
from scipy.integrate import trapz
from scipy.optimize import curve_fit

# Pandas Column IDs
# SNR = SNR per resolution element
igrins_cols = ['Wavelength', 'Flux', 'SNR','zero']
# IGRINS rpectral resolution element
spec_res = 1e-5 # micron per pixel

# cont_window_size = 20*spec_res

def gaussian(x,*p):
    amplitude, mean, std = p
    # Gaussian Distribution
    return ((amplitude/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mean)**2/std**2))) + 1 #+ (slope*x)+b


def gaussian_area(amp, std):
    return np.abs(amp*std)*np.sqrt(2*np.pi)

# /(std*np.sqrt(2*np.pi))
# def multi_gauss(x ,*params):
#     """Sum of multiple Gaussians."""
#     n = len(params) // 3  # Number of Gaussians
#     gaussians = [gaussian(x, params[i], params[i+1], params[i+2]) for i in range(0, len(params), 3)]
#     return np.sum(gaussians, axis=0)

def multi_gauss_fit(x, y, init_params, max_iter):
    """
    Fits multiple Gaussians to the data.

    Args:
    - x: Independent variable data.
    - y: Dependent variable data.
    - init_params: Initial parameters for the Gaussians. Format: [amp1, cen1, sigma1, amp2, cen2, sigma2, ...]
    - max_iter: Maximum number of iterations for curve fitting.

    Returns:
    - popt: Optimal parameters for the fit.
    - pcov: Covariance matrix of the fit.
    - best_model: Best-fit model.
    """
    n_params = len(init_params)
    
    if n_params % 3 != 0:
        raise ValueError("The number of initial parameters must be a multiple of 3.")

    def multi_gauss(x, num, *params):
        """Sum of multiple Gaussians."""
        gaussians = [gaussian(x, params[i], params[i+1], params[i+2]) for i in range(0, len(params), num)]
        return np.sum(gaussians)

    popt, pcov = curve_fit(f=multi_gauss,
                           xdata=x,
                           ydata=y,
                           p0=init_params,
                           maxfev=max_iter,
                           nan_policy='omit')
    # Generate the best-fit model using the optimal parameters
    best_model = multi_gauss(x, *popt)

    return popt, pcov, best_model


# /(std*np.sqrt(2*np.pi))
# could use same sigma for multi gauss fits
def two_gaussian(x, *p):
    amp1, c1, std1, amp2, c2, std2 = p
    return (gaussian(x, amp1,c1,std1) + gaussian(x, amp2,c2,std2) - 1)

def three_gaussian(x, *p):
    amp1, c1, std1, amp2, c2, std2, amp3, c3, std3 = p
    return (gaussian(x, amp1,c1,std1) + gaussian(x,amp2,c2,std2) + gaussian(x, amp3, c3, std3) - 2)

def four_gaussian(x, *p):
    amp1, c1, std1, amp2, c2, std2, amp3, c3, std3, amp4, c4, std4 = p
    return (gaussian(x, amp1, c1, std1) +
            gaussian(x, amp2, c2, std2) +
            gaussian(x, amp3, c3, std3) +
            gaussian(x, amp4, c4, std4) - 3)


def five_gaussian(x, *p):
    amp1, c1, std1, amp2, c2, std2, amp3, c3, std3, amp4, c4, std4, amp5, c5, std5 = p
    return (gaussian(x, amp1, c1, std1) +
            gaussian(x, amp2, c2, std2) +
            gaussian(x, amp3, c3, std3) +
            gaussian(x, amp4, c4, std4) +
            gaussian(x, amp5, c5, std5) - 4)

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

def model_fit(func,wavelen,norm_flux,flux_err,init_params,max_iter):
    '''
    Fit a model to some spectral region
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
                           maxfev = max_iter,
                           nan_policy = 'omit')
    
    # Give the optimal parameters as caluclated by curve fit to the Gaussian model
    best_model = func(wavelen,*popt)

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
    data = fits.getdata(filepath)
    wavelen = data[0]
    flux = data[1]
    snr = data[2]

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
    table = pd.read_csv(file_list, delimiter='\s+', comment='#', names=igrins_cols)
    # Convert SNR column to float
    table['SNR'] = table['SNR'].astype(float)
    
    table = table[(table['Wavelength'].gt(2.)) & table['SNR'].gt(50)]
    wavlen = table['Wavelength']

    return table,wavlen

def local_continuum_fit(wavelen_arr, flux_arr, poly_order, line_center, spec_res, regions):
    '''
    Local Continuum Fitting to spectral features

    Input:
    ---
    wavelen_arr: 1D numpy array
        Array of wavelengths
    flux_arr: 1D numpy array
        Array of raw flux
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
    region_indices: list of tuples
        List of indices defining the regions where continuum is determined
    '''

    contwave_array = []
    contflux_array = []
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

        # Append wavelength and flux regions to arrays
        contwave_array.extend(wavelen_arr[cont_reg_lo_idx:cont_reg_hi_idx])
        contflux_array.extend(flux_arr[cont_reg_lo_idx:cont_reg_hi_idx])

        region_indices.append((cont_reg_lo_idx, cont_reg_hi_idx))

    # Estimate continuum using 1D polyfit to points in selected range
    continuum_fit = np.polyfit(x=contwave_array, y=contflux_array, deg=poly_order)
    fitval = np.poly1d(continuum_fit)
    continuum = fitval(wavelen_arr[region_indices[0][0]:region_indices[-1][1]])

    # Return the full continuum fit and the indices of the regions used
    return continuum, region_indices


# contlo_min, contlo_max, conthi_min, conthi_max
# def gauss_fit(wavelen,norm_flux,flux_err,init_params,max_iter):
#     '''
#     Fit a single Gaussian to some spectrum
#     wavelen
#     norm_flux
#     flux_error
#     init_params
#     max_iter
#     '''

#     # wavelen_mask = (wavelen > wavelen_min) & (wavelen < wavelen_max)
#     # wavelen = wavelen[wavelen_mask]
#     # norm_flux = norm_flux[wavelen_mask]

#     popt, pcov = curve_fit(f = gaussian,
#                            xdata = wavelen,
#                            ydata = norm_flux,
#                            sigma = flux_err,
#                            p0 = init_params,
#                            maxfev = max_iter,
#                            nan_policy = 'omit')
    
#     # Give the optimal parameters as caluclated by curve fit to the Gaussian model
#     best_model = gaussian(wavelen,*popt)

#     return popt, pcov, best_model

# def two_gauss_fit(wavelen,norm_flux,flux_err,init_params,max_iter):
#     '''
#     wavelen
#     norm_flux
#     line_center
#     wavelen_min
#     wavelen_max
#     init_params
#     '''
#     # wavelen_mask = (wavelen > wavelen_min) & (wavelen < wavelen_max)
#     # wavelen = wavelen[wavelen_mask]
#     # norm_flux = norm_flux[wavelen_mask]

#     popt, pcov = curve_fit(f=two_gaussian,
#                            xdata=wavelen,
#                            ydata=norm_flux,
#                            sigma=flux_err,
#                            p0=init_params,
#                            maxfev=max_iter,
#                            nan_policy='omit')
    
#     # Give the optimal parameters as caluclated by curve fit to the Gaussian model
#     best_model = two_gaussian(wavelen,*popt)

#     return popt, pcov, best_model

# def three_gauss_fit(wavelen,norm_flux,flux_err,init_params,max_iter):
#     '''
#     wavelen
#     norm_flux
#     line_center
#     wavelen_min
#     wavelen_max
#     init_params
#     '''
#     # wavelen_mask = (wavelen > wavelen_min) & (wavelen < wavelen_max)
#     # wavelen = wavelen[wavelen_mask]
#     # norm_flux = norm_flux[wavelen_mask]

#     popt, pcov = curve_fit(f=three_gaussian,
#                            xdata=wavelen,
#                            ydata=norm_flux,
#                            sigma=flux_err,
#                            p0=init_params,
#                            maxfev=max_iter,
#                            nan_policy='omit')
    
#     # Give the optimal parameters as caluclated by curve fit to the Gaussian model
#     best_model = three_gaussian(wavelen,*popt)

#     return popt, pcov, best_model

# def four_gauss_fit(wavelen,norm_flux,flux_err,init_params,max_iter):
#     '''
#     wavelen
#     norm_flux
#     line_center
#     wavelen_min
#     wavelen_max
#     init_params
#     '''
#     # wavelen_mask = (wavelen > wavelen_min) & (wavelen < wavelen_max)
#     # wavelen = wavelen[wavelen_mask]
#     # norm_flux = norm_flux[wavelen_mask]

#     popt, pcov = curve_fit(f=four_gaussian,
#                            xdata=wavelen,
#                            ydata=norm_flux,
#                            sigma=flux_err,
#                            p0=init_params,
#                            maxfev=max_iter,
#                            nan_policy='omit')
#     # Give the optimal parameters as caluclated by curve fit to the Gaussian model
#     best_model = four_gaussian(wavelen,*popt)

#     return popt, pcov, best_model

# def five_gauss_fit(wavelen,norm_flux,flux_err,init_params,max_iter):
#     '''
#     wavelen
#     norm_flux
#     line_center
#     wavelen_min
#     wavelen_max
#     init_params
#     '''
#     # wavelen_mask = (wavelen > wavelen_min) & (wavelen < wavelen_max)
#     # wavelen = wavelen[wavelen_mask]
#     # norm_flux = norm_flux[wavelen_mask]

#     popt, pcov = curve_fit(f=five_gaussian,
#                            xdata=wavelen,
#                            ydata=norm_flux,
#                            sigma=flux_err,
#                            p0=init_params,
#                            maxfev=max_iter,
#                            nan_policy='omit')
#     # Give the optimal parameters as caluclated by curve fit to the Gaussian model
#     best_model = five_gaussian(wavelen,*popt)

#     return popt, pcov, best_model

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