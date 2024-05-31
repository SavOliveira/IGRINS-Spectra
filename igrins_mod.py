import numpy as np
import matplotlib.pyplot as plt
# import mplcursors
import pandas as pd

from astropy.io import fits
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

    def multi_gauss(x, *params):
        """Sum of multiple Gaussians."""
        gaussians = [gaussian(x, params[i], params[i+1], params[i+2]) for i in range(0, len(params), 3)]
        return np.sum(gaussians)

    popt, pcov = curve_fit(f=multi_gauss,
                           xdata=x,
                           ydata=y,
                           p0=init_params,
                           maxfev=max_iter)
    # Generate the best-fit model using the optimal parameters
    best_model = multi_gauss(x, *popt)

    return popt, pcov, best_model


# /(std*np.sqrt(2*np.pi))
# could use same sigma for multi gauss fits
def two_gaussian(x, *p):
    amp1, c1, std, amp2, c2, = p
    return ((amp1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - c1)**2/std**2))) + ((amp2/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - c2)**2/std**2))) + 1

def three_gaussian(x, *p):
    amp1, c1, std, amp2, c2, amp3, c3 = p
    return (amp1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - c1)**2/std**2)) + (amp2/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - c2)**2/std**2)) + (amp3/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - c3)**2/std**2)) + 1

def four_gaussian(x, *p):
    amp1, c1, std, amp2, c2, amp3, c3, amp4, c4 = p
    return (amp1/(std*np.sqrt(2*np.pi)) * np.exp(-0.5*((x - c1)**2/std**2)) +
        (amp2/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - c2)**2/std**2)) +
        (amp3/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - c3)**2/std**2)) +
       (amp4/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - c4)**2/std**2)) + 1)


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


def local_continuum_fit(wavelen_arr, flux_arr, poly_order, line_center, spec_res, window_size,left_num,right_num):
    '''
    Local Continuum Fitting to spectral features

    Input:
    ---
    wavelen_arr: 1D numpy array
        Array of wavelengths
    flux_arr: 1D numpy array
        Array of raw flux
    line_center: float
        Center wavelength of the spectral feature
    spec_res: float
        Spectral resolution of the instrument
    window_size:
        size of window to estimate continuum
    left_num:
        determines how many pixels away left of line center
    right_num:
        determines how many pixels away right of line center

    Output:
    ---
    continuum: 1D numpy array
        Continuum fit to the spectral feature
    contlo_min, contlo_max, conthi_min, conthi_max: int
        Indices defining the regions where continuum is determined
    '''

    cont_window_size = window_size * spec_res

    # Define spectral window
    wave_reg1_left = line_center - (left_num * spec_res)
    wave_reg2_right = line_center + (right_num * spec_res)

    # wave_regions = []

    # for i in range(len(nums)):
    #     region = ((line_center-(nums[i]*spec_res)),(line_center-(nums[i]*spec_res)))
    #     wave_regions = region

    # Spectral feature max and min wavelength indices
    wavemin_idx = np.nanargmin(np.abs(wavelen_arr - wave_reg1_left))
    wavemax_idx = np.nanargmin(np.abs(wavelen_arr - wave_reg2_right))

    # Choose spectral regions on either side of spectral feature to define a continuum
    cont_reg1_lo = wavelen_arr[wavemin_idx] - cont_window_size # left
    cont_reg1_hi = wavelen_arr[wavemin_idx] # right

    cont_reg2_lo = wavelen_arr[wavemax_idx] # left
    cont_reg2_hi = wavelen_arr[wavemax_idx] + cont_window_size # right

    # Find the indices for the continuum regions on either side of the spectral feature
    cont_reg1_lo_idx = np.nanargmin(np.abs(wavelen_arr[:]-cont_reg1_lo))
    cont_reg1_hi_idx = np.nanargmin(np.abs(wavelen_arr[:]-cont_reg1_hi))

    cont_reg2_lo_idx = np.nanargmin(np.abs(wavelen_arr[:]-cont_reg2_lo))
    cont_reg2_hi_idx = np.nanargmin(np.abs(wavelen_arr[:]-cont_reg2_hi))

    # Estimate continuum using mean of points in selected range

    # Wavelength range of where I'm estimating continuum
    cont_reg1_wave = wavelen_arr[cont_reg1_lo_idx:cont_reg1_hi_idx] # 1st continuum region
    cont_reg2_wave = wavelen_arr[cont_reg2_lo_idx:cont_reg2_hi_idx] # 2nd continuum region

    # Flux range of where I'm estimating continuum
    cont_reg1_flux = flux_arr[cont_reg1_lo_idx:cont_reg1_hi_idx]
    cont_reg2_flux = flux_arr[cont_reg2_lo_idx:cont_reg2_hi_idx]

    contwave_array = np.concatenate((cont_reg1_wave, cont_reg2_wave))
    contflux_array = np.concatenate((cont_reg1_flux, cont_reg2_flux))

    # Estimate continuum using 1D polyfit to points in selected range
    continuum_fit = np.polyfit(contwave_array, contflux_array, poly_order)
    fitval = np.poly1d(continuum_fit)
    continuum = fitval(wavelen_arr)
    continuum = continuum[cont_reg1_lo_idx:cont_reg2_hi_idx]

    return continuum, cont_reg1_lo_idx, cont_reg1_hi_idx, cont_reg2_lo_idx, cont_reg2_hi_idx

def gauss_fit(wavelen,norm_flux,init_params,max_iter):
    '''
    Fit a single Gaussian to some spectrum
    wavelen
    norm_flux
    line_center
    wavelen_min
    wavelen_max
    init_params
    '''

    # wavelen_mask = (wavelen > wavelen_min) & (wavelen < wavelen_max)
    # wavelen = wavelen[wavelen_mask]
    # norm_flux = norm_flux[wavelen_mask]

    popt, pcov = curve_fit(f=gaussian,
                           xdata=wavelen,
                           ydata=norm_flux,
                           p0=init_params,
                           maxfev=max_iter)
    
    # Give the optimal parameters as caluclated by curve fit to the Gaussian model
    best_model = gaussian(wavelen,*popt)

    return popt, pcov, best_model

def two_gauss_fit(wavelen,norm_flux,init_params,max_iter):
    '''
    wavelen
    norm_flux
    line_center
    wavelen_min
    wavelen_max
    init_params
    '''
    # wavelen_mask = (wavelen > wavelen_min) & (wavelen < wavelen_max)
    # wavelen = wavelen[wavelen_mask]
    # norm_flux = norm_flux[wavelen_mask]

    popt, pcov = curve_fit(f=two_gaussian,
                           xdata=wavelen,
                           ydata=norm_flux,
                           p0=init_params,
                           maxfev=max_iter)
    
    # Give the optimal parameters as caluclated by curve fit to the Gaussian model
    best_model = two_gaussian(wavelen,*popt)

    return popt, pcov, best_model

def three_gauss_fit(wavelen,norm_flux,init_params,max_iter):
    '''
    wavelen
    norm_flux
    line_center
    wavelen_min
    wavelen_max
    init_params
    '''
    # wavelen_mask = (wavelen > wavelen_min) & (wavelen < wavelen_max)
    # wavelen = wavelen[wavelen_mask]
    # norm_flux = norm_flux[wavelen_mask]

    popt, pcov = curve_fit(f=three_gaussian,
                           xdata=wavelen,
                           ydata=norm_flux,
                           p0=init_params,
                           maxfev=max_iter)
    # Give the optimal parameters as caluclated by curve fit to the Gaussian model
    best_model = three_gaussian(wavelen,*popt)

    return popt, pcov, best_model

def four_gauss_fit(wavelen,norm_flux,init_params,max_iter):
    '''
    wavelen
    norm_flux
    line_center
    wavelen_min
    wavelen_max
    init_params
    '''
    # wavelen_mask = (wavelen > wavelen_min) & (wavelen < wavelen_max)
    # wavelen = wavelen[wavelen_mask]
    # norm_flux = norm_flux[wavelen_mask]

    popt, pcov = curve_fit(f=four_gaussian,
                           xdata=wavelen,
                           ydata=norm_flux,
                           p0=init_params,
                           maxfev=max_iter)
    # Give the optimal parameters as caluclated by curve fit to the Gaussian model
    best_model = four_gaussian(wavelen,*popt)

    return popt, pcov, best_model


# contlo_min, contlo_max, conthi_min, conthi_max

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