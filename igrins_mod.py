import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from scipy.integrate import trapz
from scipy.optimize import curve_fit

# Pandas Column IDs
# SNR = SNR per resolution element
igrins_cols = ['Wavelength', 'Flux', 'SNR','zero']
# IGRINS rpectral resolution element
spec_res = 1e-5 # micron per pixel

cont_window_size = 20*spec_res

def Gaussian(x,amplitude, mean, std, b):
    # Normalized Gaussian Distribution
    return ((amplitude)/(std*np.sqrt(2*np.pi)) * np.exp(-0.5*((x - mean)/std)**2)) + b

def mult_Gaussian(x, amp1, c1, std1, amp2, c2, std2, b):
    return ((amp1)/(std1*np.sqrt(2*np.pi)) * np.exp(-0.5*((x - c1)/std1)**2)) + ((amp2)/(std2*np.sqrt(2*np.pi)) * np.exp(-0.5*((x - c2)/std2)**2)) + b

def get_fitsdata(filepath):
    '''
    Get data from fits file + do some cleaning
    Input: 
    filepath = string, might change to file list to do all the same time

    Output:
    wavelen = np array
    flux = np array
    snr = np array
    '''
    first_data = fits.getdata(filepath)

    n = len(filepath)

    wavelen = first_data[0]
    flux = first_data[1]
    snr = first_data[2]

    ##  Work in Progress?
    # for i in range(0,n):
    #    spec = fits.getdata(filepath[i])
    #    spec_stack[:,i] = spec[:]
    
    # Clean data a bit
    snr_min = 50 # Minimum SNR
    snr_max = 1e4 # Maxmimum SNR
    snr_cut = (snr > snr_min) & (snr < snr_max) # bitwise SNR masking

    flux_min = 0 # minimum flux
    flux_cut = flux > flux_min # bitwise flux masking

    wavelen_min = 1.92
    wavelen_max = 2.48
    wavelen_cut = (wavelen > wavelen_min) & (wavelen < wavelen_max)

    wavelen = wavelen[snr_cut & flux_cut & wavelen_cut]
    flux = flux[snr_cut & flux_cut & wavelen_cut]
    snr = snr[snr_cut & flux_cut & wavelen_cut]

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



def local_continuum_fit(wavelen_arr,flux_arr,line_center):
    '''
    Local Continuum Fitting to spectral features

    Input:
    ---
    wavelen_arr
    flux_arr
    line_center
    

    Output:
    ---

    '''
    cont_window_size = 15*spec_res

    # Find the index for central wavelength of spectral feature
    wave_left = line_center - (75*spec_res)
    wave_right = line_center + (75*spec_res)

    # Spectral feature max and min wavelength indices
    wavemin_idx = np.abs(wavelen_arr - wave_left).argmin()
    wavemax_idx = np.abs(wavelen_arr - wave_right).argmin()

    # Choose spectral regions on either side of spectral feature to define a continuum
    contlo_1 = wave_left-cont_window_size
    contlo_2 = wave_left

    conthi_1 = wave_right
    conthi_2 = wave_right+cont_window_size

    # Find the indices for the continuum regions on either side of the spectral feature
    contlo_min = np.abs(wavelen_arr - contlo_1).argmin()
    contlo_max = np.abs(wavelen_arr - contlo_2).argmin()

    conthi_min = np.abs(wavelen_arr - conthi_1).argmin()
    conthi_max = np.abs(wavelen_arr - conthi_2).argmin()

    specline_window = wavelen_arr[contlo_max:conthi_max]
    #################################################
    # estimate continuum using mean of points in selected range

    # wavelength range of where I'm estimating continuum
    contlo_wave = wavelen_arr[contlo_min:contlo_max]
    conthi_wave = wavelen_arr[conthi_min:conthi_max]

    # fluxe range of where I'm estimating continuum
    contlo_flux = flux_arr[contlo_min:contlo_max]
    conthi_flux = flux_arr[conthi_min:conthi_max]

    contwave_array = np.concatenate((contlo_wave, conthi_wave))
    contflux_array = np.concatenate((contlo_flux, conthi_flux))

    mean_cont = np.sum(contflux_array)/len(contflux_array)

    # estimate continuum using 1d polyfit to points in selected range
    cont_fit = np.polyfit(contwave_array, contflux_array, 1)
    fitval = np.poly1d(cont_fit)
    continuum = fitval(wavelen_arr)

    return continuum, contlo_min, contlo_max, conthi_min, conthi_max

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

def gauss_fit(wavelen,norm_flux,line_center,contlo_min,conthi_max):
    # initial parameters for the Gaussian
    init_param = 1-(norm_flux[contlo_min:conthi_max]).max(), line_center, 1., 0 # Amplitude, Center, STD, y-offset

    # param_bounds = ([-1,line_center-(5*spec_res),0.,-1.],[1,line_center+(5*spec_res),1.,1.])

    popt, pcov = curve_fit(f=Gaussian,
                           xdata=wavelen[contlo_min:conthi_max],
                           ydata=norm_flux[contlo_min:conthi_max],
                           maxfev=50000)
    # Give the optimal parameters as caluclated by curve fit to the Gaussian model
    best_model = Gaussian(wavelen,*popt)

    return popt, pcov, best_model

def multigauss_fit(wavelen,norm_flux,line_center,contlo_min,conthi_max):
    # initial parameters for the Gaussian

    init_param = 1-(norm_flux[contlo_min:conthi_max]).max(), line_center, 1., 1-(norm_flux[contlo_min:conthi_max]).max(), line_center, 1., 0 # Amplitude, Center, STD, y-offset

    
    popt, pcov = curve_fit(f=mult_Gaussian,
                           xdata=wavelen[contlo_min:conthi_max],
                           ydata=norm_flux[contlo_min:conthi_max],
                           p0=init_param,
                           maxfev=50000)
    # Give the optimal parameters as caluclated by curve fit to the Gaussian model
    best_model = mult_Gaussian(wavelen,*popt)

    return popt, pcov, best_model

# Look at the 2.0920 micron feature cell to do better!
# Use Normalized Gaussian Distribution
# def gaussian_func(x,ampl,center,std,b):
#     return ((ampl)/(std*np.sqrt(2*np.pi)) * np.exp(-0.5*(x - center)**2 / (std**2))) + b

# # Find the indices for the min and max wavelengths of the spectral feature 
# linemin_idx = (wavlen-wave_lim1).abs().idxmin()
# linemax_idx = (wavlen-wave_lim2).abs().idxmin()

# wavemin = wavlen[linemin_idx]
# wavemax = wavlen[linemax_idx]

# # initial guesses, need 4 inputs: Amplitude, center, std_dev, b (y offset)
# init_params = -0.1,  wave_center, .01, 1.

# # Bounds on the parameters
# bound = ((-3.,wave_center-(25*spec_res),0,-1),(3,wave_center+(25*spec_res),10.,1.))

# popt, pcov = curve_fit(f=gaussian_func,
#                     xdata=wavlen.loc[contlo_min:conthi_max],
#                     ydata=cont_norm.loc[contlo_min:conthi_max],
#                     p0=init_params, bounds=bound,
#                     maxfev=1000)

# # Gaussian Fit to spectral feature using the best fit parameters
# best_model = gaussian_func(wavlen, *popt)

# print("Best Fit Parameters:", popt)