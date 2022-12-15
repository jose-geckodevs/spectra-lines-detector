from astropy.io import fits
from astropy.visualization import astropy_mpl_style
from astropy.table import Table
from astropy import units as u
from astropy.wcs import WCS
from astropy.visualization import quantity_support
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum, fit_continuum
from specutils.manipulation import noise_region_uncertainty
from specutils.fitting import find_lines_threshold
from specutils.fitting import find_lines_derivative
from specutils.fitting import estimate_line_parameters
from specutils.manipulation import extract_region
from specutils.analysis import centroid, fwhm
import matplotlib.pyplot as plt
import numpy as np
import copy, os, sys, getopt, warnings
from tabulate import tabulate

Halpha = 6563
Hbeta = 4861
Hgamma = 4341
Hdelta = 4102
WavelenghtLowerLimit = 4000
WavelenghtUpperLimit = 7000

def limitSpectraArray(low, up, spectrum):
    foundLowerIndex = 0
    foundUpperIndex = 0
    for index, wavelength in enumerate(spectrum.wavelength):
        #print(index, wavelength.value, foundLowerIndex, foundUpperIndex);
        if wavelength.value >= low and spectrum.flux.value[index] != 0 and not foundLowerIndex:
            foundLowerIndex = index
        if wavelength.value >= up and not foundUpperIndex:
            foundUpperIndex = index
            break;
    return spectrum.flux.value[foundLowerIndex:foundUpperIndex] * spectrum.flux.unit, spectrum.wavelength.value[foundLowerIndex:foundUpperIndex] * spectrum.wavelength.unit

def main(argv):
    quantity_support() 
    plt.style.use(astropy_mpl_style)
    
    path = './'
    debug = False
    
    try:
        opts, args = getopt.getopt(argv,"hp:d:",["path=","debug"])
    except getopt.GetoptError:
        print('display_fits_spectra_advance.py -p <include path for spectra> -d <debug mode true or false>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('display_fits_spectra_advance.py -p <include path for spectra> -d <debug mode true or false>')
            sys.exit()
        elif opt in ("-p", "--path"):
            path = arg + '/'
        elif opt in ("-d", "--debug"):
            debug = arg == 'true'
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        print('Trying to analyse spectra on folder ' + os.path.abspath(path) + ' ...')
        print('')
        
        for filename in os.listdir(path):
            if not filename.endswith(".fits"):
                continue
            
            hdul = fits.open(path + filename, mode="readonly", memmap = True)
            hdul.info()
            print('')
            
            # Read the spectrum
            spec = Spectrum1D.read(path + filename, format='wcs1d-fits')
            if debug:
                print(repr(spec.meta['header']))
                print('')
            
            # Limit the spectrum between the lower and upper range
            flux, wavelength = limitSpectraArray(WavelenghtLowerLimit, WavelenghtUpperLimit, spec)
            
            if debug:
                print('Flux and wavelength spectra:')
                print(repr(flux))
                print(repr(wavelength))
                print('')
            
            # Make a copy of the spectrum object with the new flux and wavelenght arrays
            meta = copy.copy(spec.meta)
            # Not sure if we need to modify the header, but we do just in case
            meta['header']['NAXIS1'] = len(wavelength)
            meta['header']['CRVAL1'] = wavelength.value[0]
            spec = Spectrum1D(spectral_axis=wavelength, flux=flux, meta=meta)
            
            fig = plt.figure()
            fig.suptitle(filename)
            fig.set_figheight(15)
            gs = fig.add_gridspec(4,4)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            ax4 = fig.add_subplot(gs[0, 3])
            ax5 = fig.add_subplot(gs[1, :])
            ax6 = fig.add_subplot(gs[2, :])
            ax7 = fig.add_subplot(gs[3, :])
            
            padding = 50
            ax1.set_xlim(Halpha - padding, Halpha + padding)
            ax2.set_xlim(Hbeta - padding, Hbeta + padding)
            ax3.set_xlim(Hgamma - padding, Hgamma + padding)
            ax4.set_xlim(Hdelta - padding, Hdelta + padding)
            
            ax1.set_xlabel("Halpha")
            ax2.set_xlabel("Hbeta")
            ax3.set_xlabel("Hgamma")
            ax4.set_xlabel("Hdelta")
            
            ax2.set_ylabel("")
            ax3.set_ylabel("")
            ax4.set_ylabel("")
            
            ax1.plot(wavelength, flux)
            ax2.plot(wavelength, flux)
            ax3.plot(wavelength, flux)
            ax4.plot(wavelength, flux)
            ax5.plot(wavelength, flux)
            
            # Try find the continuum without the lines
            noise_region = SpectralRegion(WavelenghtLowerLimit * u.AA, WavelenghtUpperLimit * u.AA) # u.AA for Angstrom
            spec_noise = noise_region_uncertainty(spec, noise_region)
            lines = find_lines_threshold(spec_noise, noise_factor=1)
            includeRegions = []
            excludeRegions = []
            wavelengthContinuumRegions = []
            fluxContinuumRegions = [] # As reference we will use the first flux value on the spectrum as include region and 0
            previousLine = 0
            padding = 50
            for row in lines:
                if (previousLine <= 0 and row[0].value - padding > WavelenghtLowerLimit): 
                    # First line found, add first part of the spectrum
                    includeRegions.append((WavelenghtLowerLimit, row[0].value - padding) * u.AA)
                    # Include first regions
                    fluxContinuumRegions.append(flux[0].value)
                    wavelengthContinuumRegions.append(WavelenghtLowerLimit)
                    fluxContinuumRegions.append(flux[0].value)
                    wavelengthContinuumRegions.append(row[0].value - padding)
                    fluxContinuumRegions.append(0)
                    wavelengthContinuumRegions.append(row[0].value - padding)
                elif (row[0].value - padding > previousLine and row[0].value + padding < WavelenghtUpperLimit):
                    includeRegions.append((previousLine + padding, row[0].value - padding) * u.AA)
                    # Include regions
                    fluxContinuumRegions.append(0)
                    wavelengthContinuumRegions.append(previousLine + padding)
                    fluxContinuumRegions.append(flux[0].value)
                    wavelengthContinuumRegions.append((previousLine + padding))
                    fluxContinuumRegions.append(flux[0].value)
                    wavelengthContinuumRegions.append((row[0].value - padding))
                    fluxContinuumRegions.append(0)
                    wavelengthContinuumRegions.append((row[0].value - padding))
                previousLine = row[0].value
                
            # Add last region until end of spectrum
            if (previousLine + padding < WavelenghtUpperLimit):
                includeRegions.append((previousLine + padding, WavelenghtUpperLimit) * u.AA)
                # Include last region
                fluxContinuumRegions.append(flux[0].value)
                wavelengthContinuumRegions.append(previousLine + padding)
                fluxContinuumRegions.append(flux[0].value)
                wavelengthContinuumRegions.append(WavelenghtUpperLimit)
            else:
                # Include last region
                fluxContinuumRegions.append(0)
                wavelengthContinuumRegions.append(WavelenghtUpperLimit)
                
            print('Continuum include regions:')
            print(tabulate(includeRegions, headers=['Start','End']))
            print('')
            
            # Draw the continuum regions for reference
            if debug:
                print('Continuum regions:')
                print(repr(fluxContinuumRegions))
                print(repr(wavelengthContinuumRegions))
                print('')
            ax5.plot(wavelengthContinuumRegions, fluxContinuumRegions);
            
            # If no lines found, be sure we add the whole spectrum
            if (len(includeRegions) <= 0):
                includeRegions.append((WavelenghtLowerLimit, WavelenghtUpperLimit) * u.AA)
            
            ax6.set_ylabel("Continuum")
            g1_fit = fit_generic_continuum(spec, exclude_regions=[SpectralRegion(6500 * u.AA, 6600 * u.AA)])
            #g1_fit = fit_continuum(spec, window=includeRegions)
            y_continuum_fitted = g1_fit(wavelength)
            ax6.plot(wavelength, flux);
            ax6.plot(wavelength, y_continuum_fitted);
            
            ax7.set_ylabel("Normalised")
            spec_normalized = spec / y_continuum_fitted
            ax7.plot(spec_normalized.spectral_axis, spec_normalized.flux);
            
            # Find now lines by thresholding using the normalised spectrum
            noise_region = SpectralRegion(WavelenghtLowerLimit * u.AA, WavelenghtUpperLimit * u.AA) # u.AA for Angstrom
            spec_noise = noise_region_uncertainty(spec_normalized, noise_region)
            lines = find_lines_threshold(spec_noise, noise_factor=1)
            
            # Try identify Balmer series
            lines.add_column(name='match', col='          ')
            for row in lines:
                if (abs(row[0].value - Halpha) < 10):
                    row[3] = 'H alpha'
                elif (abs(row[0].value - Hbeta) < 10):
                    row[3] = 'H beta'
                elif (abs(row[0].value - Hgamma) < 10):
                    row[3] = 'H gamma'
                elif (abs(row[0].value - Hdelta) < 10):
                    row[3] = 'H delta'
                else:
                    row[3] = ''
            
            print('Found lines (noise region uncertainty factor 1):')
            print(tabulate(lines, headers=['Line center','Type','Index','Match']))
            print('')

            # Find lines by derivating
            lines = find_lines_derivative(spec_normalized, flux_threshold=0.95)
            
            # Try identify Balmer series
            lines.add_column(name='match', col='          ')
            for row in lines:
                if (abs(row[0].value - Halpha) < 10):
                    row[3] = 'H alpha'
                elif (abs(row[0].value - Hbeta) < 10):
                    row[3] = 'H beta'
                elif (abs(row[0].value - Hgamma) < 10):
                    row[3] = 'H gamma'
                elif (abs(row[0].value - Hdelta) < 10):
                    row[3] = 'H delta'
                else:
                    row[3] = ''
                    
            print('Found lines (derivative threshold 0.95):')
            print(tabulate(lines, headers=['Line center','Type','Index','Match']))

            plt.savefig(path + filename + '.png')
            #plt.show()

            plt.clf()
            hdul.close()
            
            print('')
            #break # Just a test to only process the first spectrum of the folder
            
if __name__ == "__main__":
   main(sys.argv[1:])
   
