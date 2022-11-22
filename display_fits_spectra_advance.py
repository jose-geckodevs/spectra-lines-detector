from astropy.io import fits
from astropy.visualization import astropy_mpl_style
from astropy.table import Table
from astropy import units as u
from astropy.wcs import WCS
from astropy.visualization import quantity_support
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from specutils.manipulation import noise_region_uncertainty
from specutils.fitting import find_lines_threshold
from specutils.fitting import find_lines_derivative
from specutils.fitting import estimate_line_parameters
from specutils.manipulation import extract_region
from specutils.analysis import centroid, fwhm
import matplotlib.pyplot as plt
import numpy as np
import copy, os, sys, getopt, warnings

#path = "V5114Sgr/"

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
        for filename in os.listdir(path):
            hdul = fits.open(path + filename, mode="readonly", memmap = True)
            hdul.info()
            
            # Specutils way
            spec = Spectrum1D.read(path + filename, format='wcs1d-fits')
            if debug:
                print(repr(spec.meta['header']))
            
            # Limit the spectrum between 4000 and 7000 Angstrom
            flux, wavelength = limitSpectraArray(4000, 7000, spec)
            
            if debug:
                print(repr(flux))
                print(repr(wavelength))
            
            # Make a copy of the spectrum object with the new flux and wavelenght arrays
            meta = copy.copy(spec.meta)
            # Not sure if we need to modify the header, but we do just in case
            meta['header']['NAXIS1'] = len(wavelength)
            meta['header']['CRVAL1'] = wavelength.value[0]
            spec = Spectrum1D(spectral_axis=wavelength, flux=flux, meta=meta)
            
            #plt.plot(wavelength, flux)
            #plt.ylim(0, )
            #plt.xlim(6540, 6580)
            #plt.xlabel('Wavelength (Angstrom)')
            #plt.ylabel('Flux (erg/cm2/s/A)')
            
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
            ax1.set_xlim(6563-padding, 6563+padding)
            ax2.set_xlim(4861-padding, 4861+padding)
            ax3.set_xlim(4340-padding, 4340+padding)
            ax4.set_xlim(4101-padding, 4101+padding)
            #ax5.set_xlim(4000, 7000)
            #ax6.set_xlim(4000, 7000)
            #ax7.set_xlim(4000, 7000)
            
            ax2.set_ylabel("")
            ax3.set_ylabel("")
            ax4.set_ylabel("")
            
            ax1.plot(wavelength, flux);
            ax2.plot(wavelength, flux);
            ax3.plot(wavelength, flux);
            ax4.plot(wavelength, flux);
            ax5.plot(wavelength, flux);
            
            ax6.set_ylabel("Continuum")
            g1_fit = fit_generic_continuum(spec)
            y_continuum_fitted = g1_fit(wavelength)
            ax6.plot(wavelength, flux);
            ax6.plot(wavelength, y_continuum_fitted);
            
            ax7.set_ylabel("Normalised")
            spec_normalized = spec / y_continuum_fitted
            ax7.plot(spec_normalized.spectral_axis, spec_normalized.flux);
            
            # Find lines by thresholding
            noise_region = SpectralRegion(4000*u.AA, 7000*u.AA) # u.AA for Angstrom
            spec_noise = noise_region_uncertainty(spec, noise_region)
            lines = find_lines_threshold(spec_noise, noise_factor=1)
            print(repr(lines))
            # Find lines by derivating
            #lines = find_lines_derivative(spec, flux_threshold=0.75)
            #print(repr(lines))

            plt.savefig(path + filename + '.png')
            #plt.show()

            plt.clf()
            hdul.close()
            
            print('')
            #break # Just a test to only process the first spectrum of the folder
            
if __name__ == "__main__":
   main(sys.argv[1:])
   
