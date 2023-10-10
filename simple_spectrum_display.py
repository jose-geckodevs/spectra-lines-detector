from astropy.io import fits
from astropy.visualization import astropy_mpl_style
from astropy.visualization import quantity_support
from astropy import units as u
from specutils.spectra import Spectrum1D
import matplotlib.pyplot as plt
import sys, getopt, warnings
from tabulate import tabulate
from dateutil.parser import parse
import pandas as pd
import numpy as np

def main(argv):
    quantity_support() 
    plt.style.use(astropy_mpl_style)

    try:
        opts, args = getopt.getopt(argv,'p',['path=','separator='])
    except getopt.GetoptError:
        sys.exit(2)

    path = ''
    separator = '  '
    for opt, arg in opts:
        if opt in ('-p', '--path'):
            path = arg
        elif opt in ('--separator'):
            separator = arg

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        if path.endswith('.dat'):
            data = pd.read_csv(path, delimiter=separator, names=['wavelength','flux'], header=None)
            meta = {}
            meta['header'] = {}
            meta['header']['NAXIS1'] = len(data.wavelength)
            meta['header']['CRVAL1'] = data.wavelength[0]
            spec = Spectrum1D(spectral_axis=np.array(data.wavelength) * u.AA, flux=np.array(data.flux) * (u.erg / u.Angstrom / u.s / u.cm / u.cm), meta=meta)
        else:
            hdul = fits.open(path, mode='readonly', memmap = True)
            print(tabulate(hdul.info(False)) + '\n')
            print(hdul[1].data)
            hdul.close()
            
            spec = Spectrum1D.read(path)

        print(spec)

        fig, ax = plt.subplots()
        fig.set_figwidth(10)
        fig.set_figheight(7)
        ax.plot(spec.wavelength, spec.flux, label = 'Flux')
        ax.set(xlabel = 'Wavelenght', ylabel = "Normalised")
        plt.show()

if __name__ == '__main__':
   main(sys.argv[1:])
