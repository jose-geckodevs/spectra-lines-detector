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

def limit_spectra_array(_low: int, _up: int, _spectrum: Spectrum1D):
    _foundLowerIndex = 0
    _foundUpperIndex = 0
    for index, wavelength in enumerate(_spectrum.wavelength):
        if wavelength.value >= _low and _spectrum.flux.value[index] != 0 and not _foundLowerIndex:
            _foundLowerIndex = index
        if wavelength.value >= _up and not _foundUpperIndex:
            _foundUpperIndex = index
            break
    if (not _foundUpperIndex):
        _foundUpperIndex = index
    return _spectrum.flux.value[_foundLowerIndex:_foundUpperIndex] * _spectrum.flux.unit, _spectrum.wavelength.value[_foundLowerIndex:_foundUpperIndex] * _spectrum.wavelength.unit

def main(argv):
    quantity_support() 
    plt.style.use(astropy_mpl_style)

    try:
        opts, args = getopt.getopt(argv,'p',['path=','separator=',
                                            'wavelenghtLowerLimit=','wavelenghtUpperLimit=',])
    except getopt.GetoptError:
        sys.exit(2)

    path = ''
    separator = '  '
    WavelenghtLowerLimit = 4000
    WavelenghtUpperLimit = 7000

    for opt, arg in opts:
        if opt in ('-p', '--path'):
            path = arg
        elif opt in ('--separator'):
            separator = arg
        elif opt in ('--wavelenghtLowerLimit'):
            WavelenghtLowerLimit = int(arg)
        elif opt in ('--wavelenghtUpperLimit'):
            WavelenghtUpperLimit = int(arg)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        fig, ax = plt.subplots()
        fig.set_figwidth(10)
        fig.set_figheight(7)

        files = path.split(',')
        print(files)

        for filename in files:
            print('Procesing ' + filename)
            if filename.endswith('.dat'):
                data = pd.read_csv(filename, delimiter=separator, names=['wavelength','flux'], header=None)
                meta = {}
                meta['header'] = {}
                meta['header']['NAXIS1'] = len(data.wavelength)
                meta['header']['CRVAL1'] = data.wavelength[0]
                spec = Spectrum1D(spectral_axis=np.array(data.wavelength) * u.AA, flux=np.array(data.flux) * (u.erg / u.Angstrom / u.s / u.cm / u.cm), meta=meta)
            else:
                hdul = fits.open(filename, mode='readonly', memmap = True)
                print(tabulate(hdul.info(False)) + '\n')
                print(hdul[0].data)
                hdul.close()
                spec = Spectrum1D.read(filename)

            for header in spec.meta['header']:
                if (header):
                    print(str(header) + ' = ' + str(spec.meta['header'][header]))
                    
            flux, wavelength = limit_spectra_array(WavelenghtLowerLimit, WavelenghtUpperLimit, spec)
            ax.plot(wavelength, flux)

        ax.set(xlabel = 'Wavelenght', ylabel = "Flux")
        plt.show()

if __name__ == '__main__':
   main(sys.argv[1:])
