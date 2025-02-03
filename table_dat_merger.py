import os
import sys
import getopt
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style, quantity_support
from specutils.spectra import Spectrum1D
from astropy import units as u

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
        opts, args = getopt.getopt(argv, 'p', ['path=', 'separator=', 'wavelenghtLowerLimit=', 'wavelenghtUpperLimit='])
    except getopt.GetoptError:
        sys.exit(2)

    path = ''
    separator = '  '
    WavelenghtLowerLimit = 1000
    WavelenghtUpperLimit = 25000

    for opt, arg in opts:
        if opt in ('-p', '--path'):
            path = arg
            if (path[-1] != '/'):
                path = path + '/'
        elif opt in ('--separator'):
            separator = arg
        elif opt in ('--wavelenghtLowerLimit'):
            WavelenghtLowerLimit = int(arg)
        elif opt in ('--wavelenghtUpperLimit'):
            WavelenghtUpperLimit = int(arg)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        files = sorted([f for f in os.listdir(path) if f.endswith('.dat')])

        counter = 0
        for i in range(0, len(files), 3):
            group = files[i:i+3]
            if len(group) < 3:
                continue

            # Ordenar el grupo
            group.sort(key=lambda x: ('UVB' not in x, 'VIS' not in x, 'NIR' not in x))

            fig, ax = plt.subplots()
            fig.set_figwidth(10)
            fig.set_figheight(7)

            grouped_flux = []
            grouped_wavelenght = []

            print('Procesing group:')
            for filename in group:
                print(filename)

                data = np.loadtxt(path + filename)
                wavelength = data[:, 0] * u.AA
                flux = data[:, 1] * (u.erg / u.Angstrom / u.s / u.cm / u.cm)

                grouped_flux.extend(flux.value)
                grouped_wavelenght.extend(wavelength.value)

            spec = Spectrum1D(spectral_axis=(grouped_wavelenght * u.AA), flux=(grouped_flux * (u.erg / u.Angstrom / u.s / u.cm / u.cm)))
            flux, wavelength = limit_spectra_array(WavelenghtLowerLimit, WavelenghtUpperLimit, spec)

            if not os.path.exists(path + 'combined'):
                os.makedirs(path + 'combined')

            combined_dat_filename = f"{(counter+1):03}-combined.dat"
            np.savetxt(path + 'combined/' + combined_dat_filename, np.column_stack((wavelength.value, flux.value)), fmt=['%.4f','%.6e'], delimiter=separator)

            ax.plot(wavelength, flux)
            ax.set(xlabel='Wavelenght', ylabel='Flux')
            combined_image_filename = f"{(counter+1):03}-combined.png"
            plt.savefig(path + 'combined/' + combined_image_filename)
            plt.clf()

            counter = counter + 1

if __name__ == '__main__':
    main(sys.argv[1:])
