from astropy.io import fits
from astropy.table import Table
from astropy.visualization import astropy_mpl_style
from astropy.visualization import quantity_support
from astropy import units as u
from specutils.spectra import Spectrum1D
import matplotlib.pyplot as plt
import os, sys, getopt, warnings
from tabulate import tabulate
from dateutil.parser import parse
import pandas as pd
import numpy as np
from datetime import datetime

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

def reduce_sorted_array_by_filter(item: list):
    return item[0]

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

        files = []
        for filename in sorted(os.listdir(path)):
            if not filename.endswith('.fits'):
                continue

            hdul = fits.open(path + filename, mode='readonly')
            header = hdul[0].header

            # Prepare group by instrument
            filter_name = header['DISPELEM'].strip()
            wavelength_min = header['WAVELMIN']
            wavelength_max = header['WAVELMAX']
            files.append([filename, filter_name, wavelength_min, wavelength_max])

        # Group files per instrument
        grouped_files = {}
        for array in files:
            key = array[1] # Second element is used for grouping: ['UVB', 'NIR', 'VIS']
            if key not in grouped_files:
                grouped_files[key] = []
            grouped_files[key].append(array)

        # Sort filters per wavelenght
        different_keys = list(grouped_files.keys())
        unsorted_keys = []
        for filter in different_keys:
            wavelength_min = grouped_files[filter][0][2]
            wavelength_max = grouped_files[filter][0][3]
            unsorted_keys.append([filter, wavelength_min, wavelength_max])
        sorted_keys = sorted(unsorted_keys, key=lambda x: x[2])
        different_sorted_keys = list(map(reduce_sorted_array_by_filter, sorted_keys))

        # Group per epoc and instrument, we assume they all in order and alternated
        filter_epoc_grouped = []
        count = 0
        for index, value in enumerate(grouped_files[different_sorted_keys[0]]):
            filter_epoc_grouped.append({})
            for filter in different_sorted_keys:
                if len(grouped_files[filter]) > index:
                    filename = grouped_files[filter][index]
                    filter_epoc_grouped[count][filter] = filename[0]
            count = count + 1

        # Ignore incomplete groups
        cleaned_filter_epoc_grouped = []
        for group in filter_epoc_grouped:
            if (len(group) == len(different_keys)):
                cleaned_filter_epoc_grouped.append(group)

        # Process each group now
        for group in cleaned_filter_epoc_grouped:
            fig, ax = plt.subplots()
            fig.set_figwidth(10)
            fig.set_figheight(7)

            grouped_flux = []
            grouped_wavelenght = []

            print('Procesing group:')
            for key in group:
                filename = group[key]
                print(filename + ' - ' + key)

                table = Table.read(path + filename)
                # ['WAVE', 'FLUX', 'ERR', 'QUAL', 'SNR', 'FLUX_REDUCED', 'ERR_REDUCED']
                for column in table.colnames:
                    column_data = table[column]
                    if column == 'WAVE':
                        wavelength = column_data.data.flatten() * column_data.unit
                    if column == 'FLUX':
                        flux = column_data.data.flatten() * column_data.unit
                    if column == 'SNR':
                        snr = column_data.data.flatten()
                    if column == 'ERR':
                        err = column_data.data.flatten()

                # Remove data if SNR is too low
                count = 0
                clean_flux = []
                clean_wavelenght = []
                for value in flux.value:
                    if snr[count] > 5 and (len(grouped_wavelenght) == 0 or wavelength.value[count] > grouped_wavelenght[-1]):
                        clean_flux.append(value)
                        clean_wavelenght.append(wavelength.value[count])
                    count = count + 1

                grouped_flux.extend(clean_flux)
                grouped_wavelenght.extend(clean_wavelenght)
            
            spec = Spectrum1D(spectral_axis=(grouped_wavelenght * wavelength.unit).to(u.AA), flux=(grouped_flux * flux.unit).to(u.erg / u.Angstrom / u.s / u.cm / u.cm))
            flux, wavelength = limit_spectra_array(WavelenghtLowerLimit, WavelenghtUpperLimit, spec)
            
            # Save combined fits into dat file
            if not os.path.exists(path + 'combined'):
                os.makedirs(path + 'combined')
            combined_dat_filename = group[different_sorted_keys[0]] + '-combined.dat'
            np.savetxt(path + 'combined/' + combined_dat_filename, np.column_stack((wavelength.value, flux.value)), fmt=['%.4f','%.6e'], delimiter=separator)

            # Plot the combined spectrum
            ax.plot(wavelength, flux)
            ax.set(xlabel = 'Wavelenght', ylabel = "Flux")
            combined_image_filename = group[different_sorted_keys[0]] + '-combined.png'
            plt.savefig(path + 'combined/' + combined_image_filename)
            plt.clf()
            #plt.show()

            #break # Test onyl first one

if __name__ == '__main__':
   main(sys.argv[1:])
