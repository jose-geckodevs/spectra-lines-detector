from astropy.io import fits
from astropy import constants as const
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
from specutils.fitting import fit_lines
from astropy.modeling import models
from specutils.manipulation import extract_region
from specutils.analysis import centroid, fwhm, line_flux, equivalent_width
from dust_extinction.averages import *
from dust_extinction.parameter_averages import *
from dust_extinction.grain_models import *
from dust_extinction.shapes import *
import matplotlib.pyplot as plt
import numpy as np
import copy, os, sys, getopt, warnings
from tabulate import tabulate
from datetime import datetime, date
import pandas as pd
from dateutil.parser import parse
import math
import statistics
import yaml
import html

VERSION = "2.0"

def find_nearest_index(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def convert_symmetric_velocities(array_x_axis, array_y_axis):
    _indexZero = find_nearest_index(array_x_axis, 0)
    if (array_x_axis[_indexZero] < 0):
        _indexZero = _indexZero + 1
    _splice_array_x_axis = np.concatenate((np.flip(array_x_axis[_indexZero + 1:]) * -1, array_x_axis[_indexZero:]))
    _splice_array_y_axis = np.concatenate((np.flip(array_y_axis[_indexZero + 1:]), array_y_axis[_indexZero:]))
    return _splice_array_x_axis, _splice_array_y_axis
  
def is_date(string, format):
    try:
        return bool(datetime.strptime(string, format))
    except ValueError:
        return False
    
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

def reset_spectra_array_except_range(_low: int, _up: int, _spectrum: Spectrum1D):
    for index, wavelength in enumerate(_spectrum.wavelength):
         if wavelength.value < _low or wavelength.value > _up:
             _spectrum.flux.value[index] = 0
    return _spectrum        

def reduce_sorted_fits_array_by_filename(item: list):
    return item[0]

def reduce_sorted_fits_array_by_date(item: list):
    return item[1]

def measure_line_continuum_bigger_padding(_center: float, _spec_norm: Spectrum1D, _spec_flux: Spectrum1D, _angstromIncrement: int, _histogramStDevPercent: float):
    # Improved by using the bigger padding found (the previous simetric one would be like using the smaller one)
    _leftPadding = _angstromIncrement
    _rightPadding = _angstromIncrement
    _regions = []

    _flux, _wavelength = limit_spectra_array(_center - 200, _center + 200, _spec_flux)

    # Best calculate median from whole spectrum
    median = statistics.median(_spec_flux.flux.value)
    sd = statistics.stdev(_spec_flux.flux.value)

    min_continuum = median - sd * _histogramStDevPercent
    max_continuum = median + sd * _histogramStDevPercent

    _count_pass_continuum = 0
    while(_leftPadding < 200):
        _indexFlux = find_nearest_index(_spec_flux.wavelength.value, _center - _leftPadding)
        _flux = _spec_flux.flux[_indexFlux].value
        if (_flux <= max_continuum and _flux >= min_continuum):
            _count_pass_continuum = _count_pass_continuum + 1
            if (_count_pass_continuum > 2):
                break
        else:
            _count_pass_continuum = 0
        _leftPadding += 1

    _count_pass_continuum = 0
    while(_rightPadding < 200):
        _indexFlux = find_nearest_index(_spec_flux.wavelength.value, _center + _rightPadding)
        _flux = _spec_flux.flux[_indexFlux].value
        if (_flux <= max_continuum and _flux >= min_continuum):
            _count_pass_continuum = _count_pass_continuum + 1
            if (_count_pass_continuum > 2):
                break
        else:
            _count_pass_continuum = 0
        _rightPadding += 1

    if (_leftPadding >= 200):
        _leftPadding = 200 - _angstromIncrement
    if (_rightPadding >= 200):
        _rightPadding = 200 - _angstromIncrement

    _padding = _angstromIncrement
    if (_leftPadding > _rightPadding):
        _padding = _leftPadding
    else:
        _padding = _rightPadding
    _regions = [SpectralRegion((_center - _padding) * u.AA, (_center + _padding) * u.AA )]

    # Check if there is data in the region and if not return empty_measure_line_values()
    _check_flux, _wavelength = limit_spectra_array(_center - _padding, _center + _padding, _spec_flux)
    if (len(_check_flux) <= 0):
        return empty_measure_line_values()

    try:
        _fwhmData = fwhm(_spec_flux, regions = _regions)
        _fluxData = line_flux(_spec_flux, regions = _regions)
        _equivalentWidthData = equivalent_width(_spec_norm, continuum=1, regions = _regions)
        _centroidData = centroid(_spec_flux, regions = _regions)
    except:
        return empty_measure_line_values()

    # TODO: if we reach the padding limit, we could try call recursivelly this function increasing the histogram stdev percent

    return _fluxData[0], _fwhmData[0], _equivalentWidthData[0], _centroidData[0], _padding, _padding

def empty_measure_line_values():
    return [u.Quantity(0), u.Quantity(0), u.Quantity(0), u.Quantity(0), 0, 0]

def print_help():
    print('display_fits_spectra_advance.py')
    print('         --debug')
    print('         --only-one')
    print('         --path <include path for spectra folder>')
    print('         --datPath <include path for spectra folder>')
    print('         --datSeparator <separator for data file spectra>')
    print('         --ebv <Ebv dust extintion value>')
    print('         --rv <Rv dust extintion value>')
    print('         --model <dust extintion model>')
    print('         --angstromIncrement <int value to increment when finding lines continuum>')
    print('         --histogramStDevPercent <standard deviation from spectrum histogram to use when finding lines continuum>')
    print('         --linesConfig <path to lines configuration file>')
    print('         --centroidDifferenceInSpeed <int value>')
    print('         --continuumPolynomialModel <Polynomial1D, Chebyshev1D, Legendre1D, Hermite1D>')
    print('         --continuumPolynomialModelDegree <1, 2, 3, 4...>')
    print('         --saveContinuum <file prefix to save continuum to file>')
    print('         --loadContinuum <path to load continuum files>')
    print('If no wavelenght limtis configured, 4000 to 7000 Angstrom will be used')

def save_continuum_to_file(wavelength, continuum, output_path, filename):
    continuum_filename = os.path.join(output_path + '/continuum', f"{filename}_continuum.dat")
    np.savetxt(continuum_filename, np.column_stack((wavelength.value, continuum.value)), fmt=['%.4f', '%.6e'], delimiter=' ')

def html_to_mathtext(label):
    return label.replace('&', '$\\').replace(';', '$')

def load_lines_from_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['lines']

def main(argv):
    quantity_support() 
    plt.style.use(astropy_mpl_style)

    path = './'
    datPath = ''
    datSeparator = '  '
    debug = False
    onlyOne = False
    
    WavelenghtLowerLimit = 4000
    WavelenghtUpperLimit = 7000
    Ebv = 0
    Rv = 3.1
    Model = F99
    AngstromIncrement = 5
    HistogramStDevPercent = 0.5
    FolderSuffix = ''
    CentroidDifferenceInSpeed = 500
    ContinuumPolynomialModel = ''
    ContinuumPolynomialModelDegree = 1
    SaveContinuum = None
    LoadContinuum = None
    inputParams = ''
    linesConfigPath = ''

    try:
        opts, args = getopt.getopt(argv,'hp:d',['help','path=','datPath=','datSeparator=','debug','only-one','ebv=','rv=','model=',
                                                'wavelenghtLowerLimit=','wavelenghtUpperLimit=',
                                                'angstromIncrement=','histogramStDevPercent=',
                                                'linesConfig=','folderSuffix=','centroidDifferenceInSpeed=',
                                                'continuumPolynomialModel=','continuumPolynomialModelDegree=',
                                                'saveContinuum=','loadContinuum='])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    for opt, arg in opts:
        if opt not in ('-p', '--path', '--datPath', '--datSeparator', '--saveContinuum', '--loadContinuum'):
            inputParams += opt + arg

        if opt in ('-h', '--help'):
            print_help()
            sys.exit()
        elif opt in ('-p', '--path'):
            path = arg + '/'
        elif opt in ('--datPath'):
            datPath = arg + '/'
        elif opt in ('--datSeparator'):
            datSeparator = arg
        elif opt in ('-d', '--debug'):
            debug = True
        elif opt in ('--only-one'):
            onlyOne = True
        elif opt in ('--ebv'):
            Ebv = float(arg)
        elif opt in ('--rv'):
            Rv = float(arg)
        elif opt in ('--model'):
            Model = eval(arg)
        elif opt in ('--linesConfig'):
            linesConfigPath = arg
        elif opt in ('--wavelenghtLowerLimit'):
            WavelenghtLowerLimit = int(arg)
        elif opt in ('--wavelenghtUpperLimit'):
            WavelenghtUpperLimit = int(arg)
        elif opt in ('--angstromIncrement'):
            AngstromIncrement = int(arg)
        elif opt in ('--histogramStDevPercent'):
            HistogramStDevPercent = float(arg)
        elif opt in ('--folderSuffix'):
            FolderSuffix = arg
        elif opt in ('--continuumPolynomialModel'):
            ContinuumPolynomialModel = arg
        elif opt in ('--continuumPolynomialModelDegree'):
            ContinuumPolynomialModelDegree = int(arg)
        elif opt in ('--saveContinuum'):
            SaveContinuum = arg
        elif opt in ('--loadContinuum'):
            LoadContinuum = arg

    if (FolderSuffix != ''):
        inputParams = FolderSuffix

    if (datPath != ''):
        path = datPath

    if (inputParams == ''):
        output_path = path + 'default/'
    else:
        output_path = path + 'params' + inputParams + '/'

    # Prepare folder for files
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Prepare folder for regenreated spectra processed files
    if not os.path.exists(output_path + '/processed'):
        os.makedirs(output_path + '/processed')

    if SaveContinuum != None and not os.path.exists(output_path + '/continuum'):
        os.makedirs(output_path + '/continuum')

    if linesConfigPath:
        lines = load_lines_from_config(linesConfigPath)
    else:
        lines = []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        startTime = datetime.now()
        print('Start running at ' + startTime.strftime('%H:%M:%S'))

        if (datPath != ''):
            # Sort FITs by filename
            sortedFiles = []
            sortedDates = []
            for filename in os.listdir(path):
                if not filename.endswith('.dat'):
                    continue
                sortedFiles.append(filename)
                sortedDates.append(filename.replace('.dat', ''))
            
            if (len(sortedDates) <= 0):
                print('No DATs found on folder ' + path)
                sys.exit()

            sortedFiles = sorted(sortedFiles)
            sortedDates = sorted(sortedDates)

        else:
            # Sort FITs by date in header
            listFITS = []
            for filename in os.listdir(path):
                if not filename.endswith('.fits'):
                    continue
                spec = Spectrum1D.read(path + filename)
                if ('DATE-OBS' in spec.meta['header']):
                    listFITS.append([filename, spec.meta['header']['DATE-OBS']])
                else:
                    listFITS.append([filename, filename])
            
            if (len(listFITS) <= 0):
                print('No FITs found on folder ' + path)
                sys.exit()
            
            sortedFiles = list(map(reduce_sorted_fits_array_by_filename, sorted(listFITS)))
            sortedDates = list(map(reduce_sorted_fits_array_by_date, sorted(listFITS)))
        
        # Prepare csv report
        numLines = 0
        csv_report = open(output_path + 'lines_measurements.csv', 'w')
        csv_report.write('Spectra file;')
        for line in lines:
            csv_report.write(html.unescape(line['label']) + ' centroid;' + html.unescape(line['label']) + ' flux;' + html.unescape(line['label']) + ' deblended flux;' + html.unescape(line['label']) + ' model flux;' + html.unescape(line['label']) + ' eqw;' + html.unescape(line['label']) + ' fwhm;')
            numLines = numLines + 1
        csv_report.write('\n')

        # Prepare html graph report
        html_report = open(output_path + 'graph_report.html', 'w')
        html_report.write(f"<html><body>\n")
        html_report.write(f"<h1>Emission lines spectra analyser v{VERSION}</h1>\n")
        html_report.write(f"<h2>Lines: {', '.join([line['label'] for line in lines])}</h2>\n")
        html_report.write(f"<span>Generated at {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</span>\n")
        html_report.write(f"<br /><br />\n");
        counter = 0
        for filename in sortedFiles:
            if not filename.endswith('.fits') and not filename.endswith('.dat'):
                continue
            
            report = open(output_path + filename + '.txt', 'w')
            report.write('Filename: ' + filename + '\n')

            csv_report.write(filename + ';')
            
            html_report.write(f"<h3>{filename}</h3>\n")

            if filename.endswith('.dat'):
                # Read the spectrum from dat file
                data = pd.read_csv(path + filename, delimiter=datSeparator, names=['wavelength','flux'], header=None)
                meta = {}
                meta['header'] = {}
                meta['header']['NAXIS1'] = len(data.wavelength)
                meta['header']['CRVAL1'] = data.wavelength[0]
                spec_original = Spectrum1D(spectral_axis=np.array(data.wavelength) * u.AA, flux=np.array(data.flux) * (u.erg / u.Angstrom / u.s / u.cm / u.cm), meta=meta)

            else:
                # Read FITS
                hdul = fits.open(path + filename, mode='readonly', memmap = True)
                if debug:
                    report.write(tabulate(hdul.info(False)) + '\n')
                    report.write('\n')
                hdul.close()

                # Read the spectrum
                spec_original = Spectrum1D.read(path + filename)
                
                if debug:
                    report.write(repr(spec_original.meta['header']) + '\n')
                    report.write('\n')

                if ('DATE-OBS' in spec.meta['header']):
                    report.write('Date observation: ' + spec_original.meta['header']['DATE-OBS'] + '\n')
                if ('EXPTIME' in spec.meta['header']):
                    report.write('Exposure time: ' + str(spec_original.meta['header']['EXPTIME']) + '\n')
                if ('TELESCOP' in spec.meta['header']):
                    report.write('Telescope: ' + spec_original.meta['header']['TELESCOP'] + '\n')
                if ('INSTRUME' in spec.meta['header']):
                    report.write('Instrument: ' + spec_original.meta['header']['INSTRUME'] + '\n')
                if ('OBJECT' in spec.meta['header']):
                    report.write('Object: ' + spec_original.meta['header']['OBJECT'] + '\n')

            # Limit the spectrum between the lower and upper range
            flux, wavelength = limit_spectra_array(WavelenghtLowerLimit, WavelenghtUpperLimit, spec_original)
            report.write('Wavelenth limited from ' + str(WavelenghtLowerLimit) + ' to ' + str(WavelenghtUpperLimit) + ' Angstrom\n')

            if debug:
                report.write('Flux and wavelength spectra:' + '\n')
                report.write(repr(flux) + '\n')
                report.write(repr(wavelength) + '\n')
                report.write('\n')
            
            # Make a copy of the spectrum object with the new flux and wavelenght arrays
            meta = copy.copy(spec_original.meta)
            # Not sure if we need to modify the header, but we do just in case
            meta['header']['NAXIS1'] = len(wavelength)
            meta['header']['CRVAL1'] = wavelength.value[0]
            spec_limited = Spectrum1D(spectral_axis=wavelength, flux=flux, meta=meta)

            # Unextinguish (deredden) the spectrum if value <> 0
            if (Ebv == 0):
                spec = Spectrum1D(spectral_axis=spec_limited.wavelength, flux=spec_limited.flux, meta=spec_limited.meta)
            else:
                try:
                    ext = Model(Rv=Rv)
                except:
                    ext = Model()
                flux_ext = spec_limited.flux / ext.extinguish(spec_limited.spectral_axis, Ebv=Ebv)
                spec = Spectrum1D(spectral_axis=spec_limited.wavelength, flux=flux_ext, meta=spec_limited.meta)
                report.write('Applied dust extintion factor of: ' + str(Ebv) + ' and model ' + Model.__name__ + ' to deredden spectrum\n')
                report.write('\n')

            fig = plt.figure()
            figureColumns = 2
            fig.set_figwidth(15)
            columns = len(lines)
            rows = (columns // figureColumns) + (1 if columns % figureColumns != 0 else 0)
            fig.set_figheight((5 * rows) + (5 * figureColumns))
            gs = fig.add_gridspec(rows + 4, figureColumns)  # Ajustar el tamaño de la cuadrícula para incluir las filas adicionales

            axes = []
            for i in range(columns):
                row = i // figureColumns
                col = i % figureColumns
                axes.append(fig.add_subplot(gs[row, col]))
            
            ax5 = fig.add_subplot(gs[rows, :])
            ax6 = fig.add_subplot(gs[rows + 1, :])
            ax7 = fig.add_subplot(gs[rows + 2, :])
            ax8 = fig.add_subplot(gs[rows + 3, :])
            
            # Plot initial spectrum
            ax5.plot(spec_limited.wavelength, spec_limited.flux)

            if (LoadContinuum != None):
                # Load continuum from file
                continuum_filepath = os.path.join(LoadContinuum, f"{(counter+1):03}-combined.dat")
                continuum_data = pd.read_csv(continuum_filepath, delimiter=datSeparator, names=['wavelength','flux'], header=None)
                continuum_meta = {}
                continuum_meta['header'] = {}
                continuum_meta['header']['NAXIS1'] = len(data.wavelength)
                continuum_meta['header']['CRVAL1'] = data.wavelength[0]
                spec_continuum = Spectrum1D(spectral_axis=np.array(continuum_data.wavelength) * u.AA, flux=np.array(continuum_data.flux) * (u.erg / u.Angstrom / u.s / u.cm / u.cm), meta=continuum_meta)
                
                # Interpolate continuum to match the wavelength axis of the original spectrum
                y_continuum_fitted = np.interp(spec.wavelength.value, spec_continuum.wavelength.value, spec_continuum.flux.value) * spec_continuum.flux.unit

                ax6.plot(spec.wavelength, spec.flux)
                ax6.plot(spec.wavelength, y_continuum_fitted)

                spec_normalized = spec / y_continuum_fitted
                spec_flux = spec - y_continuum_fitted
                ax8.set_ylabel('Flux')
                ax8.plot(spec_flux.spectral_axis, spec_flux.flux)

                ax7.set_ylabel('Normalised')
                ax7.plot(spec_normalized.spectral_axis, spec_normalized.flux)
                ax7.set_ylim((-1, 2))

                # Find now lines by thresholding using the flux substracted contiuum spectrum
                noise_region = SpectralRegion(WavelenghtLowerLimit * u.AA, WavelenghtUpperLimit * u.AA)
                spec_noise = noise_region_uncertainty(spec_flux, noise_region)
                lines_found = find_lines_threshold(spec_noise, noise_factor=1)
                numLinesSecondIteration = len(lines_found)

            else:
                # Try find the lines without the continuum
                noise_region = SpectralRegion(WavelenghtLowerLimit * u.AA, WavelenghtUpperLimit * u.AA)
                spec_noise = noise_region_uncertainty(spec, noise_region)
                lines_found = find_lines_threshold(spec_noise, noise_factor=1)
                numLinesFirstIteration = len(lines_found)
                
                if debug:
                    # Try identify lines
                    lines_found.add_column(name='match', col='          ')
                    for row in lines_found:
                        for line in lines:
                            if abs(row[0].value - line['centroid']) < 10:
                                row[3] = line['label']
                                break
                        else:
                            row[3] = ''
                    
                    if debug:
                        report.write('Initial lines found (noise region uncertainty factor 1):' + '\n')
                        report.write(tabulate(lines_found, headers=['Line center','Type','Index']) + '\n')
                        report.write('\n')
                
                includeRegions = []
                excludeRegions = []
                wavelengthContinuumRegions = []
                fluxContinuumRegions = [] # As reference we will use the first flux value on the spectrum as include region and 0
                previousLine = 0
                padding = 50
                for row in lines_found:
                    if (previousLine <= 0 and row[0].value - padding > WavelenghtLowerLimit):
                        # First line found, add first part of the spectrum
                        includeRegions.append((WavelenghtLowerLimit, row[0].value - padding) * u.AA)
                        excludeRegions.append((row[0].value - padding, row[0].value + padding) * u.AA)
                        # Include first regions
                        fluxContinuumRegions.append(spec_limited.flux[0].value)
                        wavelengthContinuumRegions.append(WavelenghtLowerLimit)
                        fluxContinuumRegions.append(spec_limited.flux[0].value)
                        wavelengthContinuumRegions.append(row[0].value - padding)
                        fluxContinuumRegions.append(0)
                        wavelengthContinuumRegions.append(row[0].value - padding)
                        previousLine = row[0].value
                        
                    elif (previousLine > 0 and row[0].value - padding > previousLine and row[0].value + padding < WavelenghtUpperLimit):
                        includeRegions.append((previousLine + padding, row[0].value - padding) * u.AA)
                        excludeRegions.append((row[0].value - padding, row[0].value + padding) * u.AA)
                        # Include regions
                        fluxContinuumRegions.append(0)
                        wavelengthContinuumRegions.append(previousLine + padding)
                        fluxContinuumRegions.append(spec_limited.flux[0].value)
                        wavelengthContinuumRegions.append((previousLine + padding))
                        fluxContinuumRegions.append(spec_limited.flux[0].value)
                        wavelengthContinuumRegions.append((row[0].value - padding))
                        fluxContinuumRegions.append(0)
                        wavelengthContinuumRegions.append((row[0].value - padding))
                        previousLine = row[0].value
                    
                # Add last region until end of spectrum
                if (previousLine + padding < WavelenghtUpperLimit):
                    includeRegions.append((previousLine + padding, WavelenghtUpperLimit) * u.AA)
                    # Include last region
                    fluxContinuumRegions.append(0)
                    wavelengthContinuumRegions.append(previousLine + padding)
                    fluxContinuumRegions.append(spec_limited.flux[0].value)
                    wavelengthContinuumRegions.append(previousLine + padding)
                    fluxContinuumRegions.append(spec_limited.flux[0].value)
                    wavelengthContinuumRegions.append(WavelenghtUpperLimit)
                else:
                    # Include last region
                    fluxContinuumRegions.append(0)
                    wavelengthContinuumRegions.append(WavelenghtUpperLimit)
                    
                if debug:
                    report.write('Continuum include regions:' + '\n')
                    report.write(tabulate(includeRegions, headers=['Start','End']) + '\n')
                    report.write('\n')
                    
                    report.write('Continuum exclude regions:' + '\n')
                    report.write(tabulate(excludeRegions, headers=['Start','End']) + '\n')
                    report.write('\n')
                
                # Draw the continuum regions for reference
                if debug:
                    report.write('Continuum regions:' + '\n')
                    report.write(repr(fluxContinuumRegions) + '\n')
                    report.write(repr(wavelengthContinuumRegions) + '\n')
                    report.write('\n')
                ax5.plot(wavelengthContinuumRegions, fluxContinuumRegions)
                
                # If no lines found, be sure we add the whole spectrum
                if (len(includeRegions) <= 0):
                    includeRegions.append((WavelenghtLowerLimit, WavelenghtUpperLimit) * u.AA)        
                    
                ax6.set_ylabel('Continuum')

                # Exclude atmosferic windows
                excludeRegions.append((13000, 15000) * u.AA)
                excludeRegions.append((18000, 20000) * u.AA)

                # Try detect the contiuum on a first iteration
                if ContinuumPolynomialModel == 'Polynomial1D':
                    g1_fit = fit_continuum(spectrum=spec, model=models.Polynomial1D(ContinuumPolynomialModelDegree), exclude_regions=SpectralRegion(excludeRegions))
                elif ContinuumPolynomialModel == 'Chebyshev1D':
                    g1_fit = fit_continuum(spectrum=spec, model=models.Chebyshev1D(ContinuumPolynomialModelDegree), exclude_regions=SpectralRegion(excludeRegions))
                elif ContinuumPolynomialModel == 'Legendre1D':
                    g1_fit = fit_continuum(spectrum=spec, model=models.Legendre1D(ContinuumPolynomialModelDegree), exclude_regions=SpectralRegion(excludeRegions))
                elif ContinuumPolynomialModel == 'Hermite1D':
                    g1_fit = fit_continuum(spectrum=spec, model=models.Hermite1D(ContinuumPolynomialModelDegree), exclude_regions=SpectralRegion(excludeRegions))
                else:
                    g1_fit = fit_continuum(spec, exclude_regions=SpectralRegion(excludeRegions))

                y_continuum_fitted = g1_fit(spec.wavelength)
                
                # Guardar el continuo en un fichero
                if (SaveContinuum != None):
                    save_continuum_to_file(spec.wavelength, y_continuum_fitted, output_path, f"{(counter+1):03}_{SaveContinuum}")
                
                ax6.plot(spec.wavelength, spec.flux)
                ax6.plot(spec.wavelength, y_continuum_fitted)
                
                ax7.set_ylabel('Normalised')
                spec_normalized = spec / y_continuum_fitted
                spec_flux = spec - y_continuum_fitted
                ax8.set_ylabel('Flux')
                ax8.plot(spec_flux.spectral_axis, spec_flux.flux)
                
                # Find now lines by thresholding using the flux substracted contiuum spectrum
                noise_region = SpectralRegion(WavelenghtLowerLimit * u.AA, WavelenghtUpperLimit * u.AA)
                spec_noise = noise_region_uncertainty(spec_flux, noise_region)
                lines_found = find_lines_threshold(spec_noise, noise_factor=1)
                numLinesSecondIteration = len(lines_found)
                
                # Try fit continuum again with new lines found
                report.write('Num. lines first iteration: ' + str(numLinesFirstIteration) + '\n')
                report.write('Num. lines second iteration: ' + str(numLinesSecondIteration) + '\n')
                report.write('\n')
                if (numLinesFirstIteration != numLinesSecondIteration):
                    includeRegions = []
                    excludeRegions = []
                    wavelengthContinuumRegions = []
                    fluxContinuumRegions = [] # As reference we will use the first flux value on the spectrum as include region and 0
                    previousLine = 0
                    padding = 25
                    for row in lines_found:
                        if (previousLine <= 0 and row[0].value - padding > WavelenghtLowerLimit):
                            # First line found, add first part of the spectrum
                            includeRegions.append((WavelenghtLowerLimit, row[0].value - padding) * u.AA)
                            excludeRegions.append((row[0].value - padding, row[0].value + padding) * u.AA)
                            # Include first regions
                            fluxContinuumRegions.append(spec_limited.flux[0].value)
                            wavelengthContinuumRegions.append(WavelenghtLowerLimit)
                            fluxContinuumRegions.append(spec_limited.flux[0].value)
                            wavelengthContinuumRegions.append(row[0].value - padding)
                            fluxContinuumRegions.append(0)
                            wavelengthContinuumRegions.append(row[0].value - padding)
                            previousLine = row[0].value
                            
                        elif (previousLine > 0 and row[0].value - padding > previousLine and row[0].value + padding < WavelenghtUpperLimit):
                            includeRegions.append((previousLine + padding, row[0].value - padding) * u.AA)
                            excludeRegions.append((row[0].value - padding, row[0].value + padding) * u.AA)
                            # Include regions
                            fluxContinuumRegions.append(0)
                            wavelengthContinuumRegions.append(previousLine + padding)
                            fluxContinuumRegions.append(spec_limited.flux[0].value)
                            wavelengthContinuumRegions.append((previousLine + padding))
                            fluxContinuumRegions.append(spec_limited.flux[0].value)
                            wavelengthContinuumRegions.append((row[0].value - padding))
                            fluxContinuumRegions.append(0)
                            wavelengthContinuumRegions.append((row[0].value - padding))
                            previousLine = row[0].value
                        
                    # Add last region until end of spectrum
                    if (previousLine + padding < WavelenghtUpperLimit):
                        includeRegions.append((previousLine + padding, WavelenghtUpperLimit) * u.AA)
                        # Include last region
                        fluxContinuumRegions.append(0)
                        wavelengthContinuumRegions.append(previousLine + padding)
                        fluxContinuumRegions.append(spec_limited.flux[0].value)
                        wavelengthContinuumRegions.append(previousLine + padding)
                        fluxContinuumRegions.append(spec_limited.flux[0].value)
                        wavelengthContinuumRegions.append(WavelenghtUpperLimit)
                    else:
                        # Include last region
                        fluxContinuumRegions.append(0)
                        wavelengthContinuumRegions.append(WavelenghtUpperLimit)
                    
                    if debug:
                        report.write('New continuum include regions:' + '\n')
                        report.write(tabulate(includeRegions, headers=['Start','End']) + '\n')
                        report.write('\n')
                        
                        report.write('New continuum exclude regions:' + '\n')
                        report.write(tabulate(excludeRegions, headers=['Start','End']) + '\n')
                        report.write('\n')
                        
                    # Draw the continuum regions for reference
                    ax5.plot(wavelengthContinuumRegions, fluxContinuumRegions)
                    
                    # If no lines found, be sure we add the whole spectrum
                    if (len(includeRegions) <= 0):
                        includeRegions.append((WavelenghtLowerLimit, WavelenghtUpperLimit) * u.AA)        
                    
                    # Exclude atmosferic windows
                    excludeRegions.append((13000, 15000) * u.AA)
                    excludeRegions.append((18000, 20000) * u.AA)

                    # Try detect the contiuum on a second iteration
                    if ContinuumPolynomialModel == 'Polynomial1D':
                        g1_fit = fit_continuum(spectrum=spec, model=models.Polynomial1D(ContinuumPolynomialModelDegree), exclude_regions=SpectralRegion(excludeRegions))
                    elif ContinuumPolynomialModel == 'Chebyshev1D':
                        g1_fit = fit_continuum(spectrum=spec, model=models.Chebyshev1D(ContinuumPolynomialModelDegree), exclude_regions=SpectralRegion(excludeRegions))
                    elif ContinuumPolynomialModel == 'Legendre1D':
                        g1_fit = fit_continuum(spectrum=spec, model=models.Legendre1D(ContinuumPolynomialModelDegree), exclude_regions=SpectralRegion(excludeRegions))
                    elif ContinuumPolynomialModel == 'Hermite1D':
                        g1_fit = fit_continuum(spectrum=spec, model=models.Hermite1D(ContinuumPolynomialModelDegree), exclude_regions=SpectralRegion(excludeRegions))
                    else:
                        g1_fit = fit_continuum(spec, exclude_regions=SpectralRegion(excludeRegions))

                    y_continuum_fitted = g1_fit(spec.wavelength)

                    # Guardar el nuevo continuo en el mismo fichero
                    if (SaveContinuum != None):
                        save_continuum_to_file(spec.wavelength, y_continuum_fitted, output_path, f"{(counter+1):03}_{SaveContinuum}")
                        
                    ax6.plot(spec.wavelength, y_continuum_fitted)
                    
                    spec_normalized = spec / y_continuum_fitted
                    spec_flux = spec - y_continuum_fitted
                    ax7.clear()
                    ax7.set_ylabel('Normalised')
                    ax7.plot(spec_normalized.spectral_axis, spec_normalized.flux)
                    ax7.set_ylim((-1, 2))

                    ax8.clear()
                    ax8.set_ylabel('Flux')
                    ax8.plot(spec_flux.spectral_axis, spec_flux.flux)
                    
                    # Find now lines by thresholding using the flux substracted contiuum spectrum
                    noise_region = SpectralRegion(WavelenghtLowerLimit * u.AA, WavelenghtUpperLimit * u.AA)
                    spec_noise = noise_region_uncertainty(spec_flux, noise_region)
                    lines_found = find_lines_threshold(spec_noise, noise_factor=1)
                
            # Try identify lines
            lines_found.add_column(name='match', col='          ')
            for row in lines_found:
                for line in lines:
                    if abs(row[0].value - line['centroid']) < 10:
                        row[3] = line['label']
                        break
                else:
                    row[3] = ''
            
            # Finally find most prominent lines
            noise_region = SpectralRegion(WavelenghtLowerLimit * u.AA, WavelenghtUpperLimit * u.AA)
            spec_noise = noise_region_uncertainty(spec_flux, noise_region)
            lines_found = find_lines_threshold(spec_noise, noise_factor=0.5)
            report.write('Most prominent line by noise factor\n')
            grouped_lines = []
            partial_grouped_lines = []
            partial_type = ''
            padding = 50
            for row in lines_found:
                if (len(partial_grouped_lines) == 0):
                    partial_type = row[1]
                    partial_grouped_lines.append(row[0].value)
                if (partial_grouped_lines[-1] < row[0].value - padding):
                    grouped_lines.append([statistics.median(partial_grouped_lines) * u.AA, partial_type])
                    partial_grouped_lines = []

            report.write(tabulate(grouped_lines, headers=['Wavelength','Type']) + '\n')
            report.write('\n')

            # Plot individual lines
            for i, line in enumerate(lines):
                axes[i].clear()
                #axes[i].set_xlim(line['centroid'] - padding, line['centroid'] + padding)
                filtered_indices = (spec_flux.spectral_axis.value >= line['centroid'] - padding) & (spec_flux.spectral_axis.value <= line['centroid'] + padding)
                filtered_wavelength = spec_flux.spectral_axis[filtered_indices]
                filtered_flux = spec_flux.flux[filtered_indices]
                axes[i].set_xlabel(html_to_mathtext(line['label']))
                axes[i].plot(filtered_wavelength, filtered_flux)
            
            if debug:
                report.write('Found lines (noise region uncertainty factor 1):' + '\n')
                report.write(tabulate(lines_found, headers=['Line center','Type','Index','Match']) + '\n')
                report.write('\n')

            # Find lines by derivating
            lines_found = find_lines_derivative(spec_normalized, flux_threshold=0.95)
            
            # Try identify lines
            lines_found.add_column(name='match', col='          ')
            for row in lines_found:
                for line in lines:
                    if abs(row[0].value - line['centroid']) < 10:
                        row[3] = line['label']
                        break
                else:
                    row[3] = ''
                    
            if debug:
                report.write('Found lines (derivative threshold 0.95):' + '\n')
                report.write(tabulate(lines_found, headers=['Line center','Type','Index','Match']) + '\n')
                report.write('\n')

            # Measure lines finding paddigns
            line_calculations = []
            for i, line in enumerate(lines):
                line_calculations.append(None)
            for i, line in enumerate(lines):
                line_calculations[i] = measure_line_continuum_bigger_padding(line['centroid'], spec_normalized, spec_flux, AngstromIncrement, HistogramStDevPercent)

            fluxData = [calc[0] for calc in line_calculations]
            fwhmData = [calc[1] for calc in line_calculations]
            equivalentWidthData = [calc[2] for calc in line_calculations]
            centroidData = [calc[3] for calc in line_calculations]

            # Draw padding limits on line calculation
            for i, calc in enumerate(line_calculations):
                if calc[4] > 50 or calc[5] > 50:
                    axes[i].set_xlim(lines[i]['centroid'] - calc[4] - 10, lines[i]['centroid'] + calc[5] + 10)
                axes[i].axvline(x=lines[i]['centroid'] - calc[4], color='r')
                axes[i].axvline(x=lines[i]['centroid'] + calc[5], color='r')
                axes[i].axvline(x=lines[i]['centroid'], color='y', linestyle='dashed')

            # Plot main figure
            #plt.subplots_adjust(left=0.1, right=0.1, top=0.1, bottom=0.1)
            fig.tight_layout()
            plt.savefig(output_path + filename + '.png')
            plt.clf()

            # Plot lines average shape overlap with median and median symetric
            fig, ax = plt.subplots()
            fig.set_figwidth(15)
            fig.set_figheight(7)
            maxMargin = max([calc[4] for calc in line_calculations] + [calc[5] for calc in line_calculations])
            
            #speedMargin = 300000 * maxMargin / lines[0]['centroid']
            #speedMargin = 300000 * maxMargin / min([line['centroid'] for line in lines])
            speedMargin = 0
            for i, line in enumerate(lines):
                _speedMargin = 300000 * maxMargin / lines[i]['centroid']
                if (_speedMargin > speedMargin):
                    speedMargin = _speedMargin
            ax.set_xlim(-speedMargin, speedMargin)
            
            xs = []
            ys = []
            mins = []
            maxs = []
            lines_wavelength = []
            lines_flux = []
            max_lines_flux = []

            for i, line in enumerate(lines):
                xs.append(None)
                ys.append(None)
                mins.append(None)
                maxs.append(None)
                lines_flux.append(None)
                lines_wavelength.append(None)
                max_lines_flux.append(None)

            for i, line in enumerate(lines):
                lines_flux[i], lines_wavelength[i] = limit_spectra_array(line['centroid'] - maxMargin, line['centroid'] + maxMargin, spec_flux)
                
                # Find the max flux value at rest frequecy, so the reference value of each line
                max_lines_flux[i] = lines_flux[i][find_nearest_index(lines_wavelength[i].value, line['centroid'])].value

                xs[i] = 300000 * ((lines_wavelength[i].value - line['centroid']) / line['centroid'])
                ys[i] = lines_flux[i].value / max_lines_flux[i]

                mins[i] = min(xs[i])
                maxs[i] = max(xs[i])

            
            min_x = min(mins)
            max_x = max(maxs)

            # Try discard not fit lines from median
            ignore_deblending = []
            fit_calculations = []
            for i, line in enumerate(lines):
                ignore_deblending.append(False)
                fit_calculations.append(empty_measure_line_values())
            for i, line in enumerate(lines):
                # Try fitting a gaussian model on the line
                _spectrum = Spectrum1D(flux=lines_flux[i], spectral_axis=lines_wavelength[i])
                _indexFlux = find_nearest_index(lines_wavelength[i].value, line['centroid'])
                _g_init = models.Gaussian1D(amplitude=lines_flux[i][_indexFlux], mean=line['centroid'] * lines_wavelength[i].unit, stddev=statistics.stdev(lines_wavelength[i].value) * lines_wavelength[i].unit)
                _g_fit = fit_lines(_spectrum, _g_init)
                _y_fit = _g_fit(lines_wavelength[i])

                if (np.sum(_y_fit) != 0):
                    _spectrum_fit = Spectrum1D(flux=_y_fit, spectral_axis=lines_wavelength[i])
                    _y_continuum_interpolated = np.interp(lines_wavelength[i], spec.wavelength, y_continuum_fitted)
                    _spectrum_fit_norm = (_spectrum_fit + _y_continuum_interpolated) / _y_continuum_interpolated
                    fit_calculations[i] = measure_line_continuum_bigger_padding(line['centroid'], _spectrum_fit_norm, _spectrum_fit, AngstromIncrement, HistogramStDevPercent)

                    # Check if centroid calculated model is too far from reference, to discard line
                    centroidDifference = fit_calculations[i][3].value - line['centroid']
                    centroidDifferenceSpeed = 300000 * (centroidDifference / line['centroid'])
                    if (math.fabs(centroidDifferenceSpeed) > CentroidDifferenceInSpeed and math.fabs(fit_calculations[i][2].value) > 0):
                        ignore_deblending[i]= True
                else:
                    ignore_deblending[i]= True

            for i, line in enumerate(lines):
                if not ignore_deblending[i]:
                    ax.plot(xs[i], ys[i], label = html_to_mathtext(line['label']))
            ax.set(xlabel = f"{(u.kilometer / u.second)}", ylabel='Normalised')

            range_x_axis = np.arange(min_x, max_x, 50.0)
            median_y_axis = []
            stdev_y_axis = []
            for index, value in enumerate(range_x_axis):
                values_median = []
                for i, line in enumerate(lines):
                    if not ignore_deblending[i]:
                        _nearest = find_nearest_index(xs[i], value)
                        if (_nearest > 0 and _nearest < len(xs[i]) - 1) :
                            values_median.append(ys[i][_nearest])
                
                if (len(values_median) > 1):
                    median_y_axis.append(statistics.median(values_median))
                    stdev_y_axis.append(statistics.stdev(values_median))
                elif (len(values_median) == 1):
                    median_y_axis.append(values_median[0])
                    stdev_y_axis.append(0.0)
                else:
                    median_y_axis.append(0.0)
                    stdev_y_axis.append(0.0)
            
            ax.plot(range_x_axis, median_y_axis, label = 'Median', color='y', linestyle='dashed')
            symmetric_x_axis, symmetric_y_axis = convert_symmetric_velocities(range_x_axis, median_y_axis)
            ax.plot(symmetric_x_axis, symmetric_y_axis, label = 'Symmetric', color='m', linestyle='dashed')

            plt.legend()
            fig.tight_layout()
            plt.savefig(output_path + filename + '.lines_shape_overlap.png')
            plt.clf()

            # Restore median line for all lines and substract to the lines
            restored_median_xs = []
            restored_median_ys = []

            for i, line in enumerate(lines):
                restored_median_xs.append(None)
                restored_median_ys.append(None)
            for i, line in enumerate(lines):
                restored_median_xs[i] = ((range_x_axis / 300000) * line['centroid']) + line['centroid']
                restored_median_ys[i] = np.array(median_y_axis) * max_lines_flux[i]

            # Deblending process
            fig = plt.figure()
            figureColumns = 2
            fig.set_figwidth(15)
            columns = len(lines)
            rows = (columns // figureColumns) + (1 if columns % figureColumns != 0 else 0)
            fig.set_figheight(5 * rows)
            gs = fig.add_gridspec(rows, figureColumns)  # Ajustar el tamaño de la cuadrícula para incluir las filas adicionales
            
            axes = []
            for i in range(columns):
                row = i // figureColumns
                col = i % figureColumns
                axes.append(fig.add_subplot(gs[row, col]))
            
            lines_flux_interpolated = []
            lines_flux_deblended = []

            for i, line in enumerate(lines):
                lines_flux_interpolated.append(None)
                lines_flux_deblended.append(None)

            for i, line in enumerate(lines):
                axes[i].set_xlabel(html_to_mathtext(line['label']))
                axes[i].plot(lines_wavelength[i], lines_flux[i], label = 'l')

                if (not ignore_deblending[i]):
                    axes[i].plot(restored_median_xs[i], restored_median_ys[i], label = 'm')
                    lines_flux_interpolated[i] = np.interp(lines_wavelength[i].value, restored_median_xs[i], restored_median_ys[i])
                    lines_flux_deblended[i] = (lines_flux[i].value - (lines_flux[i].value - lines_flux_interpolated[i])) * lines_flux[i].unit
                    axes[i].plot(lines_wavelength[i], (lines_flux[i].value - lines_flux_interpolated[i]) * lines_flux[i].unit, label = 'l - m')
                    axes[i].plot(lines_wavelength[i], lines_flux_deblended[i], label = 'd')
                axes[i].axvline(x=line['centroid'], color='m', linestyle='dashed')
                axes[i].legend()

            plt.legend()
            fig.tight_layout()
            plt.savefig(output_path + filename + '.lines_deblending.png')
            plt.clf()

            # Recalculate the spectra substracting the deblended lines to the original (deredden) spectra
            all_lines = []
            for i, line in enumerate(lines):
                if not ignore_deblending[i]:
                    line_flux_deblended_interpolated = np.interp(spec.wavelength, lines_wavelength[i], lines_flux_deblended[i])
                    line_flux_deblended_interpolated_spec = Spectrum1D(spectral_axis=spec.wavelength, flux=line_flux_deblended_interpolated, meta=meta)

                    # Be sure we null all flux outside the lines
                    line_flux_deblended_interpolated_spec = reset_spectra_array_except_range(line['centroid'] - line_calculations[i][4], line['centroid'] + line_calculations[i][5], line_flux_deblended_interpolated_spec)
                    line_flux_deblended_interpolated = line_flux_deblended_interpolated_spec.flux

                    # Generate separate lines plots for reference
                    fig, ax = plt.subplots()
                    fig.set_figwidth(15)
                    fig.set_figheight(7)
                    ax.plot(spec.wavelength, line_flux_deblended_interpolated, label = html_to_mathtext(line['label']))
                    ax.set(xlabel = 'Wavelenght', ylabel = "Flux")
                    fig.tight_layout()
                    plt.savefig(output_path + filename + '.' + html.unescape(line['label']).lower() + '_deblended.png')
                    plt.clf()

                    if len(all_lines) > 0:
                        all_lines = all_lines + line_flux_deblended_interpolated
                    else:
                        all_lines = line_flux_deblended_interpolated

            if (len(all_lines) > 0):
                fig, ax = plt.subplots()
                fig.set_figwidth(15)
                fig.set_figheight(7)
                ax.plot(spec.wavelength, all_lines, label = 'All lines')
                ax.set(xlabel = 'Wavelenght', ylabel = "Flux")
                fig.tight_layout()
                plt.savefig(output_path + filename + '.all_deblended_lines.png')
                plt.clf()

                # Generate plot without lines
                fig, ax = plt.subplots()
                fig.set_figwidth(15)
                fig.set_figheight(7)
                ax.plot(spec.wavelength, spec.flux - all_lines, label = 'Processed')
                ax.set(xlabel = 'Wavelenght', ylabel = "Processed")
                fig.tight_layout()
                plt.savefig(output_path + filename + '.processed.png')
                plt.clf()

                # Generate another plot to compare the original deredden spectrum with the one without lines
                fig, ax = plt.subplots()
                fig.set_figwidth(15)
                fig.set_figheight(7)
                ax.plot(spec.wavelength, spec.flux, label = 'Original', color='y', linestyle='dashed')
                ax.plot(spec.wavelength, spec.flux - all_lines, label = 'Processed', color='c', linestyle='dashed')
                ax.set(xlabel = 'Wavelenght', ylabel = "Flux")
                fig.tight_layout()
                plt.savefig(output_path + filename + '.processed_comparission.png')
                plt.clf()
            
                # Finally save the processed spectra to file
                np.savetxt(output_path + '/processed/' + filename, np.column_stack((spec.wavelength.value, (spec.flux - all_lines).value)), fmt=['%.4f','%.6e'], delimiter=datSeparator)
            else:
                # Save the origial spectra to file if no lines to remove
                np.savetxt(output_path + '/processed/' + filename, np.column_stack((spec.wavelength.value, spec.flux.value)), fmt=['%.4f','%.6e'], delimiter=datSeparator)

            # Calculate values of deblended lines
            deblended_calculations = []
            for i, line in enumerate(lines):
                if not ignore_deblending[i]:
                    _spectrum_deblended = Spectrum1D(spectral_axis=lines_wavelength[i], flux=lines_flux_deblended[i], meta=meta)
                    _y_continuum_interpolated = np.interp(lines_wavelength[i], spec.wavelength, y_continuum_fitted)
                    _spectrum_deblended_norm = (_spectrum_deblended + _y_continuum_interpolated) / _y_continuum_interpolated
                    deblended_calculations.append(measure_line_continuum_bigger_padding(line['centroid'], _spectrum_deblended_norm, _spectrum_deblended, AngstromIncrement, HistogramStDevPercent))
                else:
                    deblended_calculations.append(empty_measure_line_values())

            # Add deblended values to report
            for i, line in enumerate(lines):
                line_values = np.array([line['label'], fluxData[i], fwhmData[i], equivalentWidthData[i], centroidData[i]])
                line_values = np.append(line_values, [deblended_calculations[i][0].value, deblended_calculations[i][1].value, deblended_calculations[i][2].value, deblended_calculations[i][3].value])
                line_values = np.append(line_values, [fit_calculations[i][0].value, fit_calculations[i][1].value, fit_calculations[i][2].value, fit_calculations[i][3].value])
                lines[i]['values'] = line_values

            # Write report
            report.write('Lines analisys' + '\n')
            report.write(tabulate([line['values'] for line in lines], headers=['Line', 'Flux', 'FWHM', 'Equivalent width', 'Centroid', 'Flux deblended', 'FWHM deblended', 'Equivalent width deblended', 'Centroid deblended', 'Flux model', 'FWHM model', 'Equivalent width model', 'Centroid model']) + '\n')
            
            if len(lines) > 0:
                report.write('* Units: ' + str(fluxData[0].unit))

            report.write('\n')

            # Write spreadsheet
            for i, line in enumerate(lines):
                if not ignore_deblending[i]:
                    csv_report.write(str(centroidData[i].value) + ';' + str(fluxData[i].value) + ';' + str(fit_calculations[i][0].value) + ';' + str(deblended_calculations[i][0].value) + ';' + str(equivalentWidthData[i].value) + ';' + str(fwhmData[i].value) + ';')
                else:
                    csv_report.write(str(centroidData[i].value) + ';' + str(fluxData[i].value) + ';;;' + str(equivalentWidthData[i].value) + ';' + str(fwhmData[i].value) + ';')
            csv_report.write('\n')

            # Close report
            report.close()

            # Write the PDF report
            html_report.write(f"<img src='./{filename + '.png'}'>\n")
            html_report.write(f"<br />\n");
            html_report.write(f"<img src='./{filename+ '.processed.png'}'>\n")
            html_report.write(f"<br />\n");
            html_report.write(f"<img src='./{filename + '.lines_deblending.png'}'>\n")
            html_report.write(f"<br />\n");
            html_report.write(f"<img src='./{filename + '.lines_shape_overlap.png'}'>\n")
            html_report.write(f"<br /><br />\n");

            counter = counter + 1

            print('Completed ' + filename + ' at ' + datetime.now().strftime('%H:%M:%S'))
            
            if (onlyOne):
                break # Just as test to only process the first spectrum of the folder

        csv_report.close()

        html_report.write(f"</body><html>\n")
        html_report.close()

        endTime = datetime.now()
        print('Completed at ' + datetime.now().strftime('%H:%M:%S'))
        print('The execution took ' + str(round((endTime - startTime).total_seconds(),0)) + ' seconds')
        
if __name__ == '__main__':
   main(sys.argv[1:])
