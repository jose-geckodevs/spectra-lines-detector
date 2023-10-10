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

HALPHA_REF = 6563
HBETA_REF = 4861
HGAMMA_REF = 4341
HDELTA_REF = 4102

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

def measure_lines_fixed(_center: float, _spec_norm: Spectrum1D):
    """Deprecated"""
    _padding = 50
    _regions = [SpectralRegion((_center - _padding) * u.AA, (_center + _padding) * u.AA )]
    _fluxData = line_flux(_spec_norm, regions = _regions)
    _fwhmData = fwhm(_spec_norm, regions = _regions)
    _equivalentWidthData = equivalent_width(_spec_norm, continuum=1, regions = _regions)
    _centroidData = centroid(_spec_norm, regions = _regions)

    return _fluxData, _fwhmData, _equivalentWidthData, _centroidData, _padding

def measure_line_max_fwhm(_center: float, _spec_norm: Spectrum1D, _spec_flux: Spectrum1D):
    """Deprecated"""
    _padding = 5
    _precision = 2
    _previousFwhm = u.Quantity(0)
    _regions = []
    while(_padding < 100):
        _regions = [SpectralRegion((_center - _padding) * u.AA, (_center + _padding) * u.AA )]
        _fwhmData = fwhm(_spec_norm, regions = _regions)
        _indexLeftFlux = find_nearest_index(_spec_flux.wavelength.value, _center - _padding)
        _indexRightFlux = find_nearest_index(_spec_flux.wavelength.value, _center + _padding)
        if ((_spec_flux.flux[_indexLeftFlux] < 0 or _spec_flux.flux[_indexRightFlux] < 0) and round(_fwhmData[0].value, _precision) <= round(_previousFwhm.value, _precision)):
            break
        _previousFwhm = _fwhmData[0]
        _padding += 5

    _fluxData = line_flux(_spec_flux, regions = _regions)
    _equivalentWidthData = equivalent_width(_spec_norm, continuum=1, regions = _regions)
    _centroidData = centroid(_spec_norm, regions = _regions)

    if (_padding >= 100):
        _padding = 95
    return _fluxData[0], _previousFwhm, _equivalentWidthData[0], _centroidData[0], _padding, _padding

def measure_line_max_fwhm_asimetric(_center: float, _spec_norm: Spectrum1D, _spec_flux: Spectrum1D):
    """Deprecated"""
    _leftPadding = 5
    _rightPadding = 5
    _precision = 2
    _previousFwhm = u.Quantity(0)
    _regions = []
    while(_leftPadding < 100):
        _regions = [SpectralRegion((_center - _leftPadding) * u.AA, _center * u.AA )]
        _fwhmData = fwhm(_spec_norm, regions = _regions)
        _indexFlux = find_nearest_index(_spec_flux.wavelength.value, _center - _leftPadding)
        if (_spec_flux.flux[_indexFlux] < 0 and round(_fwhmData[0].value, _precision) <= round(_previousFwhm.value, _precision)):
            break
        _previousFwhm = _fwhmData[0]
        _leftPadding += 5

    while(_rightPadding < 100):
        _regions = [SpectralRegion((_center - _leftPadding) * u.AA, (_center + _rightPadding) * u.AA )]
        _fwhmData = fwhm(_spec_norm, regions = _regions)
        _indexFlux = find_nearest_index(_spec_flux.wavelength.value, _center + _rightPadding)
        if (_spec_flux.flux[_indexFlux] < 0 and round(_fwhmData[0].value, _precision) <= round(_previousFwhm.value, _precision)):
            break
        _previousFwhm = _fwhmData[0]
        _rightPadding += 5

    _fluxData = line_flux(_spec_flux, regions = _regions)
    _equivalentWidthData = equivalent_width(_spec_norm, continuum=1, regions = _regions)
    _centroidData = centroid(_spec_norm, regions = _regions)

    if (_leftPadding >= 100):
        _leftPadding = 95
    if (_rightPadding >= 100):
        _rightPadding = 95
    return _fluxData[0], _previousFwhm, _equivalentWidthData[0], _centroidData[0], _leftPadding, _rightPadding

def measure_line_continuum_asimetric(_center: float, _spec_norm: Spectrum1D, _spec_flux: Spectrum1D, _angstromIncrement: int, _histogramStDevPercent: float):
    _leftPadding = _angstromIncrement
    _rightPadding = _angstromIncrement
    _regions = []

    _flux, _wavelength = limit_spectra_array(_center - 200, _center + 200, _spec_flux)

    #median = statistics.median(_flux.value)
    #sd = statistics.stdev(_flux.value)
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
    _regions = [SpectralRegion((_center - _leftPadding) * u.AA, (_center + _rightPadding) * u.AA )]

    _fwhmData = fwhm(_spec_norm, regions = _regions)
    _fluxData = line_flux(_spec_flux, regions = _regions)
    _equivalentWidthData = equivalent_width(_spec_norm, continuum=1, regions = _regions)
    _centroidData = centroid(_spec_norm, regions = _regions)

    # TODO: if we reach the padding limit, we could try call recursivelly this function increasing the histogram stdev percent

    return _fluxData[0], _fwhmData[0], _equivalentWidthData[0], _centroidData[0], _leftPadding, _rightPadding

def measure_line_continuum_simetric(_center: float, _spec_norm: Spectrum1D, _spec_flux: Spectrum1D, _angstromIncrement: int, _histogramStDevPercent: float):
    _padding = _angstromIncrement
    _regions = []

    _flux, _wavelength = limit_spectra_array(_center - 200, _center + 200, _spec_flux)

    #median = statistics.median(_flux.value)
    #sd = statistics.stdev(_flux.value)
    # Best calculate median from whole spectrum
    median = statistics.median(_spec_flux.flux.value)
    sd = statistics.stdev(_spec_flux.flux.value)

    min_continuum = median - sd * _histogramStDevPercent
    max_continuum = median + sd * _histogramStDevPercent

    _count_pass_continuum = 0
    while(_padding < 200):
        _indexLeftFlux = find_nearest_index(_spec_flux.wavelength.value, _center - _padding)
        _indexRightFlux = find_nearest_index(_spec_flux.wavelength.value, _center + _padding)
        _leftFlux = _spec_flux.flux[_indexLeftFlux].value
        _rightFlux = _spec_flux.flux[_indexRightFlux].value
        if ((_leftFlux <= max_continuum and _leftFlux >= min_continuum) or (_rightFlux <= max_continuum and _rightFlux >= min_continuum)):
            _count_pass_continuum = _count_pass_continuum + 1
            if (_count_pass_continuum > 2):
                break
        else:
            _count_pass_continuum = 0
        _padding += 1

    if (_padding >= 200):
        _padding = 200 - _angstromIncrement
    _regions = [SpectralRegion((_center - _padding) * u.AA, (_center + _padding) * u.AA )]

    _fwhmData = fwhm(_spec_norm, regions = _regions)
    _fluxData = line_flux(_spec_flux, regions = _regions)
    _equivalentWidthData = equivalent_width(_spec_norm, continuum=1, regions = _regions)
    _centroidData = centroid(_spec_norm, regions = _regions)

    # TODO: if we reach the padding limit, we could try call recursivelly this function increasing the histogram stdev percent

    return _fluxData[0], _fwhmData[0], _equivalentWidthData[0], _centroidData[0], _padding, _padding

def measure_line_continuum_bigger_padding(_center: float, _spec_norm: Spectrum1D, _spec_flux: Spectrum1D, _angstromIncrement: int, _histogramStDevPercent: float):
    # Improved by using the bigger padding found (the previous simetric one would be like using the smaller one)
    _leftPadding = _angstromIncrement
    _rightPadding = _angstromIncrement
    _regions = []

    _flux, _wavelength = limit_spectra_array(_center - 200, _center + 200, _spec_flux)

    #median = statistics.median(_flux.value)
    #sd = statistics.stdev(_flux.value)
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

    _fwhmData = fwhm(_spec_norm, regions = _regions)
    _fluxData = line_flux(_spec_flux, regions = _regions)
    _equivalentWidthData = equivalent_width(_spec_norm, continuum=1, regions = _regions)
    _centroidData = centroid(_spec_norm, regions = _regions)

    # TODO: if we reach the padding limit, we could try call recursivelly this function increasing the histogram stdev percent

    return _fluxData[0], _fwhmData[0], _equivalentWidthData[0], _centroidData[0], _padding, _padding

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
    print('         --l[1,2,3,4]centroid <line angstrom centroid> --l[1,2,3,4]label <line label>')
    print('         --l[1,2,3,4]centroid <line angstrom centroid> --l[1,2,3,4]label <line label>')
    print('         --evolutionlabel <evolution label>')
    print('If no wavelenght limtis configured, 4000 to 7000 Angstrom will be used')
    print('If no lines configured, Halpha(4), Hbeta(3), Hgamma(2) and Hdelta(1) will be used')

def main(argv):
    quantity_support() 
    plt.style.use(astropy_mpl_style)

    path = './'
    datPath = ''
    datSeparator = '  '
    debug = False
    onlyOne = False
    
    Halpha = HALPHA_REF
    HalphaLabel = 'Halpha'
    Hbeta = HBETA_REF
    HbetaLabel = 'Hbeta'
    Hgamma = HGAMMA_REF
    HgammaLabel = 'Hgamma'
    Hdelta = HDELTA_REF
    HdeltaLabel = 'Hdelta'
    EvolutionLabel = 'Days after maximum'
    WavelenghtLowerLimit = 4000
    WavelenghtUpperLimit = 7000
    Ebv = 0
    Rv = 3.1
    Model = F99
    AngstromIncrement = 5
    HistogramStDevPercent = 0.5
    FolderSuffix = ''
    inputParams = ''

    try:
        opts, args = getopt.getopt(argv,'hp:d',['help','path=','datPath=','datSeparator=','debug','only-one','ebv=','rv=','model=',
                                                'wavelenghtLowerLimit=','wavelenghtUpperLimit=',
                                                'angstromIncrement=','histogramStDevPercent=',
                                                'l1centroid=','l2centroid=','l3centroid=','l4centroid=',
                                                'l1label=','l2label=','l3label=','l4label=',
                                                'folderSuffix=','evolutionlabel='])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    for opt, arg in opts:
        if opt not in ('-p', '--path', '--datPath', '--datSeparator'):
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
        elif opt in ('--l1centroid'):
            Hdelta = int(arg)
        elif opt in ('--l1label'):
            HdeltaLabel = arg
        elif opt in ('--l2centroid'):
            Hgamma = int(arg)
        elif opt in ('--l2label'):
            HgammaLabel = arg
        elif opt in ('--l3centroid'):
            Hbeta = int(arg)
        elif opt in ('--l3label'):
            HbetaLabel = arg
        elif opt in ('--l4centroid'):
            Halpha = int(arg)
        elif opt in ('--l4label'):
            HalphaLabel = arg
        elif opt in ('--evolutionlabel'):
            EvolutionLabel = arg
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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        HalphaEvolution = []
        HbetaEvolution = []
        HgammaEvolution = []
        HdeltaEvolution = []
        Halpha_Hbeta = []
        Hgamma_Hbeta = []
        Hdelta_Hbeta = []
        HalphaFWHMEvolution = []
        HbetaFWHMEvolution = []
        HgammaFWHMEvolution = []
        HdeltaFWHMEvolution = []
        evolutionPlane = []
        evolutionPlaneLog = False
        
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
                listFITS.append([filename, spec.meta['header']['DATE-OBS']])
            
            if (len(listFITS) <= 0):
                print('No FITs found on folder ' + path)
                sys.exit()
            
            sortedFiles = list(map(reduce_sorted_fits_array_by_filename, sorted(listFITS)))
            sortedDates = list(map(reduce_sorted_fits_array_by_date, sorted(listFITS)))

        # Set evolution graphs to log if number of days
        if (sortedDates[0].isdigit()):
            evolutionPlaneLog = True
        elif (is_date(sortedDates[0], '%Y-%m-%d')):
            evolutionPlaneLog = True
        else:
            evolutionPlaneLog = False
            
        csv = open(output_path + 'lines_measurements.csv', 'w')
        csv.write('Spectra file;')
        csv.write(HalphaLabel + ' centroid;' + HalphaLabel + ' flux;' + HalphaLabel + ' deblended flux;' + HalphaLabel + ' eqw;' + HalphaLabel + ' fwhm;')
        csv.write(HbetaLabel + ' centroid;' + HbetaLabel + ' flux;' + HbetaLabel + ' deblended flux;' + HbetaLabel + ' eqw;' + HbetaLabel + ' fwhm;')
        csv.write(HgammaLabel + ' centroid;' + HgammaLabel + ' flux;' + HgammaLabel + ' deblended flux;' + HgammaLabel + ' eqw;' + HgammaLabel + ' fwhm;')
        csv.write(HdeltaLabel + ' centroid;' + HdeltaLabel + ' flux;' + HdeltaLabel + ' deblended flux;' + HdeltaLabel + ' eqw;' + HdeltaLabel + ' fwhm')
        csv.write('\n')

        counter = 0
        for filename in sortedFiles:
            if not filename.endswith('.fits') and not filename.endswith('.dat'):
                continue
            
            report = open(output_path + filename + '.txt', 'w')
            report.write('Filename: ' + filename + '\n')

            csv.write(filename + ';')
            
            if filename.endswith('.dat'):
                # Read the spectrum from dat file
                data = pd.read_csv(path + filename, delimiter=datSeparator, names=['wavelength','flux'], header=None)
                meta = {}
                meta['header'] = {}
                meta['header']['NAXIS1'] = len(data.wavelength)
                meta['header']['CRVAL1'] = data.wavelength[0]
                meta['header']['DATE-OBS'] = sortedDates[counter]
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

                report.write('Date observation: ' + spec_original.meta['header']['DATE-OBS'] + '\n')
                report.write('Exposure time: ' + str(spec_original.meta['header']['EXPTIME']) + '\n')
                report.write('Telescope: ' + spec_original.meta['header']['TELESCOP'] + '\n')
                report.write('Instrument: ' + spec_original.meta['header']['INSTRUME'] + '\n')
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
            fig.suptitle(filename)
            fig.set_figwidth(15)
            fig.set_figheight(25)
            gs = fig.add_gridspec(5,4)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            ax4 = fig.add_subplot(gs[0, 3])
            ax5 = fig.add_subplot(gs[1, :])
            ax6 = fig.add_subplot(gs[2, :])
            ax7 = fig.add_subplot(gs[3, :])
            ax8 = fig.add_subplot(gs[4, :])
            
            # Plot initial spectrum
            ax5.plot(spec_limited.wavelength, spec_limited.flux)
            
            # Try find the lines without the lines initally without the continuum
            noise_region = SpectralRegion(WavelenghtLowerLimit * u.AA, WavelenghtUpperLimit * u.AA) # u.AA for Angstrom
            spec_noise = noise_region_uncertainty(spec, noise_region)
            lines = find_lines_threshold(spec_noise, noise_factor=1)
            numLinesFirstIteration = len(lines)
            
            if debug:
                # Try identify lines
                lines.add_column(name='match', col='          ')
                for row in lines:
                    if (abs(row[0].value - Halpha) < 10):
                        row[3] = HalphaLabel
                    elif (abs(row[0].value - Hbeta) < 10):
                        row[3] = HbetaLabel
                    elif (abs(row[0].value - Hgamma) < 10):
                        row[3] = HgammaLabel
                    elif (abs(row[0].value - Hdelta) < 10):
                        row[3] = HdeltaLabel
                    else:
                        row[3] = ''
                
                if debug:
                    report.write('Initial lines found (noise region uncertainty factor 1):' + '\n')
                    report.write(tabulate(lines, headers=['Line center','Type','Index']) + '\n')
                    report.write('\n')
            
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

            # Try detect the contiuum on a first iteration
            #g1_fit = fit_generic_continuum(spec, exclude_regions=SpectralRegion(excludeRegions))
            #g1_fit = fit_continuum(spec, window=includeRegions)
            g1_fit = fit_continuum(spec, exclude_regions=SpectralRegion(excludeRegions))
            #g1_fit = fit_continuum(spec, window=includeRegions, exclude_regions=SpectralRegion(excludeRegions))

            y_continuum_fitted = g1_fit(spec.wavelength)
            ax6.plot(spec.wavelength, spec.flux)
            ax6.plot(spec.wavelength, y_continuum_fitted)
            
            ax7.set_ylabel('Normalised')
            spec_normalized = spec / y_continuum_fitted
            spec_flux = spec - y_continuum_fitted
            ax7.plot(spec_normalized.spectral_axis, spec_normalized.flux)

            ax8.set_ylabel('Flux')
            ax8.plot(spec_flux.spectral_axis, spec_flux.flux)
            
            # Find now lines by thresholding using the flux substracted contiuum spectrum
            noise_region = SpectralRegion(WavelenghtLowerLimit * u.AA, WavelenghtUpperLimit * u.AA) # u.AA for Angstrom
            spec_noise = noise_region_uncertainty(spec_flux, noise_region)
            lines = find_lines_threshold(spec_noise, noise_factor=1)
            numLinesSecondIteration = len(lines)
            
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
                for row in lines:
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
                    
                # Try detect the contiuum on a second iteration
                #g1_fit = fit_generic_continuum(spec, exclude_regions=SpectralRegion(excludeRegions))
                #g1_fit = fit_continuum(spec, window=includeRegions)
                g1_fit = fit_continuum(spec, exclude_regions=SpectralRegion(excludeRegions))
                #g1_fit = fit_continuum(spec, window=includeRegions, exclude_regions=SpectralRegion(excludeRegions))
                y_continuum_fitted = g1_fit(spec.wavelength)
                ax6.plot(spec.wavelength, y_continuum_fitted)
                
                spec_normalized = spec / y_continuum_fitted
                spec_flux = spec - y_continuum_fitted
                ax7.clear()
                ax7.set_ylabel('Normalised')
                ax7.plot(spec_normalized.spectral_axis, spec_normalized.flux)

                ax8.clear()
                ax8.set_ylabel('Flux')
                ax8.plot(spec_flux.spectral_axis, spec_flux.flux)
                
                # Find now lines by thresholding using the flux substracted contiuum spectrum
                noise_region = SpectralRegion(WavelenghtLowerLimit * u.AA, WavelenghtUpperLimit * u.AA) # u.AA for Angstrom
                spec_noise = noise_region_uncertainty(spec_flux, noise_region)
                lines = find_lines_threshold(spec_noise, noise_factor=1)
                
            # Try identify lines
            lines.add_column(name='match', col='          ')
            for row in lines:
                if (abs(row[0].value - Halpha) < 10):
                    row[3] = HalphaLabel
                elif (abs(row[0].value - Hbeta) < 10):
                    row[3] = HbetaLabel
                elif (abs(row[0].value - Hgamma) < 10):
                    row[3] = HgammaLabel
                elif (abs(row[0].value - Hdelta) < 10):
                    row[3] = HdeltaLabel
                else:
                    row[3] = ''
            
            # Plot individual H lines
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            
            padding = 50
            ax1.set_xlim(Halpha - padding, Halpha + padding)
            ax2.set_xlim(Hbeta - padding, Hbeta + padding)
            ax3.set_xlim(Hgamma - padding, Hgamma + padding)
            ax4.set_xlim(Hdelta - padding, Hdelta + padding)
            
            ax1.set_xlabel(HalphaLabel)
            ax2.set_xlabel(HbetaLabel)
            ax3.set_xlabel(HgammaLabel)
            ax4.set_xlabel(HdeltaLabel)
            
            ax2.set_ylabel('')
            ax3.set_ylabel('')
            ax4.set_ylabel('')
            
            ax1.plot(spec_flux.spectral_axis, spec_flux.flux)
            ax2.plot(spec_flux.spectral_axis, spec_flux.flux)
            ax3.plot(spec_flux.spectral_axis, spec_flux.flux)
            ax4.plot(spec_flux.spectral_axis, spec_flux.flux)
            
            if debug:
                report.write('Found lines (noise region uncertainty factor 1):' + '\n')
                report.write(tabulate(lines, headers=['Line center','Type','Index','Match']) + '\n')
                report.write('\n')

            # Find lines by derivating
            lines = find_lines_derivative(spec_normalized, flux_threshold=0.95)
            
            # Try identify lines
            lines.add_column(name='match', col='          ')
            for row in lines:
                if (abs(row[0].value - Halpha) < 10):
                    row[3] = HalphaLabel
                elif (abs(row[0].value - Hbeta) < 10):
                    row[3] = HbetaLabel
                elif (abs(row[0].value - Hgamma) < 10):
                    row[3] = HgammaLabel
                elif (abs(row[0].value - Hdelta) < 10):
                    row[3] = HdeltaLabel
                else:
                    row[3] = ''
                    
            if debug:
                report.write('Found lines (derivative threshold 0.95):' + '\n')
                report.write(tabulate(lines, headers=['Line center','Type','Index','Match']) + '\n')
                report.write('\n')
            
            # Measure lines finding paddigns
            haCalculations = measure_line_continuum_bigger_padding(Halpha, spec_normalized, spec_flux, AngstromIncrement, HistogramStDevPercent)
            hbCalculations = measure_line_continuum_bigger_padding(Hbeta, spec_normalized, spec_flux, AngstromIncrement, HistogramStDevPercent)
            hgCalculations = measure_line_continuum_bigger_padding(Hgamma, spec_normalized, spec_flux, AngstromIncrement, HistogramStDevPercent)
            hdCalculations = measure_line_continuum_bigger_padding(Hdelta, spec_normalized, spec_flux, AngstromIncrement, HistogramStDevPercent)
            fluxData = [haCalculations[0], hbCalculations[0], hgCalculations[0], hdCalculations[0]]
            fwhmData = [haCalculations[1], hbCalculations[1], hgCalculations[1], hdCalculations[1]]
            equivalentWidthData = [haCalculations[2], hbCalculations[2], hgCalculations[2], hdCalculations[2]]
            centroidData = [haCalculations[3], hbCalculations[3], hgCalculations[3], hdCalculations[3]]

            # Draw padding limits on line calculation
            if (haCalculations[4] > 50 or haCalculations[5] > 50):
                ax1.set_xlim(Halpha - haCalculations[4] - 10, Halpha + haCalculations[5] + 10)
            if (hbCalculations[4] > 50 or hbCalculations[5] > 50):
                ax2.set_xlim(Hbeta - hbCalculations[4] - 10, Hbeta + hbCalculations[5] + 10)
            if (hgCalculations[4] > 50 or hgCalculations[5] > 50):
                ax3.set_xlim(Hgamma - hgCalculations[4] - 10, Hgamma + hgCalculations[5] + 10)
            if (hdCalculations[4] > 50 or hdCalculations[5] > 50):
                ax4.set_xlim(Hdelta - hdCalculations[4] - 10, Hdelta + hdCalculations[5] + 10)

            ax1.axvline(x=Halpha-haCalculations[4], color='r')
            ax1.axvline(x=Halpha+haCalculations[5], color='r')
            ax2.axvline(x=Hbeta-hbCalculations[4], color='r')
            ax2.axvline(x=Hbeta+hbCalculations[5], color='r')
            ax3.axvline(x=Hgamma-hgCalculations[4], color='r')
            ax3.axvline(x=Hgamma+hgCalculations[5], color='r')
            ax4.axvline(x=Hdelta-hdCalculations[4], color='r')
            ax4.axvline(x=Hdelta+hdCalculations[5], color='r')

            haValues = np.array([HalphaLabel, fluxData[0], fwhmData[0], equivalentWidthData[0], centroidData[0]])
            hbValues = np.array([HbetaLabel, fluxData[1], fwhmData[1], equivalentWidthData[1], centroidData[1]])
            hgValues = np.array([HgammaLabel, fluxData[2], fwhmData[2], equivalentWidthData[2], centroidData[2]])
            hdValues = np.array([HdeltaLabel, fluxData[3], fwhmData[3], equivalentWidthData[3], centroidData[3]])
            
            # Calculate evolution graphs
            HalphaEvolution.append(fluxData[0].value)
            HbetaEvolution.append(fluxData[1].value)
            HgammaEvolution.append(fluxData[2].value)
            HdeltaEvolution.append(fluxData[3].value)
            
            if (Halpha == HALPHA_REF and Hbeta == HBETA_REF and Hgamma == HGAMMA_REF and Hdelta == HDELTA_REF):
                # Only generate this graph if measuring the main H lines
                Halpha_Hbeta.append(fluxData[0] / fluxData[1])
                Hgamma_Hbeta.append(fluxData[2] / fluxData[1])
                Hdelta_Hbeta.append(fluxData[3] / fluxData[1])

            HalphaFWHMEvolution.append((fwhmData[0].value / centroidData[0].value) * const.c.to('km/s').value)
            HbetaFWHMEvolution.append((fwhmData[1].value / centroidData[1].value) * const.c.to('km/s').value)
            HgammaFWHMEvolution.append((fwhmData[2].value / centroidData[2].value) * const.c.to('km/s').value)
            HdeltaFWHMEvolution.append((fwhmData[3].value / centroidData[3].value) * const.c.to('km/s').value)

            if (sortedDates[counter].isdigit()):
                evolutionPlane.append(int(sortedDates[counter]))
            elif (is_date(sortedDates[counter], '%Y-%m-%d')):
                # Get number of days from maximum (first date is day 1)
                delta = datetime.strptime(sortedDates[counter], '%Y-%m-%d') - datetime.strptime(sortedDates[0], '%Y-%m-%d')
                evolutionPlane.append(delta.days + 1)
            else:
                # Order by string
                evolutionPlane.append(sortedDates[counter])
            
            # Plot main figure
            plt.savefig(output_path + filename + '.png')
            plt.clf()

            # Plot lines average shape overlap with median and median symetric
            fig, ax = plt.subplots()
            fig.set_figwidth(10)
            fig.set_figheight(7)
            maxMargin = np.max([haCalculations[4], haCalculations[5], hbCalculations[4], hbCalculations[5], hgCalculations[4], hgCalculations[5], hdCalculations[4], hdCalculations[5]])
            speedMargin = 300000 * maxMargin / Halpha
            ax.set_xlim(-speedMargin, speedMargin)
            fluxHa, wavelengthHa = limit_spectra_array(Halpha - haCalculations[4], Halpha + haCalculations[5], spec_flux)
            fluxHb, wavelengthHb = limit_spectra_array(Hbeta - hbCalculations[4], Hbeta + hbCalculations[5], spec_flux)
            fluxHg, wavelengthHg = limit_spectra_array(Hgamma - hgCalculations[4], Hgamma + hgCalculations[5], spec_flux)
            fluxHd, wavelengthHd = limit_spectra_array(Hdelta - hdCalculations[4], Hdelta + hdCalculations[5], spec_flux)
            
            # Find the max flux value at rest frequecy, so the reference value of each line
            maxHalpha = fluxHa[find_nearest_index(wavelengthHa.value, Halpha)].value
            maxHbeta = fluxHb[find_nearest_index(wavelengthHb.value, Hbeta)].value
            maxHgamma = fluxHg[find_nearest_index(wavelengthHg.value, Hgamma)].value
            maxHdelta = fluxHd[find_nearest_index(wavelengthHd.value, Hdelta)].value
            
            xs = [
                300000 * ((wavelengthHa.value - Halpha) / Halpha),
                300000 * ((wavelengthHb.value - Hbeta) / Hbeta),
                300000 * ((wavelengthHg.value - Hgamma) / Hgamma),
                300000 * ((wavelengthHd.value - Hdelta) / Hdelta)
            ]
            ys = [
                fluxHa.value / maxHalpha,
                fluxHb.value / maxHbeta,
                fluxHg.value / maxHgamma,
                fluxHd.value / maxHdelta
            ]
            ax.plot(xs[0], ys[0], label = HalphaLabel)
            ax.plot(xs[1], ys[1], label = HbetaLabel)
            ax.plot(xs[2], ys[2], label = HgammaLabel)
            ax.plot(xs[3], ys[3], label = HdeltaLabel)
            ax.set(xlabel = f"{(u.kilometer / u.second)}", ylabel='Normalised')
            
            min_x = min([min(xs[0]), min(xs[1]), min(xs[2]), min(xs[3])])
            max_x = max([max(xs[0]), max(xs[1]), max(xs[2]), max(xs[3])])
            range_x_axis = np.arange(min_x, max_x, 50.0)
            median_y_axis = []
            stdev_y_axis = []
            for index, value in enumerate(range_x_axis):
                _indexHa = find_nearest_index(xs[0], value)
                _indexHb = find_nearest_index(xs[1], value)
                _indexHg = find_nearest_index(xs[2], value)
                _indexHd = find_nearest_index(xs[3], value)
                median_y_axis.append(statistics.median([ys[0][_indexHa], ys[1][_indexHb], ys[2][_indexHg], ys[3][_indexHd]]))
                stdev_y_axis.append(statistics.stdev([ys[0][_indexHa], ys[1][_indexHb], ys[2][_indexHg], ys[3][_indexHd]]))
            
            ax.plot(range_x_axis, median_y_axis, label = 'Median', color='y', linestyle='dashed')
            symmetric_x_axis, symmetric_y_axis = convert_symmetric_velocities(range_x_axis, median_y_axis)
            ax.plot(symmetric_x_axis, symmetric_y_axis, label = 'Symmetric', color='m', linestyle='dashed')

            plt.legend()
            plt.savefig(output_path + filename + '.lines_shape_overlap.png')
            plt.clf()

            # Restore median line for all 4 lines and substract to the lines
            restored_median_xs = [
                ((range_x_axis / 300000) * Halpha) + Halpha,
                ((range_x_axis / 300000) * Hbeta) + Hbeta,
                ((range_x_axis / 300000) * Hgamma) + Hgamma,
                ((range_x_axis / 300000) * Hdelta) + Hdelta
            ]
            restored_median_ys = [
                np.array(median_y_axis) * maxHalpha,
                np.array(median_y_axis) * maxHbeta,
                np.array(median_y_axis) * maxHgamma,
                np.array(median_y_axis) * maxHdelta
            ]

            fig, ax = plt.subplots()
            fig.set_figwidth(15)
            fig.set_figheight(5)
            gs = fig.add_gridspec(1,4)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            ax4 = fig.add_subplot(gs[0, 3])
            ax1.set_xlabel(HalphaLabel)
            ax2.set_xlabel(HbetaLabel)
            ax3.set_xlabel(HgammaLabel)
            ax4.set_xlabel(HdeltaLabel)
            ax2.set_ylabel('')
            ax3.set_ylabel('')
            ax4.set_ylabel('')

            fluxHa_interpolated = np.interp(wavelengthHa.value, restored_median_xs[0], restored_median_ys[0])
            fluxHb_interpolated = np.interp(wavelengthHb.value, restored_median_xs[1], restored_median_ys[1])
            fluxHg_interpolated = np.interp(wavelengthHg.value, restored_median_xs[2], restored_median_ys[2])
            fluxHd_interpolated = np.interp(wavelengthHd.value, restored_median_xs[3], restored_median_ys[3])
            
            fluxHa_deblended = (fluxHa.value - (fluxHa.value - fluxHa_interpolated)) * fluxHa.unit
            fluxHb_deblended = (fluxHb.value - (fluxHb.value - fluxHb_interpolated)) * fluxHb.unit
            fluxHg_deblended = (fluxHg.value - (fluxHg.value - fluxHg_interpolated)) * fluxHg.unit
            fluxHd_deblended = (fluxHd.value - (fluxHd.value - fluxHd_interpolated)) * fluxHd.unit

            ax1.plot(wavelengthHa, fluxHa, label = 'l')
            ax1.plot(restored_median_xs[0], restored_median_ys[0], label = 'm')
            ax1.plot(wavelengthHa, (fluxHa.value - fluxHa_interpolated) * fluxHa.unit, label = 'l - m')
            ax1.plot(wavelengthHa, fluxHa_deblended, label = 'd')

            ax2.plot(wavelengthHb, fluxHb, label = 'l')
            ax2.plot(restored_median_xs[1], restored_median_ys[1], label = 'm')
            ax2.plot(wavelengthHb, (fluxHb.value - fluxHb_interpolated) * fluxHb.unit, label = 'l - m')
            ax2.plot(wavelengthHb, fluxHb_deblended, label = 'd')

            ax3.plot(wavelengthHg, fluxHg, label = 'l')
            ax3.plot(restored_median_xs[2], restored_median_ys[2], label = 'm')
            ax3.plot(wavelengthHg, (fluxHg.value - fluxHg_interpolated) * fluxHg.unit, label = 'l - m')
            ax3.plot(wavelengthHg, fluxHg_deblended, label = 'd')

            ax4.plot(wavelengthHd, fluxHd, label = 'l')
            ax4.plot(restored_median_xs[3], restored_median_ys[3], label = 'm')
            ax4.plot(wavelengthHd, (fluxHd.value - fluxHd_interpolated) * fluxHd.unit, label = 'l - m')
            ax4.plot(wavelengthHd, fluxHd_deblended, label = 'd')
            plt.legend()
            plt.savefig(output_path + filename + '.lines_deblending.png')
            plt.clf()

            # Recalculate the spectra substracting the deblended lines to the original (dereddem) spectra
            fluxHa_deblended_interpolated = np.interp(spec.wavelength, wavelengthHa, fluxHa_deblended)
            fluxHb_deblended_interpolated = np.interp(spec.wavelength, wavelengthHb, fluxHb_deblended)
            fluxHg_deblended_interpolated = np.interp(spec.wavelength, wavelengthHg, fluxHg_deblended)
            fluxHd_deblended_interpolated = np.interp(spec.wavelength, wavelengthHd, fluxHd_deblended)
            
            fluxHa_deblended_interpolated_spec = Spectrum1D(spectral_axis=spec.wavelength, flux=fluxHa_deblended_interpolated, meta=meta)
            fluxHb_deblended_interpolated_spec = Spectrum1D(spectral_axis=spec.wavelength, flux=fluxHb_deblended_interpolated, meta=meta)
            fluxHg_deblended_interpolated_spec = Spectrum1D(spectral_axis=spec.wavelength, flux=fluxHg_deblended_interpolated, meta=meta)
            fluxHd_deblended_interpolated_spec = Spectrum1D(spectral_axis=spec.wavelength, flux=fluxHd_deblended_interpolated, meta=meta)

            # Be sure we null all flux outside the lines
            fluxHa_deblended_interpolated_spec = reset_spectra_array_except_range(Halpha - haCalculations[4], Halpha + haCalculations[5], fluxHa_deblended_interpolated_spec)
            fluxHb_deblended_interpolated_spec = reset_spectra_array_except_range(Hbeta - hbCalculations[4], Hbeta + hbCalculations[5], fluxHb_deblended_interpolated_spec)
            fluxHg_deblended_interpolated_spec = reset_spectra_array_except_range(Hgamma - hgCalculations[4], Hgamma + hgCalculations[5], fluxHg_deblended_interpolated_spec)
            fluxHd_deblended_interpolated_spec = reset_spectra_array_except_range(Hdelta - hdCalculations[4], Hdelta + hdCalculations[5], fluxHd_deblended_interpolated_spec)
            
            fluxHa_deblended_interpolated = fluxHa_deblended_interpolated_spec.flux
            fluxHb_deblended_interpolated = fluxHb_deblended_interpolated_spec.flux
            fluxHg_deblended_interpolated = fluxHg_deblended_interpolated_spec.flux
            fluxHd_deblended_interpolated = fluxHd_deblended_interpolated_spec.flux

            # Generate separate lines plots for reference
            fig, ax = plt.subplots()
            fig.set_figwidth(10)
            fig.set_figheight(7)
            ax.plot(spec.wavelength, fluxHa_deblended_interpolated, label = HalphaLabel)
            ax.set(xlabel = 'Wavelenght', ylabel = "Flux")
            plt.savefig(output_path + filename + '.' + HalphaLabel.lower() + '_deblended.png')
            plt.clf()

            fig, ax = plt.subplots()
            fig.set_figwidth(10)
            fig.set_figheight(7)
            ax.plot(spec.wavelength, fluxHb_deblended_interpolated, label = HbetaLabel)
            ax.set(xlabel = 'Wavelenght', ylabel = "Flux")
            plt.savefig(output_path + filename + '.' + HbetaLabel.lower() + '_deblended.png')
            plt.clf()

            fig, ax = plt.subplots()
            fig.set_figwidth(10)
            fig.set_figheight(7)
            ax.plot(spec.wavelength, fluxHg_deblended_interpolated, label = HgammaLabel)
            ax.set(xlabel = 'Wavelenght', ylabel = "Flux")
            plt.savefig(output_path + filename + '.' + HgammaLabel.lower() + '_deblended.png')
            plt.clf()

            fig, ax = plt.subplots()
            fig.set_figwidth(10)
            fig.set_figheight(7)
            ax.plot(spec.wavelength, fluxHd_deblended_interpolated, label = HdeltaLabel)
            ax.set(xlabel = 'Wavelenght', ylabel = "Flux")
            plt.savefig(output_path + filename + '.' + HdeltaLabel.lower() + '_deblended.png')
            plt.clf()

            all_lines = fluxHa_deblended_interpolated + fluxHb_deblended_interpolated + fluxHg_deblended_interpolated + fluxHd_deblended_interpolated
            fig, ax = plt.subplots()
            fig.set_figwidth(10)
            fig.set_figheight(7)
            ax.plot(spec.wavelength, all_lines, label = 'All lines')
            ax.set(xlabel = 'Wavelenght', ylabel = "Flux")
            plt.savefig(output_path + filename + '.all_deblended_lines.png')
            plt.clf()

            # Finally save the processed spectra to file
            np.savetxt(output_path + '/processed/' + filename + '.processed.dat', np.column_stack((spec.wavelength.value, (spec.flux - all_lines).value)), fmt=['%.4f','%.6e'], delimiter=datSeparator)

            # Calculate flux of deblended lines
            spec_ha_deblended = Spectrum1D(spectral_axis=wavelengthHa, flux=fluxHa_deblended, meta=meta)
            pec_hb_deblended = Spectrum1D(spectral_axis=wavelengthHb, flux=fluxHb_deblended, meta=meta)
            pec_hg_deblended = Spectrum1D(spectral_axis=wavelengthHg, flux=fluxHg_deblended, meta=meta)
            pec_hd_deblended = Spectrum1D(spectral_axis=wavelengthHd, flux=fluxHd_deblended, meta=meta)

            haFluxDataDeblended = line_flux(spec_ha_deblended, regions = [SpectralRegion((Halpha - haCalculations[4]) * u.AA, (Halpha + haCalculations[5]) * u.AA )])
            hbFluxDataDeblended = line_flux(pec_hb_deblended, regions = [SpectralRegion((Hbeta - hbCalculations[4]) * u.AA, (Hbeta + hbCalculations[5]) * u.AA )])
            hgFluxDataDeblended = line_flux(pec_hg_deblended, regions = [SpectralRegion((Hgamma - hgCalculations[4]) * u.AA, (Hgamma + hgCalculations[5]) * u.AA )])
            hdFluxDataDeblended = line_flux(pec_hd_deblended, regions = [SpectralRegion((Hdelta - hdCalculations[4]) * u.AA, (Hdelta + hdCalculations[5]) * u.AA )])

            haValues = np.insert(haValues, 2, haFluxDataDeblended[0].value)
            hbValues = np.insert(hbValues, 2, hbFluxDataDeblended[0].value)
            hgValues = np.insert(hgValues, 2, hgFluxDataDeblended[0].value)
            hdValues = np.insert(hdValues, 2, hdFluxDataDeblended[0].value)
            
            # Write report
            lines = np.array([haValues, hbValues, hgValues, hdValues])
            report.write('Lines analisys' + '\n')
            report.write(tabulate(lines, headers=['Line','Flux','Deblended Flux', 'FWHM', 'Equivalent width', 'Centroid']) + '\n')
            report.write('* Units: ' + str(fluxData[0].unit))
            report.write('\n')

            # Write spreadsheet
            csv.write(str(centroidData[0].value) + ';' + str(fluxData[0].value) + ';' + str(haFluxDataDeblended[0].value) + ';' + str(equivalentWidthData[0].value) + ';' + str(fwhmData[0].value) + ';')
            csv.write(str(centroidData[1].value) + ';' + str(fluxData[1].value) + ';' + str(hbFluxDataDeblended[0].value) + ';' + str(equivalentWidthData[1].value) + ';' + str(fwhmData[1].value) + ';')
            csv.write(str(centroidData[2].value) + ';' + str(fluxData[2].value) + ';' + str(hgFluxDataDeblended[0].value) + ';' + str(equivalentWidthData[2].value) + ';' + str(fwhmData[2].value) + ';')
            csv.write(str(centroidData[3].value) + ';' + str(fluxData[3].value) + ';' + str(hdFluxDataDeblended[0].value) + ';' + str(equivalentWidthData[3].value) + ';' + str(fwhmData[3].value))
            csv.write('\n')

            # Close report
            report.close()

            counter = counter + 1

            print('Completed ' + filename + ' at ' + datetime.now().strftime('%H:%M:%S'))
            
            if (onlyOne):
                break # Just as test to only process the first spectrum of the folder
            
        if (counter > 0):
            # Only generate evoluti0on graphs if more than one spectra analysed
            fig, ax = plt.subplots()
            fig.set_figwidth(10)
            fig.set_figheight(7)
            ax.plot(evolutionPlane, HalphaEvolution, label = HalphaLabel)
            ax.plot(evolutionPlane, HbetaEvolution, label = HbetaLabel)
            ax.plot(evolutionPlane, HgammaEvolution, label = HgammaLabel)
            ax.plot(evolutionPlane, HdeltaEvolution, label = HdeltaLabel)
            ax.set(xlabel = EvolutionLabel, ylabel = f"Flux ({(u.erg / u.Angstrom / u.s / u.cm / u.cm).to_string('latex_inline')})")
            ax.set_yscale('log')
            if (evolutionPlaneLog):
                ax.set_xscale('log')
            else:
                fig.autofmt_xdate()
            plt.legend()
            plt.savefig(output_path + 'lines_flux_evolution.png')
            plt.clf()

            if (Halpha == HALPHA_REF and Hbeta == HBETA_REF and Hgamma == HGAMMA_REF and Hdelta == HDELTA_REF):
                # Only generate this graph if measuring the main H lines
                fig, ax = plt.subplots()
                fig.set_figwidth(10)
                fig.set_figheight(7)
                ax.plot(evolutionPlane, Halpha_Hbeta, label = HalphaLabel + '/' + HbetaLabel)
                ax.plot(evolutionPlane, Hgamma_Hbeta, label = HgammaLabel + '/' + HbetaLabel)
                ax.plot(evolutionPlane, Hdelta_Hbeta, label = HdeltaLabel + '/' + HbetaLabel)
                ax.set(xlabel = EvolutionLabel, ylabel = 'Line ratio')
                ax.set_yscale('log')
                if (evolutionPlaneLog):
                    ax.set_xscale('log')
                else:
                    fig.autofmt_xdate()
                plt.legend()
                plt.savefig(output_path + 'lines_ratio_evolution.png')
                plt.clf()

            fig, ax = plt.subplots()
            fig.set_figwidth(10)
            fig.set_figheight(7)
            ax.plot(evolutionPlane, HalphaFWHMEvolution, label = HalphaLabel)
            ax.plot(evolutionPlane, HbetaFWHMEvolution, label = HbetaLabel)
            ax.plot(evolutionPlane, HgammaFWHMEvolution, label = HgammaLabel)
            ax.plot(evolutionPlane, HdeltaFWHMEvolution, label = HdeltaLabel)
            ax.set(xlabel = EvolutionLabel, ylabel = f"FWHM ({(u.kilometer / u.second)})")
            ax.set_yscale('log')
            if (evolutionPlaneLog):
                ax.set_xscale('log')
            else:
                fig.autofmt_xdate()
            plt.legend()
            plt.savefig(output_path + 'lines_fwhm_evolution.png')
            plt.clf()

        csv.close()

        endTime = datetime.now()
        print('Completed at ' + datetime.now().strftime('%H:%M:%S'))
        print('The execution took ' + str(round((endTime - startTime).total_seconds(),0)) + ' seconds')
        
if __name__ == '__main__':
   main(sys.argv[1:])
