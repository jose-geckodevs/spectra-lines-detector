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
from specutils.analysis import centroid, fwhm, line_flux, equivalent_width
from dust_extinction.averages import *
from dust_extinction.parameter_averages import *
from dust_extinction.grain_models import *
from dust_extinction.shapes import *
import matplotlib.pyplot as plt
import numpy as np
import copy, os, sys, getopt, warnings
from tabulate import tabulate
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
    return _spectrum.flux.value[_foundLowerIndex:_foundUpperIndex] * _spectrum.flux.unit, _spectrum.wavelength.value[_foundLowerIndex:_foundUpperIndex] * _spectrum.wavelength.unit

def reduce_sorted_fits_array_by_filename(item: list):
    return item[0]

def reduce_sorted_fits_array_by_date(item: list):
    return item[1]

def measure_lines_fixed(_center: float, _spec_norm: Spectrum1D):
    _padding = 50
    _regions = [SpectralRegion((_center - _padding) * u.AA, (_center + _padding) * u.AA )]
    _fluxData = line_flux(_spec_norm, regions = _regions)
    _fwhmData = fwhm(_spec_norm, regions = _regions)
    _equivalentWidthData = equivalent_width(_spec_norm, continuum=1, regions = _regions)
    _centroidData = centroid(_spec_norm, regions = _regions)

    return _fluxData, _fwhmData, _equivalentWidthData, _centroidData

def measure_line_max_fwhm(_center: float, _spec_norm: Spectrum1D, _spec_flux: Spectrum1D):
    _padding = 5
    _precision = 2
    _previousFwhm = u.Quantity(0)
    _regions = []
    while(_padding < 100):
        _regions = [SpectralRegion((_center - _padding) * u.AA, (_center + _padding) * u.AA )]
        _fwhmData = fwhm(_spec_norm, regions = _regions)
        if (round(_fwhmData[0].value, _precision) <= round(_previousFwhm.value, _precision)):
            break
        _previousFwhm = _fwhmData[0]
        _padding += 5

    _fluxData = line_flux(_spec_flux, regions = _regions)
    _equivalentWidthData = equivalent_width(_spec_norm, continuum=1, regions = _regions)
    _centroidData = centroid(_spec_norm, regions = _regions)

    return _fluxData[0], _previousFwhm, _equivalentWidthData[0], _centroidData[0]

def print_help():
    print('display_fits_spectra_advance.py')
    print('         --debug')
    print('         --only-one')
    print('         --path <include path for spectra>')
    print('         --ebv <Ebv dust extintion value>')
    print('         --rv <Rv dust extintion value>')
    print('         --model <dust extintion model>')
    print('         --l[1,2,3,4]centroid <line angstrom centroid> --l[1,2,3,4]label <line label>')
    print('         --wavelenghtLowerLimit <lower angstrom limit> --wavelenghtUpperLimit <higher angstrom limit>')
    print('If no wavelenght limtis configured, 4000 to 7000 Angstrom will be used')
    print('If no lines configured, Halpha(4), Hbeta(3), Hgamma(2) and Hdelta(1) will be used')

def main(argv):
    quantity_support() 
    plt.style.use(astropy_mpl_style)
    
    path = './'
    debug = False
    onlyOne = False
    
    Halpha = 6563
    HalphaLabel = 'Halpha'
    Hbeta = 4861
    HbetaLabel = 'Hbeta'
    Hgamma = 4341
    HgammaLabel = 'Hgamma'
    Hdelta = 4102
    HdeltaLabel = 'Hdelta'
    WavelenghtLowerLimit = 4000
    WavelenghtUpperLimit = 7000
    Ebv = 0
    Rv = 3.1
    Model = F99
    inputParams = ''

    try:
        opts, args = getopt.getopt(argv,'hp:d',['help','path=','debug','only-one','ebv=','rv=','model=',
                                                'wavelenghtLowerLimit=','wavelenghtUpperLimit=',
                                                'l1centroid=','l2centroid=','l3centroid=','l4centroid=',
                                                'l1label=','l2label=','l3label=','l4label='])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    for opt, arg in opts:
        if opt not in ('-p', '--path'):
            inputParams += opt + arg

        if opt in ('-h', '--help'):
            print_help()
            sys.exit()
        elif opt in ('-p', '--path'):
            path = arg + '/'
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
        elif opt in ('--wavelenghtLowerLimit'):
            WavelenghtLowerLimit = int(arg)
        elif opt in ('--wavelenghtUpperLimit'):
            WavelenghtUpperLimit = int(arg)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        Halpha_Hbeta = []
        Hgamma_Hbeta = []
        Hdelta_Hbeta = []
        evolutionPlane = []
        count = 0
        
        startTime = datetime.now()
        print('Start running at ' + startTime.strftime('%H:%M:%S'))

        # Sort FITs by date in header
        listFITS = []
        for filename in os.listdir(path):
            if not filename.endswith('.fits'):
                continue
            spec = Spectrum1D.read(path + filename, format='wcs1d-fits')
            listFITS.append([filename, spec.meta['header']['DATE-OBS']])
        
        if (len(listFITS) <= 0):
            print('No FITs found on folder ' + path)
            sys.exit()
        
        sortedFITS = list(map(reduce_sorted_fits_array_by_filename, sorted(listFITS)))
        sortedFDates = list(map(reduce_sorted_fits_array_by_date, sorted(listFITS)))

        csv = open(path + 'lines_measurements' + inputParams + '.csv', 'w')
        csv.write('Spectra file;')
        csv.write(HalphaLabel + ' centroid;' + HalphaLabel + ' flux;' + HalphaLabel + ' eqw;' + HalphaLabel + ' fwhm;')
        csv.write(HbetaLabel + ' centroid;' + HbetaLabel + ' flux;' + HbetaLabel + ' eqw;' + HbetaLabel + ' fwhm;')
        csv.write(HgammaLabel + ' centroid;' + HgammaLabel + ' flux;' + HgammaLabel + ' eqw;' + HgammaLabel + ' fwhm;')
        csv.write(HdeltaLabel + ' centroid;' + HdeltaLabel + ' flux;' + HdeltaLabel + ' eqw;' + HdeltaLabel + ' fwhm')
        csv.write('\n')

        counter = 0
        for filename in sortedFITS:
            if not filename.endswith('.fits'):
                continue
            
            report = open(path + filename + '.txt', 'w')
            report.write('Filename: ' + filename + '\n')

            csv.write(filename + ';')

            # Read FITS
            hdul = fits.open(path + filename, mode='readonly', memmap = True)
            if debug:
                report.write(tabulate(hdul.info(False)) + '\n')
                report.write('\n')
            
            # Read the spectrum
            redshift = None # set scalar value
            spec_original = Spectrum1D.read(path + filename, format='wcs1d-fits', redshift=redshift)
            
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
            
            # Try find the continuum without the lines
            noise_region = SpectralRegion(WavelenghtLowerLimit * u.AA, WavelenghtUpperLimit * u.AA) # u.AA for Angstrom
            spec_noise = noise_region_uncertainty(spec, noise_region)
            lines = find_lines_threshold(spec_noise, noise_factor=1)
            numLinesFirstIteration = len(lines)
            
            if debug:
                # Try identify Balmer series
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
            
            # Find now lines by thresholding using the normalised spectrum
            noise_region = SpectralRegion(WavelenghtLowerLimit * u.AA, WavelenghtUpperLimit * u.AA) # u.AA for Angstrom
            spec_noise = noise_region_uncertainty(spec_normalized, noise_region)
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
                
                # Find now lines by thresholding using the normalised spectrum
                noise_region = SpectralRegion(WavelenghtLowerLimit * u.AA, WavelenghtUpperLimit * u.AA) # u.AA for Angstrom
                spec_noise = noise_region_uncertainty(spec_normalized, noise_region)
                lines = find_lines_threshold(spec_noise, noise_factor=1)
                
            # Try identify Balmer series
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
            
            ax1.plot(spec_normalized.spectral_axis, spec_normalized.flux)
            ax2.plot(spec_normalized.spectral_axis, spec_normalized.flux)
            ax3.plot(spec_normalized.spectral_axis, spec_normalized.flux)
            ax4.plot(spec_normalized.spectral_axis, spec_normalized.flux)
            
            if debug:
                report.write('Found lines (noise region uncertainty factor 1):' + '\n')
                report.write(tabulate(lines, headers=['Line center','Type','Index','Match']) + '\n')
                report.write('\n')

            # Find lines by derivating
            lines = find_lines_derivative(spec_normalized, flux_threshold=0.95)
            
            # Try identify Balmer series
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

            # Measure lines finding paddign from amx fwhm
            haCalculations = measure_line_max_fwhm(Halpha, spec_normalized, spec_flux)
            hbCalculations = measure_line_max_fwhm(Hbeta, spec_normalized, spec_flux)
            hgCalculations = measure_line_max_fwhm(Hgamma, spec_normalized, spec_flux)
            hdCalculations = measure_line_max_fwhm(Hdelta, spec_normalized, spec_flux)
            fluxData = [haCalculations[0], hbCalculations[0], hgCalculations[0], hdCalculations[0]]
            fwhmData = [haCalculations[1], hbCalculations[1], hgCalculations[1], hdCalculations[1]]
            equivalentWidthData = [haCalculations[2], hbCalculations[2], hgCalculations[2], hdCalculations[2]]
            centroidData = [haCalculations[3], hbCalculations[3], hgCalculations[3], hdCalculations[3]]

            haValues = np.array([HalphaLabel, fluxData[0], fwhmData[0], equivalentWidthData[0], centroidData[0]])
            hbValues = np.array([HbetaLabel, fluxData[1], fwhmData[1], equivalentWidthData[1], centroidData[1]])
            hgValues = np.array([HgammaLabel, fluxData[2], fwhmData[2], equivalentWidthData[2], centroidData[2]])
            hdValues = np.array([HdeltaLabel, fluxData[3], fwhmData[3], equivalentWidthData[3], centroidData[3]])
            
            lines = np.array([haValues, hbValues, hgValues, hdValues])
            report.write('Lines analisys' + '\n')
            report.write(tabulate(lines, headers=['Line','Flux','FWHM', 'Equivalent width', 'Centroid']) + '\n')
            report.write('* Units: ' + str(fluxData[0].unit))
            report.write('\n')

            csv.write(str(centroidData[0].value) + ';' + str(fluxData[0].value) + ';' + str(equivalentWidthData[0].value) + ';' + str(fwhmData[0].value) + ';')
            csv.write(str(centroidData[1].value) + ';' + str(fluxData[1].value) + ';' + str(equivalentWidthData[1].value) + ';' + str(fwhmData[1].value) + ';')
            csv.write(str(centroidData[2].value) + ';' + str(fluxData[2].value) + ';' + str(equivalentWidthData[2].value) + ';' + str(fwhmData[2].value) + ';')
            csv.write(str(centroidData[3].value) + ';' + str(fluxData[3].value) + ';' + str(equivalentWidthData[3].value) + ';' + str(fwhmData[3].value))
            csv.write('\n')
            
            # Calculate lines evolution
            Halpha_Hbeta.append(fluxData[0] / fluxData[1])
            Hgamma_Hbeta.append(fluxData[2] / fluxData[1])
            Hdelta_Hbeta.append(fluxData[3] / fluxData[1])
            evolutionPlane.append(sortedFDates[count])
            count += 1
            
            # Plot figure and close
            plt.savefig(path + filename + '.png')
            plt.clf()
            hdul.close()
            report.close()

            counter = counter + 1

            print('Completed ' + filename + ' at ' + datetime.now().strftime('%H:%M:%S'))
            
            if (onlyOne):
                break # Just as test to only process the first spectrum of the folder
            
        if (counter > 1):
            fig, ax = plt.subplots()
            ax.plot(evolutionPlane, Halpha_Hbeta, label = HalphaLabel + '/' + HbetaLabel)
            ax.plot(evolutionPlane, Hgamma_Hbeta, label = HgammaLabel + '/' + HbetaLabel)
            ax.plot(evolutionPlane, Hdelta_Hbeta, label = HdeltaLabel + '/' + HbetaLabel)
            ax.set(xlabel = 'Date', ylabel = 'Flux ' + HbetaLabel + ' factor')
            fig.autofmt_xdate()
            plt.legend()
            plt.savefig(path + 'lines_evolution' + inputParams + '.png')
            plt.clf()

        csv.close()

        endTime = datetime.now()
        print('Completed at ' + datetime.now().strftime('%H:%M:%S'))
        print('The execution took ' + str(round((endTime - startTime).total_seconds(),0)) + ' seconds')
        
if __name__ == '__main__':
   main(sys.argv[1:])
