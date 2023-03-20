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
from dust_extinction.parameter_averages import F99
import matplotlib.pyplot as plt
import numpy as np
import copy, os, sys, getopt, warnings
from tabulate import tabulate
from datetime import datetime

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
Ebv = -0.6

def limitSpectraArray(_low: int, _up: int, _spectrum: Spectrum1D):
    _foundLowerIndex = 0
    _foundUpperIndex = 0
    for index, wavelength in enumerate(_spectrum.wavelength):
        if wavelength.value >= _low and _spectrum.flux.value[index] != 0 and not _foundLowerIndex:
            _foundLowerIndex = index
        if wavelength.value >= _up and not _foundUpperIndex:
            _foundUpperIndex = index
            break
    return _spectrum.flux.value[_foundLowerIndex:_foundUpperIndex] * _spectrum.flux.unit, _spectrum.wavelength.value[_foundLowerIndex:_foundUpperIndex] * _spectrum.wavelength.unit

def reduceSortedFITSArrayByFilename(item: list):
    return item[0]

def reduceSortedFITSArrayByDate(item: list):
    return item[1]

def measureLinesFixed(_center: float, _spec_norm: Spectrum1D):
    _padding = 50
    _regions = [SpectralRegion((_center - _padding) * u.AA, (_center + _padding) * u.AA )]
    _fluxData = line_flux(_spec_norm, regions = _regions)
    _fwhmData = fwhm(_spec_norm, regions = _regions)
    _equivalentWidthData = equivalent_width(_spec_norm, continuum=1, regions = _regions)
    _centroidData = centroid(_spec_norm, regions = _regions)

    return _fluxData, _fwhmData, _equivalentWidthData, _centroidData

def measureLineMaxFwhm(_center: float, _spec_norm: Spectrum1D, _spec_flux: Spectrum1D):
    _padding = 5
    _precision = 2
    _previousFwhm = u.Quantity(0)
    _regions = []
    while(padding < 100):
        _regions = [SpectralRegion((_center - _padding) * u.AA, (_center + _padding) * u.AA )]
        fwhmData = fwhm(_spec_norm, regions = _regions)
        if (round(fwhmData[0].value, _precision) <= round(_previousFwhm.value, _precision)):
            break
        _previousFwhm = fwhmData[0]
        padding += 5

    _fluxData = line_flux(_spec_flux, regions = _regions)
    _equivalentWidthData = equivalent_width(_spec_norm, continuum=1, regions = _regions)
    _centroidData = centroid(_spec_norm, regions = _regions)

    return _fluxData[0], _previousFwhm, _equivalentWidthData[0], _centroidData[0]

def main(argv):
    quantity_support() 
    plt.style.use(astropy_mpl_style)
    
    path = './'
    debug = False
    
    try:
        opts, args = getopt.getopt(argv,'hp:d:',['path=','debug'])
    except getopt.GetoptError:
        print('display_fits_spectra_advance.py -p <include path for spectra> -d <debug mode true or false>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('display_fits_spectra_advance.py -p <include path for spectra> -d <debug mode true or false>')
            sys.exit()
        elif opt in ('-p', '--path'):
            path = arg + '/'
        elif opt in ('-d', '--debug'):
            debug = arg == 'true'
    
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
        
        sortedFITS = list(map(reduceSortedFITSArrayByFilename, sorted(listFITS)))
        sortedFDates = list(map(reduceSortedFITSArrayByDate, sorted(listFITS)))

        for filename in sortedFITS:
            if not filename.endswith('.fits'):
                continue
            
            report = open(path + filename + '.txt', 'w')
            
            report.write('Filename: ' + filename + '\n')

            # Read FITS
            hdul = fits.open(path + filename, mode='readonly', memmap = True)
            if debug:
                report.write(tabulate(hdul.info(False)) + '\n')
                report.write('\n')
            
            # Read the spectrum
            redshift = None # set scalar value
            spec = Spectrum1D.read(path + filename, format='wcs1d-fits', redshift=redshift)
            
            if debug:
                report.write(repr(spec.meta['header']) + '\n')
                report.write('\n')

            report.write('Date observation: ' + spec.meta['header']['DATE-OBS'] + '\n')
            report.write('Exposure time: ' + str(spec.meta['header']['EXPTIME']) + '\n')
            report.write('Telescope: ' + spec.meta['header']['TELESCOP'] + '\n')
            report.write('Instrument: ' + spec.meta['header']['INSTRUME'] + '\n')
            report.write('Object: ' + spec.meta['header']['OBJECT'] + '\n')

            # Limit the spectrum between the lower and upper range
            flux, wavelength = limitSpectraArray(WavelenghtLowerLimit, WavelenghtUpperLimit, spec)
            
            if debug:
                report.write('Flux and wavelength spectra:' + '\n')
                report.write(repr(flux) + '\n')
                report.write(repr(wavelength) + '\n')
                report.write('\n')
            
            # Make a copy of the spectrum object with the new flux and wavelenght arrays
            meta = copy.copy(spec.meta)
            # Not sure if we need to modify the header, but we do just in case
            meta['header']['NAXIS1'] = len(wavelength)
            meta['header']['CRVAL1'] = wavelength.value[0]
            spec = Spectrum1D(spectral_axis=wavelength, flux=flux, meta=meta)

            # Extinguish (redden) the spectrum
            ext = F99(Rv=3.1)
            flux_ext = spec.flux * ext.extinguish(spec.spectral_axis, Ebv=Ebv)
            spec = Spectrum1D(spectral_axis=spec.wavelength, flux=flux_ext, meta=spec.meta)
            
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
            
            # Plot initial spectrum
            ax5.plot(wavelength, flux)
            
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
                    fluxContinuumRegions.append(flux[0].value)
                    wavelengthContinuumRegions.append(WavelenghtLowerLimit)
                    fluxContinuumRegions.append(flux[0].value)
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
                fluxContinuumRegions.append(0)
                wavelengthContinuumRegions.append(previousLine + padding)
                fluxContinuumRegions.append(flux[0].value)
                wavelengthContinuumRegions.append(previousLine + padding)
                fluxContinuumRegions.append(flux[0].value)
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
            ax5.plot(wavelengthContinuumRegions, fluxContinuumRegions);
            
            # If no lines found, be sure we add the whole spectrum
            if (len(includeRegions) <= 0):
                includeRegions.append((WavelenghtLowerLimit, WavelenghtUpperLimit) * u.AA)        
                
            ax6.set_ylabel('Continuum')
            #g1_fit = fit_generic_continuum(spec, exclude_regions=SpectralRegion(excludeRegions))
            #g1_fit = fit_continuum(spec, window=includeRegions)
            g1_fit = fit_continuum(spec, exclude_regions=SpectralRegion(excludeRegions))
            #g1_fit = fit_continuum(spec, window=includeRegions, exclude_regions=SpectralRegion(excludeRegions))
            y_continuum_fitted = g1_fit(wavelength)
            ax6.plot(wavelength, flux);
            ax6.plot(wavelength, y_continuum_fitted);
            
            ax7.set_ylabel('Normalised')
            spec_normalized = spec / y_continuum_fitted
            spec_flux = spec - y_continuum_fitted
            ax7.plot(spec_normalized.spectral_axis, spec_normalized.flux);
            
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
                        fluxContinuumRegions.append(flux[0].value)
                        wavelengthContinuumRegions.append(WavelenghtLowerLimit)
                        fluxContinuumRegions.append(flux[0].value)
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
                    fluxContinuumRegions.append(0)
                    wavelengthContinuumRegions.append(previousLine + padding)
                    fluxContinuumRegions.append(flux[0].value)
                    wavelengthContinuumRegions.append(previousLine + padding)
                    fluxContinuumRegions.append(flux[0].value)
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
                ax5.plot(wavelengthContinuumRegions, fluxContinuumRegions);
                
                # If no lines found, be sure we add the whole spectrum
                if (len(includeRegions) <= 0):
                    includeRegions.append((WavelenghtLowerLimit, WavelenghtUpperLimit) * u.AA)        
                    
                #g1_fit = fit_generic_continuum(spec, exclude_regions=SpectralRegion(excludeRegions))
                #g1_fit = fit_continuum(spec, window=includeRegions)
                g1_fit = fit_continuum(spec, exclude_regions=SpectralRegion(excludeRegions))
                #g1_fit = fit_continuum(spec, window=includeRegions, exclude_regions=SpectralRegion(excludeRegions))
                y_continuum_fitted = g1_fit(wavelength)
                ax6.plot(wavelength, y_continuum_fitted);
                
                spec_normalized = spec / y_continuum_fitted
                spec_flux = spec - y_continuum_fitted
                ax7.clear()
                ax7.set_ylabel('Normalised')
                ax7.plot(spec_normalized.spectral_axis, spec_normalized.flux);
                
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
            haCalculations = measureLineMaxFwhm(Halpha, spec_normalized, spec_flux)
            hbCalculations = measureLineMaxFwhm(Hbeta, spec_normalized, spec_flux)
            hgCalculations = measureLineMaxFwhm(Hgamma, spec_normalized, spec_flux)
            hdCalculations = measureLineMaxFwhm(Hdelta, spec_normalized, spec_flux)
            fluxData = [haCalculations[0], hbCalculations[0], hgCalculations[0], hdCalculations[0]]
            fwhmData = [haCalculations[1], hbCalculations[1], hgCalculations[1], hdCalculations[1]]
            equivalentWidthData = [haCalculations[2], hbCalculations[2], hgCalculations[2], hdCalculations[2]]
            centroidData = [haCalculations[3], hbCalculations[3], hgCalculations[3], hdCalculations[3]]

            haValues = np.array(['Halpha', fluxData[0], fwhmData[0], equivalentWidthData[0], centroidData[0]])
            hbValues = np.array(['Hbeta', fluxData[1], fwhmData[1], equivalentWidthData[1], centroidData[1]])
            hgValues = np.array(['Hgamma', fluxData[2], fwhmData[2], equivalentWidthData[2], centroidData[2]])
            hdValues = np.array(['Hdelta', fluxData[3], fwhmData[3], equivalentWidthData[3], centroidData[3]])
            
            lines = np.array([haValues, hbValues, hgValues, hdValues])
            report.write('Lines analisys' + '\n')
            report.write(tabulate(lines, headers=['Line','Flux','FWHM', 'Equivalent width', 'Centroid']) + '\n')
            report.write('* Units: ' + str(fluxData[0].unit))
            report.write('\n')
            
            # Calculate lines evolution
            Halpha_Hbeta.append(fluxData[0] / fluxData[1])
            Hgamma_Hbeta.append(fluxData[2] / fluxData[1])
            Hdelta_Hbeta.append(fluxData[3] / fluxData[1])
            evolutionPlane.append(sortedFDates[count])
            count += 1
            
            # Plot figure
            plt.savefig(path + filename + '.png')
            plt.clf()
            hdul.close()
            
            print('Completed ' + filename + ' at ' + datetime.now().strftime('%H:%M:%S'))
            break # Just a test to only process the first spectrum of the folder
            
        fig, ax = plt.subplots()
        ax.plot(evolutionPlane, Halpha_Hbeta, label = 'Halpha/Hbeta')
        ax.plot(evolutionPlane, Hgamma_Hbeta, label = 'Hgamma/Hbeta')
        ax.plot(evolutionPlane, Hdelta_Hbeta, label = 'Hdelta/Hbeta')
        ax.set(xlabel = 'Date', ylabel = 'Flux Hbeta factor')
        fig.autofmt_xdate()
        plt.legend()
        plt.savefig(path + 'lines-evolution.png')
        plt.clf()

        endTime = datetime.now()
        print('Completed at ' + datetime.now().strftime('%H:%M:%S'))
        print('The execution took ' + str(round((endTime - startTime).total_seconds(),0)) + ' seconds')
        
if __name__ == '__main__':
   main(sys.argv[1:])
