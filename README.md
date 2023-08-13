# FITS spectra analyser
Python script to analyse a set of FITS spectra novae, extract absortion and emission lines, remove continuum, and measure 4 lines: Hdelta(1), Hgamma(2), Hbeta(3) and Halpha(4) by default.
Wavelenght limited from 4000 to 7000 Angstrom by default.
Dust extintion removal (redden).
Plot graphs and texts report for each spectra.
Plot line graph Hbeta(3) factor evolution in time against the other lines. 
Single csv report with flux, fwhm, equivalent width and centroid.

## Requires
pyhton 3.x

### Dependencies
Install with:
`pip install astropy`
`pip install specutils`
`pip install matplotlib`
`pip install tabulate`
`pip install dust_extinction`
`pip install pandas`

## Script
Run with:
`python3 display_fits_spectra_advance.py --path ./V5114Sgr/fits`
`python3 display_fits_spectra_advance.py --datPath /V5114Sgr/dat --datSeparator "  "`

Adding debug flag:
`python3 display_fits_spectra_advance.py --debug --path ./V5114Sgr/fits`

Only one flag:
`python3 display_fits_spectra_advance.py --only-one --path ./V5114Sgr/fits`

Dust extintion:
`python3 display_fits_spectra_advance.py --path ./V5114Sgr/fits --ebv 0.6 --rv 3.2 --model CCM89`
`python3 display_fits_spectra_advance.py --datPath ./V5114Sgr/dat --datSeparator "  " --ebv 0.6 --rv 3.2 --model CCM89`

Wavelenght limits
`--wavelenghtLowerLimit 4000`
`--wavelenghtUpperLimit 7000`

Advanced parameters:
`--l1centroid 4102 --l1label Hdelta`
`--l2centroid 4341 --l2label Hgamma`
`--l3centroid 4861 --l3label Hbeta`
`--l4centroid 6563 --l4label Halpha`
`--angstromIncrement 5`
`--histrogramStDevPercent 0.5`
`--folderSuffix test001`

`python3 display_fits_spectra_advance.py --path ./V5114Sgr/fits --angstromIncrement 1 --histogramStDevPercent 0.3 --folderSuffix -angstrom1-std0.3`

`python3 display_fits_spectra_advance.py --path ./V5114Sgr/dat  --datSeparator "  " --ebv 0.6 --rv 3.2 --model CCM89 --angstromIncrement 1 --histogramStDevPercent 0.3 --folderSuffix -ebc0.6-rv3.2-angstrom1-std0.3`

`python3 display_fits_spectra_advance.py --path ./V5114Sgr/dat-days  --datSeparator "  " --ebv 0.6 --rv 3.2 --model CCM89 --angstromIncrement 1 --histogramStDevPercent 0.3 --folderSuffix -ebc0.6-rv3.2-angstrom1-std0.3`

## OSX
Download pip:
`curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`

Add pip path:
`echo export PATH=~/Library/Python/3.9/bin:$PATH >> ~/.zshrc`

Upgrade pip:
`python3 -m pip install --upgrade pip`


## IRAF
name_of_package	The package is loaded; prompt changes to the package name.
?	List tasks in the most recently-loaded package.
??	List all tasks loaded, regardless of package.
package	List all packages loaded.
? images	List tasks or subpackages in package images.
bye	Exit the current package.

Add to the file ~/.Xresources (or create):
XTerm*selectToClipboard: true
xgterm*VT100.Translations: #override \
      Shift <KeyPress> Insert: insert-selection(CLIPBOARD) \n\
      Ctrl Shift <Key>V:    insert-selection(CLIPBOARD) \n\
      Ctrl Shift <Key>C:    copy-selection(CLIPBOARD) \n\
      Ctrl <Btn1Up>: exec-formatted("xdg-open '%t'", PRIMARY)

Then run the command:
xrdb -merge ~/.Xresources

xgterm &
ecl
noao > onedspec
splot v5114sgr.a.18mar04.fits
splot v5114sgr.b.19mar04.fits
splot v5114sgr.c.26mar04.fits
splot v5114sgr.d.9apr04.fits
splot v5114sgr.e.18apr04.fits
splot v5114sgr.f.13may04.fits
splot v5114sgr.g.26jun04.fits
splot v5114sgr.h.23sep04.fits



splot objeto_b_f_wave.0001
‘t’: fit a function
‘/’: normalize by the fit
‘s’+’s’ para crear regiones de ajuste ‘z’ para borrar una región
‘f’ ajustar (actualizar ajuste cada vez
cambiamos parámetros) ‘q’ terminar
‘i’ guardar imagen

a: expand and autoscale, also w and z for each axis
b: set plot base level to zero
c: clear all window and redraw
d: mark two continuum points and fit line profiles. Flux, equivalent width fwhm... are saved in the log.
e: measure equivalent width by marking two continuum points around the line
k + (g, l or v): mark two continuum points and fit a single line profile.
r: redraw
z: zoom x2 graph in x
,: shift graph to the left
.: snhift graph to the right
:log
