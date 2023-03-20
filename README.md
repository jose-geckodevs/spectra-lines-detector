# FITS spectra analyser
Python script to analyse a set of FITS spectra novae, extract absortion and emission lines, remove continuum, measure H lines (alpha, beta, gamma and delta), and plot lines Hbeta factor evolution in time.

## Requires
pyhton 3.x

### Dependencies
Install with:
`pip install astropy`
`pip install specutils`
`pip install matplotlib`
`pip install tabulate`
`pip install dust_extinction`

## Script
Run with:
`python3 display_fits_spectra_advance.py -p ./V5114Sgr`

Adding debug flag:
`python3 display_fits_spectra_advance.py -d true -p ./V5114Sgr`

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
