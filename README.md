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
