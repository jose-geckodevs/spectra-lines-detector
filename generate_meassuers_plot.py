import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

# Helper script to generate the comparision values meassured

evolutionPlane = [1, 2, 9, 23, 32, 57, 101, 150]
HalphaEvolution = [1.69E-10, 1.59E-09, 1.15E-09, 8.40E-10, 1.46E-09, 1.08E-10, 4.54E-11, 2.43E-12]
HbetaEvolution = [8.40E-11, 6.07E-10, 2.36E-10, 9.04E-11, 1.81E-10, 1.14E-11, 9.03E-12, 5.85E-13]
HgammaEvolution = [4.47E-11, 3.04E-10, 7.64E-11, 3.92E-11, 8.48E-11, 5.39E-12, 2.52E-11, 1.83E-12]
HdeltaEvolution = [2.50E-11, 2.43E-10, 3.43E-11, 5.72E-11, 5.71E-11, 4.37E-12, 5.41E-12, 3.89E-13]

HalphaIRAF = [1.74E-10, 1.70E-09, 1.26E-09, 9.11E-10, 1.55E-09, 1.14E-10, 4.76E-11, 2.72E-12]
HbetaIRAF = [9.08E-11, 6.32E-10, 2.34E-10, 9.69E-11, 1.97E-10, 1.27E-11, 1.11E-11, 7.84E-13]
HgammaIRAF = [5.27E-11, 2.81E-10, 1.02E-10, 5.70E-11, 1.08E-10, 1.08E-11, 2.92E-11, 2.04E-12]
HdeltaIRAF = [3.93E-11, 2.81E-10, 7.80E-11, 7.29E-11, 8.05E-11, 6.47E-12, 7.52E-12, 6.12E-13]

yerr_lower = [1.67E-10, 1.66E-09, 1.25E-09, 8.75E-10, 1.52E-09, 1.10E-10, 4.61E-11, 2.68E-12]
yerr_upper = [1.82E-10, 1.75E-09, 1.26E-09, 9.33E-10, 1.59E-09, 1.17E-10, 4.89E-11, 2.75E-12]
for index, item  in enumerate(HalphaIRAF):
    yerr_lower[index] = HalphaIRAF[index] - yerr_lower[index]
    yerr_upper[index] = yerr_upper[index] - HalphaIRAF[index]
errorsHalphaIRAF = [yerr_lower, yerr_upper]

yerr_lower = [7.28E-11, 6.12E-10, 2.12E-10, 9.51E-11, 1.91E-10, 1.23E-11, 1.06E-11, 7.38E-13]
yerr_upper = [1.07E-10, 7.06E-10, 2.57E-10, 1.01E-10, 2.00E-10, 1.32E-11, 1.15E-11, 8.16E-13]
for index, item  in enumerate(HbetaIRAF):
    yerr_lower[index] = HbetaIRAF[index] - yerr_lower[index]
    yerr_upper[index] = yerr_upper[index] - HbetaIRAF[index]
errorsHbetaIRAF = [yerr_lower, yerr_upper]

yerr_lower = [4.57E-11, 2.47E-10, 9.73E-11, 5.56E-11, 1.05E-10, 1.06E-11, 2.89E-11, 1.75E-12]
yerr_upper = [5.77E-11, 3.21E-10, 1.06E-10, 6.06E-11, 1.12E-10, 1.11E-11, 2.94E-11, 2.21E-12]
for index, item  in enumerate(HgammaIRAF):
    yerr_lower[index] = HgammaIRAF[index] - yerr_lower[index]
    yerr_upper[index] = yerr_upper[index] - HgammaIRAF[index]
errorsHgammaIRAF = [yerr_lower, yerr_upper]

yerr_lower = [3.67E-11, 1.99E-10, 6.62E-11, 7.10E-11, 7.72E-11, 6.33E-12, 6.97E-12, 5.00E-13]
yerr_upper = [5.63E-11, 3.08E-10, 1.13E-10, 7.73E-11, 9.86E-11, 6.54E-12, 7.96E-12, 7.47E-13]
for index, item  in enumerate(HdeltaIRAF):
    yerr_lower[index] = HdeltaIRAF[index] - yerr_lower[index]
    yerr_upper[index] = yerr_upper[index] - HdeltaIRAF[index]
errorsHdeltaIRAF = [yerr_lower, yerr_upper]

fig, ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(7)
ax.plot(evolutionPlane, HalphaEvolution, label = 'Halpha script')
ax.errorbar(evolutionPlane, HalphaIRAF, yerr=errorsHalphaIRAF, marker='o', markersize=8, linestyle='dotted', label = 'Halpha IRAF')
ax.set(xlabel = 'Days', ylabel = f"Flux ({(u.erg / u.Angstrom / u.s / u.cm / u.cm).to_string('latex_inline')})")
ax.set_xscale('log')
#ax.set_yscale('log')
plt.legend()
plt.savefig('meassures_halpha.png')
plt.clf()

fig, ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(7)
ax.plot(evolutionPlane, HbetaEvolution, label = 'Hbeta script')
ax.errorbar(evolutionPlane, HbetaIRAF, yerr=errorsHbetaIRAF, marker='o', markersize=8, linestyle='dotted', label = 'Hbeta IRAF')
ax.set(xlabel = 'Days', ylabel = f"Flux ({(u.erg / u.Angstrom / u.s / u.cm / u.cm).to_string('latex_inline')})")
ax.set_xscale('log')
ax.set_yscale('log')
plt.legend()
plt.savefig('meassures_hbeta.png')
plt.clf()

fig, ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(7)
ax.plot(evolutionPlane, HgammaEvolution, label = 'Hgamma script')
ax.errorbar(evolutionPlane, HgammaIRAF, yerr=errorsHgammaIRAF, marker='o', markersize=8, linestyle='dotted', label = 'Hgamma IRAF')
ax.set(xlabel = 'Days', ylabel = f"Flux ({(u.erg / u.Angstrom / u.s / u.cm / u.cm).to_string('latex_inline')})")
ax.set_xscale('log')
ax.set_yscale('log')
plt.legend()
plt.savefig('meassures_hgamma.png')
plt.clf()

fig, ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(7)
ax.plot(evolutionPlane, HdeltaEvolution, label = 'Hdelta script')
ax.errorbar(evolutionPlane, HdeltaIRAF, yerr=errorsHdeltaIRAF, marker='o', markersize=8, linestyle='dotted', label = 'Hdelta IRAF')
ax.set(xlabel = 'Days', ylabel = f"Flux ({(u.erg / u.Angstrom / u.s / u.cm / u.cm).to_string('latex_inline')})")
ax.set_xscale('log')
ax.set_yscale('log')
plt.legend()
plt.savefig('meassures_hdelta.png')
plt.clf()
