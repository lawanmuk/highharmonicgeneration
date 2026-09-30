#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:14:35 2021

@author: Mukhtar Lawan
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

DATA_DIR = Path(__file__).resolve().parent.parent / "HHG_datasets"

#Plotting thing
ticks = np.arange(1,101,1)

primVol  = 1 #VOLUME # Divide the total current by the unit cell volume

# Constants
au2fs = 0.024188843
h2ev = 27.211385
om = 0.03037557 #CARRIERFREQ IN Hartree

# Energy Axis
energy = np.arange(0,25*om,0.0005)


######## Loading  data files######
def load(name):
    return np.loadtxt(DATA_DIR / f"{name}total_current.dat")

hbn_hhg_bcp1 = load('bcp_I=1.5e12') #bicircular
hbn_hhg_cpa = load('cp_I=1.5e12') #collinear
hbn_hhg_lpc = load('lp_I=1.5e12') #linear

######## Processing ########
def ft(time, current, energy):
    """Windowed Fourier transform of one current component."""
    x = time / time[-1]
    mask = 1 - 3*x**2 + 2*x**3  # smooth step: 1 at t=0, 0 at the end
    current = current*mask
    dt = time[1] - time[0]
    signal = np.zeros(len(energy)).astype(np.complex128)
    for ei, e in enumerate(energy):
        signal[ei] = np.sum(np.exp(1j*e*time)*current)*dt
    return signal

def spectrum(data, energy):
    """|J(w)|^2 summed over the x, y and z current components (columns 2, 3, 4)."""
    time = data[:, 1]
    return sum(np.abs(ft(time, data[:, c], energy))**2 for c in (2, 3, 4))

spec_bcp = spectrum(hbn_hhg_bcp1, energy) #bicircular
spec_cp = spectrum(hbn_hhg_cpa, energy) #collinear
spec_lp = spectrum(hbn_hhg_lpc, energy) #linear


######## Plotting ################
plt.clf()
fig, ax = plt.subplots()
plt.title('HHG Spectrum ')
ax.xaxis.set_minor_locator(FixedLocator(ticks))
#ax.xaxis.set_major_locator(FixedLocator(ticks))
plt.grid(which='minor')
plt.grid(which='major')

ax.semilogy(energy/om, energy**2*spec_bcp, label='Bicircular', color='blue')
ax.semilogy(energy/om, energy**2*spec_cp, label='Collinear', color='r')
ax.semilogy(energy/om, energy**2*spec_lp, label='Linear', color='orange')


plt.legend(framealpha=1.0)
plt.xlabel(r'Harmonic Order [$N\omega$]',fontsize=14)
plt.ylabel(r'Intensity [arb.]',fontsize=16)
plt.tight_layout()

fig = plt.gcf()
fig.set_size_inches(15.5, 8.5)
fig.savefig(Path(__file__).resolve().parent / 'hhg_spectra_I=1.5e12.png', dpi=300)
