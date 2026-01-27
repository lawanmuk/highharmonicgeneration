#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:14:35 2021

@author: Kevin Lively and Mukhtar Lawan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator


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
hbn_hhg_bcp1 = np.loadtxt('bcp_I=1.5e12total_current') #bicircular
#hbn_hhg_bcp2 = np.loadtxt('bcp_I=5e12total_current') #bicircular
hbn_hhg_cpa = np.loadtxt('cp_I=1.5e12total_current') #collinear
#hbn_hhg_cpb = np.loadtxt('cp_I=5e12total_current') #collinear
#hbn_hhg_lpa = np.loadtxt('lp_I=1e11total_current') #linear
#hbn_hhg_lpb = np.loadtxt('lp_I=1e12total_current') #linear
hbn_hhg_lpc = np.loadtxt('lp_I=1.5e12total_current') #linear
#hbn_hhg_lpd = np.loadtxt('lp_I=5e12total_current') #linear
#hbn_hhg_lp = np.loadtxt('lp_I=1.5e12total_current') #linear

######## Processing ########
def ft(time,current, energy):
    Tcut = (1)*time[-1] #Mess with this
    tind = np.argmin(np.abs(time-Tcut))
    tcut = time[tind]
    mask = 1-(time[:tind]/tcut)**2 + 2*(time[:tind]/tcut)**3 #window function
    current = current[:tind]*mask
    signal = np.zeros(len(energy)).astype(np.complex128)
    de = energy[1]-energy[0]
    for ei, e in enumerate(energy):
        signal[ei] = np.sum(np.exp(1j*e*time[:tind])*current)*de
    return signal

a_eq_stabcp1 = ft(hbn_hhg_bcp1[:,1], hbn_hhg_bcp1[:,2], energy) #bicircular
#a_eq_stabcp2 = ft(hbn_hhg_bcp2[:,1], hbn_hhg_bcp2[:,2], energy) #bicircular
a_eq_stacpa = ft(hbn_hhg_cpa[:,1], hbn_hhg_cpa[:,2], energy) #collinear
#a_eq_stacpb = ft(hbn_hhg_cpb[:,1], hbn_hhg_cpb[:,2], energy) #collinear
#a_eq_staa = ft(hbn_hhg_lpa[:,1], hbn_hhg_lpa[:,2], energy) #linear
#a_eq_stab = ft(hbn_hhg_lpb[:,1], hbn_hhg_lpb[:,2], energy) #linear
a_eq_stac = ft(hbn_hhg_lpc[:,1], hbn_hhg_lpc[:,2], energy) #linear
#a_eq_stad = ft(hbn_hhg_lpd[:,1], hbn_hhg_lpd[:,2], energy )#linear
######## Plotting ################
plt.clf()
fig, ax = plt.subplots()
plt.title('HHG Spectrum ')
ax.xaxis.set_minor_locator(FixedLocator(ticks))
#ax.xaxis.set_major_locator(FixedLocator(ticks))
plt.grid(which='minor')
plt.grid(which='major')

ax.semilogy(energy/om,energy**2*np.abs(a_eq_stabcp1)**2,label='Bicircular',color = 'blue')
#ax.semilogy(energy/om,energy**2*np.abs(a_eq_stabcp2)**2,label='5e12',color = 'green')
ax.semilogy(energy/om, energy**2*np.abs(a_eq_stacpa)**2,label='Collinear',color = 'r')
#ax.semilogy(energy/om, energy**2*np.abs(a_eq_stacpb)**2,label='5e12',color = 'g')
#ax.semilogy(energy/om, energy**2*np.abs(a_eq_staa)**2,label='1e11',color = 'green')
#ax.semilogy(energy/om, energy**2*np.abs(a_eq_stab)**2,label='1e12',color = 'red')
ax.semilogy(energy/om, energy**2*np.abs(a_eq_stac)**2,label='Linear',color = 'orange')
#ax.semilogy(energy/om, energy**2*np.abs(a_eq_stad)**2,label='5e12',color = 'orange')


plt.legend(framealpha=1.0)
plt.xlabel(r'Harmonic Order [$N\omega$]',fontsize=14)
plt.ylabel(r'Intensity [arb.]',fontsize=16)
plt.tight_layout()

fig = plt.gcf()
fig.set_size_inches(15.5, 8.5)
fig.savefig('test2png.png', dpi=100)
