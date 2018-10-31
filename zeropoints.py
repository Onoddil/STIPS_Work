import pysynphot as pyS
import numpy as np

# Set up bandpass and source spectrum. Perform a 
# synthetic observation of the source. Note that the
# term 'counts' in PySynphot is variable for each instrument.
# For ACS, 'counts' refers to electrons.
bp = pyS.ObsBandpass('acs,wfc1,f814w,mjd#57754')
spec_bb = pyS.BlackBody(10000)
spec_bb_norm = spec_bb.renorm(1, 'counts', bp)
obs = pyS.Observation(spec_bb_norm, bp)

# Get photometric calibration information.
photflam = obs.effstim('flam') 
photplam = bp.pivot() 

# Get the magnitudes of the source spectrum in the
# bandpass. Because the source was normalized to
# 1 electron per second, the magnitudes are the 
# zeropoints in their respective systems.
# e.g. m_vega = -2.5*log10(counts) + zpt_vega
zp_vega = obs.effstim('vegamag')
zp_st = obs.effstim('stmag')
zp_ab = obs.effstim('abmag')

print(bp.pivot(),obs.efflam(), obs.pivot())
# Print the results.
print('PHOTFLAM = {}'.format(photflam))
print('PHOTPLAM = {}'.format(photplam))
print('')
print('VegaMag_ZP = {}'.format(zp_vega))
print('STMag_ZP = {}'.format(zp_st))
print('ABMag_ZP = {}'.format(zp_ab))