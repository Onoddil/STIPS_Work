import os
import sys
from glob import glob
path = '../STScI-STIPS'
sys.path.insert(1, path)
import stips
print stips.__file__, stips.__version__
import matplotlib.gridspec as gridspec

from stips.scene_module import SceneModule
from stips.observation_module import ObservationModule

# import astropy.io.fits as pyfits
# import numpy as np
# f = pyfits.open('/home/ono/Documents/STScI-STIPS/CDBS/grid/bc95/templates/bc95_d_50E8.fits')
# g = f[1].data
# print g
# print g.shape
# w = np.array([i[0] for i in g])
# fl = np.array([i[1] for i in g])
# print np.percentile(w, [0, 10, 25, 50, 75, 90, 100]), np.percentile(fl, [0, 10, 25, 50, 75, 90, 100])
# sys.exit()


def gridcreate(name, y, x, ratio, z, **kwargs):
    # Function that creates a blank axis canvas; each figure gets a name (or alternatively a number
    # if none is given), and gridspec creates an N*M grid onto which you can create axes for plots.
    # This returns a gridspec "instance" so you can specific which figure to put the axis on if you
    # have several on the go.
    plt.figure(name, figsize=(z*x, z*ratio*y))
    gs = gridspec.GridSpec(y, x, **kwargs)
    return gs


scm = SceneModule()

stellar = {'n_stars': 50000,
           'age_low': 1.0e12, 'age_high': 1.0e12,
           'z_low': -2.0, 'z_high': -2.0,
           'imf': 'salpeter', 'alpha': -2.35,
           'binary_fraction': 0.1,
           'distribution': 'invpow', 'clustered': True,
           'radius': 100.0, 'radius_units': 'pc',
           'distance_low': 20.0, 'distance_high': 20.0,
           'offset_ra': 0.0, 'offset_dec': 0.0}
stellar_cat_file = scm.CreatePopulation(stellar)

galaxy = {'n_gals': 500,
          'z_low': 0.0, 'z_high': 1.0,
          'rad_low': 0.01, 'rad_high': 2.0,
          'sb_v_low': 30.0, 'sb_v_high': 25.0,
          'distribution': 'uniform', 'clustered': False,
          'radius': 200.0, 'radius_units': 'arcsec',
          'offset_ra': 0.0, 'offset_dec': 0.0}
galaxy_cat_file = scm.CreateGalaxies(galaxy)

obs = {'instrument': 'WFI',
       'filters': ['H158'],
       'detectors': 1,
       'distortion': False,
       'oversample': 5,
       'pupil_mask': '',
       'background': 'avg',
       'observations_id': 1,
       'exptime': 1000,
       'offsets': [{'offset_id': 1, 'offset_centre': False, 'offset_ra': 0.0, 'offset_dec': 0.0, 'offset_pa': 0.0}]}
obm = ObservationModule(obs)
obm.nextObservation()
output_stellar_catalogues = obm.addCatalogue(stellar_cat_file)
output_galaxy_catalogues = obm.addCatalogue(galaxy_cat_file)
psf_file = obm.addError()
fits_file, mosaic_file, params = obm.finalize(mosaic=False)

import astropy.io.fits as pyfits
f = pyfits.open(fits_file)
image = f[1].data

import matplotlib.pyplot as plt

from astropy.visualization import simple_norm
norm = simple_norm(image, 'log', min_percent=85, max_percent=99.8)

gs = gridcreate('111', 1, 1, 0.8, 15)
ax = plt.subplot(gs[0])
ax.imshow(image, origin='lower', cmap='viridis', norm=norm)
ax.set_xlabel('x / pixel')
ax.set_ylabel('y / pixel')
plt.tight_layout()
plt.savefig('test_image.pdf')

obm = ObservationModule(obs)
obm.nextObservation()
output_stellar_catalogues = obm.addCatalogue(stellar_cat_file)
psf_file = obm.addError()
fits_file, mosaic_file, params = obm.finalize(mosaic=False)

import astropy.io.fits as pyfits
f = pyfits.open(fits_file)
image = f[1].data

import matplotlib.pyplot as plt

from astropy.visualization import simple_norm
norm = simple_norm(image, 'log', min_percent=85, max_percent=99.8)

gs = gridcreate('111', 1, 1, 0.8, 15)
ax = plt.subplot(gs[0])
ax.imshow(image, origin='lower', cmap='viridis', norm=norm)
ax.set_xlabel('x / pixel')
ax.set_ylabel('y / pixel')
plt.tight_layout()
plt.savefig('test_image_stars.pdf')

obm = ObservationModule(obs)
obm.nextObservation()
output_galaxy_catalogues = obm.addCatalogue(galaxy_cat_file)
psf_file = obm.addError()
fits_file, mosaic_file, params = obm.finalize(mosaic=False)

import astropy.io.fits as pyfits
f = pyfits.open(fits_file)
image = f[1].data

import matplotlib.pyplot as plt

from astropy.visualization import simple_norm
norm = simple_norm(image, 'log', min_percent=85, max_percent=99.8)

gs = gridcreate('111', 1, 1, 0.8, 15)
ax = plt.subplot(gs[0])
ax.imshow(image, origin='lower', cmap='viridis', norm=norm)
ax.set_xlabel('x / pixel')
ax.set_ylabel('y / pixel')
plt.tight_layout()
plt.savefig('test_image_galaxies.pdf')
