import os
import sys
from glob import glob
path = '../../STScI-STIPS'
sys.path.insert(1, path)
import stips
print stips.__file__, stips.__version__
import matplotlib.gridspec as gridspec
import numpy as np

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

import logging

from astropy.visualization import simple_norm

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


gs = gridcreate('111', 2, 2, 0.8, 15)

for i in xrange(0, 4):

    ax = plt.subplot(gs[i])

    file_ = open('creation_test.log', 'w+')
    stream_handler = logging.StreamHandler(file_)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    scm = SceneModule(logger=logger)

    np.random.seed(seed=None)
    seedg = np.random.randint(100000)
    galaxy = {'n_gals': 500,
              'z_low': 0.0, 'z_high': 1.0,
              'rad_low': 0.5, 'rad_high': 2.0,
              'sb_v_low': 25.0, 'sb_v_high': 25.0,
              'distribution': 'uniform', 'clustered': False,
              'radius': 0.0, 'radius_units': 'arcsec',
              'offset_ra': 0.0, 'offset_dec': 0.0, 'seed': seedg}
    galaxy_cat_file = scm.CreateGalaxies(galaxy)
    gal_cat_base, gal_cat_ext = os.path.splitext(galaxy_cat_file)
    new_galaxy_cat_file = gal_cat_base + '_single_galaxy' + gal_cat_ext
    f_w = open(new_galaxy_cat_file, 'w+')
    with open(galaxy_cat_file, 'r') as f_r:
        for line in f_r:
            f_w.write(line)
            if line[0] != "\\" and line[0] != "|":
                break
    f_w.close()

    # 'star' will have a distance based on the redshift of the galaxy, given by
    # m - M = \mu = 42.38 - 5 log10(h) + 5 log10(z) + 5 log10(1+z) where h = 0.7
    # (given by H0 = 100h km/s/Mpc), based on cz = H0d, \mu = 5 log10(dL) - 5, dL = (1+z)d,
    # and 5log10(c/100km/s/Mpc / pc) = 42.38.
    # pretending that F125W on WFC3/IR is 2MASS J, we set the absolute magnitude of a
    # type Ia supernova to J = -19.0 (meikle 2018). set supernova to a star of the closest
    # blackbody (10000K; Zheng 2017)

    stellar = {'n_stars': 100000,
               'age_low': 1.0e7, 'age_high': 1.0e7,
               'z_low': -2.0, 'z_high': -2.0,
               'imf': 'powerlaw', 'alpha': -0.1,
               'binary_fraction': 0.0,
               'distribution': 'invpow', 'clustered': True,
               'radius': 100.0, 'radius_units': 'pc',
               'distance_low': 20.0, 'distance_high': 20.0,
               'offset_ra': 0.0, 'offset_dec': 0.0}
    stellar_cat_file = scm.CreatePopulation(stellar)

    # See if this can be generalised to skip both \ and | comments.
    gal = np.load(new_galaxy_cat_file, skiprows=24)
    # then we need to load this file and get the redshift z to get the distance for column 3
    # below; set absolute magnitude to -19 and then calculate the apparent magnitude
    g = np.loadtxt(stellar_cat_file, skiprows=26)
    temp_fit = np.argmin(np.abs(g[:, 7] - 10000))

    # We need to change the distance and both magnitudes, so edit (zero-indexed)
    # columns 3, 11 and 12.
    g = g[temp_fit]
    g[3] = 


    star_cat_base, star_cat_ext = os.path.splitext(stellar_cat_file)
    new_stellar_cat_file = star_cat_base + '_single_star' + star_cat_ext
    f_w = open(new_stellar_cat_file, 'w+')
    with open(stellar_cat_file, 'r') as f_r:
        for line in f_r:
            if line[0] != "\\" and line[0] != "|":
                break
            f_w.write(line)
    entry = ''
    for i in g:
        entry = entry + '{} '.format(i)
    entry = entry + '\n'
    f_w.write(entry)
    f_w.close()

    sys.exit()

    seedo = np.random.randint(100000)
    obs = {'instrument': 'WFI',
           'filters': ['H158'],
           'detectors': 1,
           'distortion': False,
           'oversample': 5,
           'pupil_mask': '',
           'background': 'avg',
           'observations_id': 1,
           'exptime': 1000,
           'offsets': [{'offset_id': 1, 'offset_centre': False, 'offset_ra': 0.0, 'offset_dec': 0.0, 'offset_pa': 0.0}],
           'small_subarray': True, 'seed': seedo}

    obm = ObservationModule(obs, logger=logger)
    obm.nextObservation()
    # output_stellar_catalogues = obm.addCatalogue(stellar_cat_file)
    output_galaxy_catalogues = obm.addCatalogue(new_galaxy_cat_file)
    psf_file = obm.addError()
    fits_file, mosaic_file, params = obm.finalize(mosaic=False)

    f = pyfits.open(fits_file)
    image = f[1].data

    norm = simple_norm(image, 'log', min_percent=90, max_percent=99.95)

    ax.imshow(image, origin='lower', cmap='viridis', norm=norm)
    ax.set_xlabel('x / pixel')
    ax.set_ylabel('y / pixel')
plt.tight_layout()
plt.savefig('test_galaxy.pdf')
