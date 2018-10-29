from __future__ import division
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

import sncosmo
import astropy.units as u


def gridcreate(name, y, x, ratio, z, **kwargs):
    # Function that creates a blank axis canvas; each figure gets a name (or alternatively a number
    # if none is given), and gridspec creates an N*M grid onto which you can create axes for plots.
    # This returns a gridspec "instance" so you can specific which figure to put the axis on if you
    # have several on the go.
    plt.figure(name, figsize=(z*x, z*ratio*y))
    gs = gridspec.GridSpec(y, x, **kwargs)
    return gs

# TODO: add dithered observation of just galaxy (i.e., move centre by uniform([0, 1])*pixelscale)
# in sky position to then subsequently subtract from galaxy+supernova observation


ngals = 1
filters = ['z087', 'y106', 'w149', 'j129', 'h158', 'f184']
nfilts = len(filters)
gs = gridcreate('111', ngals, nfilts, 0.8, 15)

for i in xrange(0, ngals):
    file_ = open('creation_test.log', 'w+')
    stream_handler = logging.StreamHandler(file_)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    scm = SceneModule(logger=logger)

    # assuming surface brightnesses vary between roughly mu_e = 18-23 mag/arcsec^2 (mcgaugh
    # 1995, driver 2005)

    np.random.seed(seed=None)
    seedg = np.random.randint(100000)
    galaxy = {'n_gals': 500,
              'z_low': 0.0, 'z_high': 1.0,
              'rad_low': 0.5, 'rad_high': 2.0,
              'sb_v_low': 23.0, 'sb_v_high': 18.0,
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

    half_l_r = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], usecols=7)
    offset_r = 1.5 * half_l_r

    # 'star' will have a distance based on the redshift of the galaxy, given by
    # m - M = \mu = 42.38 - 5 log10(h) + 5 log10(z) + 5 log10(1+z) where h = 0.7
    # (given by H0 = 100h km/s/Mpc), based on cz = H0d, \mu = 5 log10(dL) - 5, dL = (1+z)d,
    # and 5log10(c/100km/s/Mpc / pc) = 42.38.
    # pretending that F125W on WFC3/IR is 2MASS J, we set the absolute magnitude of a
    # type Ia supernova to J = -19.0 (meikle 2018). set supernova to a star of the closest
    # blackbody (10000K; Zheng 2017) -- code uses Johnson I magnitude but Phillips (1993) says that
    # is also ~M = -19 -- currently just setting all absolute magnitudes to -19, but could change
    # if needed

    # random offsets for star should be in arcseconds; pixel scale is 0.11 arcsecond/pixel
    rand_ra = -offset_r + np.random.random_sample() * 2 * offset_r
    rand_dec = -offset_r + np.random.random_sample() * 2 * offset_r

    stellar = {'n_stars': 100000,
               'age_low': 1.0e7, 'age_high': 1.0e7,
               'z_low': -2.0, 'z_high': -2.0,
               'imf': 'powerlaw', 'alpha': -0.1,
               'binary_fraction': 0.0,
               'distribution': 'invpow', 'clustered': True,
               'radius': 0.0, 'radius_units': 'pc',
               'distance_low': 20.0, 'distance_high': 20.0,
               'offset_ra': rand_ra, 'offset_dec': rand_dec}
    stellar_cat_file = scm.CreatePopulation(stellar)

    z = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], usecols=3)
    # salt2 for Ia, s11-* where * is 2004hx for IIL/P, 2005hm for Ib, and 2006fo for Ic

    # draw salt2 x1 and c from salt2_parameters (gaussian, x1: x0=0.4, sigma=0.9, c: x0=-0.04,
    # sigma = 0.1) -- SALT2 only goes 2000-9200Angstrom, so is unuseable in the NIR
    # hsiao is valid over a wider wavelength range but has no stretch factor.
    # Hounsell 2017 gives SALT2 models over a wider wavelength range, given as sncosmo source
    # salt2-h17. both salt2 models have phases -20 to +50 days.
    sn_model = sncosmo.Model('salt2-h17')
    sn_model.set(t0=0.0, z=z, x1=0.5, c=0.0)

    for j in xrange(0, nfilts):
        ax = plt.subplot(gs[i, j])

        # then we need to load this file and get the redshift z to get the distance for column 3
        # below; set absolute magnitude to -19 and then calculate the apparent magnitude
        g = np.loadtxt(stellar_cat_file, comments=['\\', '|'])
        temp_fit = np.argmin(np.abs(g[:, 7] - 10000))

        # We need to change the distance and apparent magnitude, so edit (zero-indexed)
        # columns 3 and 12.
        g = g[temp_fit]

        # h = 0.7
        # mu = 42.38 - 5 * np.log10(h) + 5 * np.log10(z) + 5 * np.log10(1+z)
        # dl = 10**(mu/5 + 1)
        # M_ia = -19
        # m_ia = M_ia + mu

        # get the apparent magnitude of the supernova at a given time; first create the appropriate
        # filter for the observation
        f = pyfits.open('../../pandeia_data-1.0/wfirst/wfirstimager/filters/{}.fits'.format(filters[j]))
        data = f[1].data
        dispersion = [d[0] for d in data]
        transmission = [d[1] for d in data]
        bandpass = sncosmo.Bandpass(dispersion, transmission, wave_unit=u.micron, name=filters[j])
        sn_model.set_source_peakabsmag(-19.0, bandpass, 'ab')
        # TODO: change time to uniformly sample time series data
        m_ia = sn_model.bandmag(bandpass, magsys='ab', time=0)

        # if we need the 'star' of absolute magnitude M at distance dl then it has an apparent
        # magnitude of M + dl. thus after creating the source we need to move its distance modulus
        # and apparent magnitude by dM (the difference in absolute magnitudes)

        dmu = m_ia - g[12]
        g[12] = g[12] + dmu
        mu_s = 5 * np.log10(g[3]) - 5
        g[3] = 10**((mu_s + dmu)/5 - 1)

        star_cat_base, star_cat_ext = os.path.splitext(stellar_cat_file)
        new_stellar_cat_file = star_cat_base + '_single_star' + star_cat_ext
        f_w = open(new_stellar_cat_file, 'w+')
        with open(stellar_cat_file, 'r') as f_r:
            for line in f_r:
                if line[0] != "\\" and line[0] != "|":
                    break
                f_w.write(line)
        entry = ''
        # to force columns to line up with | breaks, each column must be a specific length, which is
        # for id, ra, dec, distance, age, metallicity, mass, teff, logg, binary, dataset, absolute and
        # apparent 8, 17, 17, 17, 17, 11, 17, 14, 12, 6, 7, 14, 12
        collengths = [8, 17, 17, 17, 17, 11, 17, 14, 12, 6, 7, 14, 12]
        dtypes = [int, float, float, float, int, float, float, float, float, int, int, float, float]
        for k, collength, dtype in zip(g, collengths, dtypes):
            k_ = dtype(k)
            entry = entry + ' {}{}'.format(k_, ' ' * (collength - len(str(k_))))
        entry = entry + '\n'
        f_w.write(entry)
        f_w.close()

        seedo = np.random.randint(100000)
        obs = {'instrument': 'WFI',
               'filters': [filters[j].upper()],
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
        output_galaxy_catalogues = obm.addCatalogue(new_galaxy_cat_file)
        output_stellar_catalogues = obm.addCatalogue(new_stellar_cat_file)
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
