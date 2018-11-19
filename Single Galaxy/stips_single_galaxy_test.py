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
from scipy.special import gammaincinv

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


def new_galaxy_file_creation(galaxy_cat_file):
    gal_cat_base, gal_cat_ext = os.path.splitext(galaxy_cat_file)
    new_galaxy_cat_file = gal_cat_base + '_single_galaxy' + gal_cat_ext
    f_w = open(new_galaxy_cat_file, 'w+')
    with open(galaxy_cat_file, 'r') as f_r:
        for line in f_r:
            f_w.write(line)
            if line[0] != "\\" and line[0] != "|":
                break
    f_w.close()

    # create second version of single galaxy file, just with minorly offset ra/dec, uniformly
    # offset by a random amount of a pixel in each direction

    new_gal_cat_base, new_gal_cat_ext = os.path.splitext(new_galaxy_cat_file)
    shifted_galaxy_cat_file = new_gal_cat_base + '_single_galaxy_shift' + new_gal_cat_ext
    f_w = open(shifted_galaxy_cat_file, 'w+')
    col_get_flag = 0
    with open(new_galaxy_cat_file, 'r') as f_r:
        for line in f_r:
            if line[0] == "|" and col_get_flag == 0:
                col_line = line
                col_get_flag = 1
            if line[0] != "\\" and line[0] != "|":
                break
            f_w.write(line)

    g1 = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], dtype=int, usecols=0)
    g2 = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], dtype=float, usecols=[1, 2, 3, 7, 8, 9, 10, 11])
    g3 = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], dtype=str, usecols=[4, 5, 6])

    g1ind, g2ind, g3ind = 0, 0, 0
    entry = ''
    # to force columns to line up with | breaks, each column must be a specific length, the gap
    # between the various | separators
    q = np.array([s == "|" for s in col_line])
    col_inds = np.arange(0, len(col_line))[q]
    collengths = np.diff(col_inds) - 1
    dtypes = [int, float, float, float, str, str, str, float, float, float, float, float]
    for k in xrange(0, len(collengths)):
        dtype_ = dtypes[k]
        collength = collengths[k]
        if dtype_ == int:
            try:
                read = int(g1[g1ind])
            except IndexError:
                read = int(g1)
            g1ind += 1
        elif dtype_ == float:
            try:
                read = float(g2[g2ind])
            except IndexError:
                read = float(g2)
            g2ind += 1
        elif dtype_ == str:
            try:
                read = str(g3[g3ind])
            except IndexError:
                read = str(g3)
            g3ind += 1
        # replace ra/dec, zero-indexed columns 1+2, with random <1 pixel offsets
        if k == 1 or k == 2:
            read += pixel_scale/3600 * (-1 + 2 * np.random.random_sample())

        entry = entry + ' {}{}'.format(read, ' ' * (collength - len(str(read))))
    entry = entry + '\n'
    f_w.write(entry)
    f_w.close()

    return new_galaxy_cat_file, shifted_galaxy_cat_file

# TODO: add dithered observation of just galaxy (i.e., move centre by uniform([0, 1])*pixelscale)
# in sky position to then subsequently subtract from galaxy+supernova observation


ngals = 10
filters = ['z087', 'y106', 'w149', 'j129', 'h158', 'f184']
nfilts = len(filters)
nfilts = 6
pixel_scale = 0.11  # arcsecond/pixel

for i in xrange(0, ngals):
    gs = gridcreate('111', 3, nfilts, 0.8, 15)
    file_ = open('temp_files/creation_test.log', 'w+')
    stream_handler = logging.StreamHandler(file_)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    scm = SceneModule(logger=logger, out_path='temp_files')

    # assuming surface brightnesses vary between roughly mu_e = 20-23 mag/arcsec^2 (mcgaugh
    # 1995, driver 2005)

    np.random.seed(seed=None)
    seedg = np.random.randint(100000)
    galaxy = {'n_gals': 5000,
              'z_low': 0.0, 'z_high': 1.0,
              'rad_low': 0.3, 'rad_high': 2.5,
              'sb_v_low': 23.0, 'sb_v_high': 18.0,
              'distribution': 'uniform', 'clustered': False,
              'radius': 0.0, 'radius_units': 'arcsec',
              'offset_ra': 0.00001, 'offset_dec': 0.00001, 'seed': seedg}
    galaxy_cat_file = scm.CreateGalaxies(galaxy)
    new_galaxy_cat_file, shifted_galaxy_cat_file = new_galaxy_file_creation(galaxy_cat_file)

    half_l_r = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], usecols=7)
    disk_type = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], usecols=6, dtype=str)
    e_disk = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], usecols=8)
    pa_disk = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], usecols=9)
    n_type = 4 if disk_type == 'devauc' else 1
    # L(< R) / Ltot = \gamma(2n, x) / \Gamma(2n); scipy.special.gammainc is lower incomplete over
    # regular gamma function. Thus gammaincinv is the inverse to gammainc, solving
    # L(< r) / Ltot = Y, where Y is a large fraction
    y_frac = 0.75
    x_ = gammaincinv(2*n_type, y_frac)
    # however, x = bn * (R/Re)**(1/n), so we have to solve for R now, approximating bn
    offset_r = (x_ / (2*n_type - 1/3))**n_type * half_l_r

    # 'star' will have a distance based on the redshift of the galaxy, given by
    # m - M = \mu = 42.38 - 5 log10(h) + 5 log10(z) + 5 log10(1+z) where h = 0.7
    # (given by H0 = 100h km/s/Mpc), based on cz = H0d, \mu = 5 log10(dL) - 5, dL = (1+z)d,
    # and 5log10(c/100km/s/Mpc / pc) = 42.38.
    # pretending that F125W on WFC3/IR is 2MASS J, we set the absolute magnitude of a
    # type Ia supernova to J = -19.0 (meikle 2018). set supernova to a star of the closest
    # blackbody (10000K; Zheng 2017) -- code uses Johnson I magnitude but Phillips (1993) says that
    # is also ~M = -19 -- currently just setting all absolute magnitudes to -19, but could change
    # if needed

    endflag = 0
    while endflag == 0:
        # random offsets for star should be in arcseconds; pixel scale is 0.11 arcsecond/pixel
        rand_ra = -offset_r + np.random.random_sample() * 2 * offset_r
        rand_dec = -offset_r + np.random.random_sample() * 2 * offset_r
        # the full equation for a shifted, rotated ellipse, with semi-major axis
        # originally aligned with the y-axis, is given by:
        # ((x-p)cos(t)-(y-q)sin(t))**2/b**2 + ((x-p)sin(t) + (y-q)cos(t))**2/a**2 = 1
        p = galaxy['offset_ra']
        q = galaxy['offset_dec']
        x = rand_ra
        y = rand_dec
        t = pa_disk
        a = offset_r
        b = e_disk * offset_r
        if ((((x - p) * np.cos(t) - (y - q) * np.sin(t)) / b)**2 +
                (((x - p) * np.sin(t) + (y - q) * np.cos(t)) / a)**2 <= 1):
            endflag = 1

    stellar = {'n_stars': 500000,
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
        g[3] = 10**((mu_s + dmu)/5 + 1)

        star_cat_base, star_cat_ext = os.path.splitext(stellar_cat_file)
        new_stellar_cat_file = star_cat_base + '_single_star' + star_cat_ext
        f_w = open(new_stellar_cat_file, 'w+')
        col_get_flag = 0
        with open(stellar_cat_file, 'r') as f_r:
            for line in f_r:
                if line[0] == "|" and col_get_flag == 0:
                    col_line = line
                    col_get_flag = 1
                if line[0] != "\\" and line[0] != "|":
                    break
                f_w.write(line)
        # to force columns to line up with | breaks, each column must be a specific length, the gap
        # between the various | separators
        q = np.array([s == "|" for s in col_line])
        col_inds = np.arange(0, len(col_line))[q]
        collengths = np.diff(col_inds) - 1

        entry = ''
        dtypes = [int, float, float, float, int, float, float, float, float, int, int, float, float]
        for k, collength, dtype in zip(g, collengths, dtypes):
            k_ = dtype(k)
            entry = entry + ' {}{}'.format(k_, ' ' * (collength - len(str(k_))))
        entry = entry + '\n'
        f_w.write(entry)
        f_w.close()

        # TODO: vary exptime to explore the effects of exposure cadence on observation
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

        obm = ObservationModule(obs, logger=logger, out_path='temp_files', noise_floor=0)
        obm.nextObservation()

        output_galaxy_catalogues = obm.addCatalogue(new_galaxy_cat_file)
        output_stellar_catalogues = obm.addCatalogue(new_stellar_cat_file)
        psf_file = obm.addError()
        fits_file, mosaic_file, params = obm.finalize(mosaic=False)

        f = pyfits.open(fits_file)
        image = f[1].data

        norm = simple_norm(image / obs['exptime'], 'linear', percent=99.9)

        ax = plt.subplot(gs[0, j])
        img = ax.imshow(image / obs['exptime'], origin='lower', cmap='viridis', norm=norm)
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('Count rate / e$^-$ s$^{-1}$')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')
        if i == 0:
            ax.set_title('{} Sn Observation'.format(filters[j].upper()))
        # here the shifted galaxy is observed...
        obm_shifted = ObservationModule(obs, logger=logger, out_path='temp_files')
        obm_shifted.nextObservation()
        output_galaxy_catalogues_shifted = obm_shifted.addCatalogue(shifted_galaxy_cat_file)
        psf_file_shifted = obm_shifted.addError()
        fits_file_shifted, mosaic_file_shifted, params = obm_shifted.finalize(mosaic=False)

        f = pyfits.open(fits_file_shifted)
        image_shifted = f[1].data

        image_diff = image - image_shifted

        ax = plt.subplot(gs[1, j])
        img = ax.imshow(image_shifted / obs['exptime'], origin='lower', cmap='viridis', norm=norm)
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('Count rate / e$^-$ s$^{-1}$')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')
        if i == 0:
            ax.set_title('{} Reference'.format(filters[j].upper()))

        norm = simple_norm(image_diff / obs['exptime'], 'linear', percent=99.9)

        ax = plt.subplot(gs[2, j])
        img = ax.imshow(image_diff / obs['exptime'], origin='lower', cmap='viridis', norm=norm)
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('Count rate / e$^-$ s$^{-1}$')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')
        if i == 0:
            ax.set_title('{} Difference'.format(filters[j].upper()))

    plt.tight_layout()
    plt.savefig('out_gals/galaxy_{}.pdf'.format(i+1))
    plt.close()

# from scipy.special import gamma, gammainc
# bn = 2*n - 1/3
# x_ = bn * (10**(1/n))  # set this to be 10*re away, which should be enough
# g = gamma(2*n) * gammainc(2*n, x_)
# i_e = flux / re**2 / 2 / np.pi / n / np.exp(bn) * bn**(2*n) / g
# pixel_brightness = flux / (self.oversample*self.oversample)
# pixel_brightness = pixel_brightness / flux * i_e
