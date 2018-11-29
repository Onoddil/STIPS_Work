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
from astropy.table import Table

from stips.scene_module import SceneModule
from stips.observation_module import ObservationModule

import sncosmo
import astropy.units as u

try:
    profile
except NameError:
    profile = lambda x: x


def gridcreate(name, y, x, ratio, z, **kwargs):
    # Function that creates a blank axis canvas; each figure gets a name (or alternatively a number
    # if none is given), and gridspec creates an N*M grid onto which you can create axes for plots.
    # This returns a gridspec "instance" so you can specific which figure to put the axis on if you
    # have several on the go.
    plt.figure(name, figsize=(z*x, z*ratio*y))
    gs = gridspec.GridSpec(y, x, **kwargs)
    return gs


def model_number(run_minutes, run_n, run_obs, n_runs):
    n = 7  # including R, eventually
    t = 50
    n_filt_choice = 0
    for k in np.arange(2, n+1):
        n_filt_choice += np.math.factorial(n) / np.math.factorial(k) / np.math.factorial(n - k)
    # cadence can vary from, say, 5 days to 25 days (5 days being the minimum needed, and 25 days
    # giving 2 data points per lightcurve), so cadence could be varied in 5s initially, and thus
    cadence_interval = 5
    cadences = np.arange(5, 25+1e-10, cadence_interval)
    n_cadence = len(cadences)
    # assuming 50 day lightcurve, observations per filter per lightcurve used as scaling for
    # run_obs to scale run_minutes (as with k and run_n scaling similarly)
    time = 0
    # finally, scale by n_runs for each cadence/filter combination
    for c in cadences:
        for k in np.arange(2, n+1):
            # we always have to make filt reference images, but obs sn+galaxy images, so we really
            # have 1+obs creation runs
            time += np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k)) * (
                (1+k) / (1+run_n)) * run_minutes * (t / c / run_obs) * n_runs

    n_tot = n_filt_choice * n_cadence

    print "{} choices, {:.0f}/{:.0f}/{:.0f} approximate minutes/hours/days".format(n_tot, time, time/60, time/60/24)


def gaussian_2d(x, x_t, mu, mu_t, sigma):
    det_sig = np.linalg.det(sigma)
    print x.shape, x_t.shape, mu.shape, mu_t.shape
    print (x_t - mu_t).shape
    p = np.matmul(x_t - mu_t, np.linalg.inv(sigma))
    mal_dist_sq = np.matmul(p, (x - mu))
    gauss_pdf = np.exp(-0.5 * mal_dist_sq) / (2 * np.pi) * np.sqrt(det_sig)
    return gauss_pdf


def mog_galaxy_test(filters, pixel_scale, exptime, filt_zp):
    nfilts = len(filters)
    file_ = open('temp_files/creation_test.log', 'w+')
    stream_handler = logging.StreamHandler(file_)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    scm = SceneModule(logger=logger, out_path='temp_files')
    np.random.seed(seed=None)
    seedg = np.random.randint(100000)
    seedo = np.random.randint(100000)
    galaxy = {'n_gals': 5000,
              'z_low': 0.1, 'z_high': 1.0,
              'rad_low': 0.3, 'rad_high': 2.5,
              'sb_v_low': 23.0, 'sb_v_high': 18.0,
              'distribution': 'uniform', 'clustered': False,
              'radius': 0.0, 'radius_units': 'arcsec',
              'offset_ra': 0.00001, 'offset_dec': 0.00001, 'seed': seedg}
    galaxy_cat_file = scm.CreateGalaxies(galaxy)
    new_galaxy_cat_file, shifted_galaxy_cat_file = new_galaxy_file_creation(galaxy_cat_file)
    obs = {'instrument': 'WFI',
           'filters': [p.upper() for p in filters],
           'detectors': 1,
           'distortion': False,
           'oversample': 5,
           'pupil_mask': '',
           'background': 'none',  # temporary to allow for comparison with tests, should be avg
           'observations_id': 1,
           'exptime': exptime,
           'offsets': [{'offset_id': 1, 'offset_centre': False, 'offset_ra': 0.0, 'offset_dec': 0.0, 'offset_pa': 0.0}],
           'small_subarray': True, 'seed': seedo}
    obm_shifted = ObservationModule(obs, logger=logger, out_path='temp_files')

    disk_type = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], usecols=[6], dtype=str)
    loading = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], usecols=[1, 2, 3, 7, 8, 9, 11])
    ra, dec, z, half_l_r, e_disk, pa_disk, mu = loading
    n_type = 4 if disk_type == 'devauc' else 1
    cm_exp = [0.00077, 0.01077, 0.07313, 0.37188, 1.39727, 3.56054, 4.74340, 1.78732]
    vm_exp_sqrt = [0.78732, 0.06490, 0.13580, 0.25096, 0.42942, 0.69672, 1.08879, 1.67294]
    cm_dev = [0.00139, 0.00941, 0.04441, 0.16162, 0.48121, 1.20357, 2.54182, 4.46441, 6.22821,
              6.15393]
    vm_dev_sqrt = [0.00087, 0.00296, 0.00792, 0.01902, 0.04289, 0.09351, 0.20168, 0.44126,
                   1.01833, 2.74555]
    pks = np.array([1])
    Vk_real = np.array([[[0.13, 0], [0, 0.13]]])
    mks = np.array([[[0], [0]]])
    Vks = Vk_real / half_l_r

    cms = cm_dev if disk_type == 'devauc' else cm_exp
    # Vm is always circular so this doesn't need to be a full matrix, but PSF m/V do need to
    vms = np.array(vm_dev_sqrt)**2 if disk_type == 'devauc' else np.array(vm_exp_sqrt)**2

    # 0.75 mag is really 2.5 * log10(2), for double flux, given area is half-light radius
    mag = mu - 2.5 * np.log10(np.pi * half_l_r**2 * e_disk) - 2.5 * np.log10(2)

    a = half_l_r
    b = e_disk * half_l_r
    t = pa_disk
    Rg = np.array([[-a * np.sin(t), b * np.cos(t)], [a * np.cos(t), b * np.sin(t)]])
    Vgm_unit = np.matmul(Rg, np.transpose(Rg))

    gs = gridcreate('adsq', 3, len(filters), 0.8, 15)

    for j in xrange(0, len(filters)):
        obm_shifted.nextObservation()
        output_galaxy_catalogues_shifted = obm_shifted.addCatalogue(shifted_galaxy_cat_file)
        # psf_file_shifted = obm_shifted.addError(parallel=True)
        fits_file_shifted, mosaic_file_shifted, params = obm_shifted.finalize(mosaic=False)
        f = pyfits.open(fits_file_shifted)
        image_full = f[1].data

        image_test = np.zeros_like(image_full)
        x_cent, y_cent = np.ceil(image_test.shape[0])/2, np.ceil(image_test.shape[1])/2
        xg = np.array([[ra / pixel_scale + x_cent],
                       [dec / pixel_scale + y_cent]])
        x_pix = (np.arange(0, image_test.shape[0]) + x_cent) * pixel_scale / half_l_r
        y_pix = (np.arange(0, image_test.shape[1]) + y_cent) * pixel_scale / half_l_r
        x, y = np.meshgrid(y_pix, x_pix)
        # this array has shape (2, 1, y, x), and if we transpose it it'll have shape (x, y, 1, 2)
        coords = np.transpose(np.array([[x], [y]]))
        # the "transpose" of the vector x turns from being a column vector (shape = (1, 2)) to a
        # row vector (shape = (2, 1)), but should still have external shape (x, y), so we start
        # with vector of (1, 2, y, x) and transpose again
        coords_t = np.transpose(np.array([[x, y]]))
        # total flux in galaxy
        Sg = 10**(-1/2.5 * (mag - filt_zp[j]))
        for k in xrange(0, len(mks)):
            pk = pks[k]
            Vk = Vks[k]
            mk = mks[k]
            for m in xrange(0, len(vms)):
                cm = cms[m]
                vm = vms[m]
                # Vgm = RVR^T = v RR^T given that V = vI
                Vgm = vm * Vgm_unit
                m = (mk + xg).reshape(1, 1, 1, 2)
                m_t = m.reshape(1, 1, 2, 1)
                v = Vgm + Vk
                image_test += Sg * cm * pk * gaussian_2d(coords, coords_t, m, m_t, v)

        for k, im, name in enumerate(zip([image_full, image_test, image_full/(image_full+image_test)], ['full', 'test', 'full / (full + test)'])):
            ax = plt.subplot(gs[k, j])
            norm = simple_norm(im, 'linear', percent=99.9)
            img = ax.imshow(im, origin='lower', cmap='viridis', norm=norm)
            cb = plt.colorbar(img, ax=ax, use_gridspec=True)
            cb.set_label('Counts / e$^-$')
            ax.set_xlabel('x / pixel')
            ax.set_ylabel('y / pixel')
    plt.tight_layout()
    plt.savefig('out_gals/test_MoG.pdf')


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

    g1 = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], dtype=int, usecols=[0])
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
            # if the string format of the position is longer than its allowed column, probably
            # when a minus sign is added, we split the '[-]aaaaaae-bb' format at 'e', remove one
            # 'a', and put it back together, effectively removing the least significant digit
            if collength < len(str(read)):
                splitter = str(read).split('e')
                read = float(splitter[0][:-1] + 'e' + splitter[1])

        entry = entry + ' {}{}'.format(read, ' ' * (collength - len(str(read))))
    entry = entry + '\n'
    f_w.write(entry)
    f_w.close()

    return new_galaxy_cat_file, shifted_galaxy_cat_file


def new_star_file_creation(stellar_cat_file, g):
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

    return new_stellar_cat_file


def make_figures(filters, img_sn, img_no_sn, diff_img, exptime, directory, counter, times):
    nfilts = len(filters)
    ntimes = len(times)
    gs = gridcreate('111', 3*ntimes, nfilts, 0.8, 15)
    for k in xrange(0, ntimes):
        for j in xrange(0, nfilts):
            image = img_sn[k][j]
            image_shifted = img_no_sn[j]
            image_diff = diff_img[k][j]
            norm = simple_norm(image / exptime, 'linear', percent=99.9)

            ax = plt.subplot(gs[0 + 3*k, j])
            img = ax.imshow(image / exptime, origin='lower', cmap='viridis', norm=norm)
            cb = plt.colorbar(img, ax=ax, use_gridspec=True)
            cb.set_label('Count rate / e$^-$ s$^{-1}$')
            ax.set_xlabel('x / pixel')
            if j == 0:
                ax.set_ylabel('Sn Observation, t = {} days\ny / pixel'.format(times[k]))
            else:
                ax.set_ylabel('y / pixel')
            ax.set_title(filters[j].upper())

            ax = plt.subplot(gs[1 + 3*k, j])
            img = ax.imshow(image_shifted / exptime, origin='lower', cmap='viridis', norm=norm)
            cb = plt.colorbar(img, ax=ax, use_gridspec=True)
            cb.set_label('Count rate / e$^-$ s$^{-1}$')
            ax.set_xlabel('x / pixel')
            if j == 0:
                ax.set_ylabel('Sn Reference\ny / pixel')
            else:
                ax.set_ylabel('y / pixel')

            norm = simple_norm(image_diff / exptime, 'linear', percent=99.9)

            ax = plt.subplot(gs[2 + 3*k, j])
            img = ax.imshow(image_diff / exptime, origin='lower', cmap='viridis', norm=norm)
            cb = plt.colorbar(img, ax=ax, use_gridspec=True)
            cb.set_label('Count rate / e$^-$ s$^{-1}$')
            ax.set_xlabel('x / pixel')
            if j == 0:
                ax.set_ylabel('Difference\ny / pixel')
            else:
                ax.set_ylabel('y / pixel')

    plt.tight_layout()
    plt.savefig('{}/galaxy_{}.pdf'.format(directory, counter))
    plt.close()


def get_sn_model(sn_type, setflag, t0=0.0, z=0.0):
    # salt2 for Ia, s11-* where * is 2004hx for IIL/P, 2005hm for Ib, and 2006fo for Ic
    # draw salt2 x1 and c from salt2_parameters (gaussian, x1: x0=0.4, sigma=0.9, c: x0=-0.04,
    # sigma = 0.1)
    # Hounsell 2017 gives SALT2 models over a wider wavelength range, given as sncosmo source
    # salt2-h17. both salt2 models have phases -20 to +50 days.

    if sn_type == 'Ia':
        sn_model = sncosmo.Model('salt2-h17')
        if setflag:
            # TODO: vary the stretch parameters as above
            x1, c = 0.5, 0.0
            sn_model.set(t0=t0, z=z, x1=x1, c=c)
    elif sn_type == 'Ib':
        sn_model = sncosmo.Model('s11-2005hm')
        if setflag:
            sn_model.set(t0=t0, z=z)
    elif sn_type == 'Ic':
        sn_model = sncosmo.Model('s11-2006fo')
        if setflag:
            sn_model.set(t0=t0, z=z)
    elif sn_type == 'IIL' or sn_type == 'IIP':
        sn_model = sncosmo.Model('s11-2004hx')
        if setflag:
            sn_model.set(t0=t0, z=z)
    # TODO: add galaxy dust via smcosmo.F99Dust([r_v])

    return sn_model


@profile
def make_images(filters, pixel_scale, sn_type, times, exptime, filt_zp):
    nfilts = len(filters)
    ntimes = len(times)

    file_ = open('temp_files/creation_test.log', 'w+')
    stream_handler = logging.StreamHandler(file_)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    scm = SceneModule(logger=logger, out_path='temp_files')

    # assuming surface brightnesses vary between roughly mu_e = 18-23 mag/arcsec^2 (mcgaugh
    # 1995, driver 2005, shen 2003)

    np.random.seed(seed=None)
    seedg = np.random.randint(100000)
    galaxy = {'n_gals': 5000,
              'z_low': 0.2, 'z_high': 1.0,
              'rad_low': 0.3, 'rad_high': 2.5,
              'sb_v_low': 23.0, 'sb_v_high': 18.0,
              'distribution': 'uniform', 'clustered': False,
              'radius': 0.0, 'radius_units': 'arcsec',
              'offset_ra': 0.00001, 'offset_dec': 0.00001, 'seed': seedg}
    galaxy_cat_file = scm.CreateGalaxies(galaxy)
    new_galaxy_cat_file, shifted_galaxy_cat_file = new_galaxy_file_creation(galaxy_cat_file)

    half_l_r = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], usecols=[7])
    disk_type = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], usecols=[6], dtype=str)
    e_disk = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], usecols=[8])
    pa_disk = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], usecols=[9])
    n_type = 4 if disk_type == 'devauc' else 1
    # L(< R) / Ltot = \gamma(2n, x) / \Gamma(2n); scipy.special.gammainc is lower incomplete over
    # regular gamma function. Thus gammaincinv is the inverse to gammainc, solving
    # L(< r) / Ltot = Y, where Y is a large fraction
    y_frac = 0.75
    x_ = gammaincinv(2*n_type, y_frac)
    # however, x = bn * (R/Re)**(1/n), so we have to solve for R now, approximating bn
    offset_r = (x_ / (2*n_type - 1/3))**n_type * half_l_r

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

    stellar = {'n_stars': 10000,
               'age_low': 1.0e7, 'age_high': 1.0e7,
               'z_low': -2.0, 'z_high': -2.0,
               'imf': 'powerlaw', 'alpha': -0.1,
               'binary_fraction': 0.0,
               'distribution': 'invpow', 'clustered': True,
               'radius': 0.0, 'radius_units': 'pc',
               'distance_low': 20.0, 'distance_high': 20.0,
               'offset_ra': rand_ra, 'offset_dec': rand_dec}
    stellar_cat_file = scm.CreatePopulation(stellar)

    z = np.loadtxt(new_galaxy_cat_file, comments=['\\', '|'], usecols=[3])

    sn_model = get_sn_model(sn_type, 1, t0=0.0, z=z)
    # pretending that F125W on WFC3/IR is 2MASS J, we set the absolute magnitude of a
    # type Ia supernova to J = -19.0 (meikle 2000). set supernova to a star of the closest
    # blackbody (10000K; Zheng 2017) -- code uses Johnson I magnitude but Phillips (1993) says that
    # is also ~M = -19 -- currently just setting absolute magnitudes to -19, but could change
    # if needed
    # TODO: verfiy the transformation from 2MASS internal zero point -- 3.129E-13Wcm-2um-1 --
    # to AB -- 31.47E-11 erg s-1 cm-2 A-1 (as per Bessel 1998) -- and how that affects the ZP
    sn_model.set_source_peakabsmag(-19.0, 'f125w', 'ab')

    images_with_sn = []
    images_without_sn = []
    diff_images = []

    # things that are needed to create the astropy.table.Table for use in fit_lc:
    # time, band (name, see registered bandpasses), flux, fluxerr [both just derived from an
    # image somehow], zp, zpsys [zeropoint and name of system]

    time_array = []
    band_array = []
    flux_array = []
    fluxerr_array = []
    zp_array = []
    zpsys_array = []

    # then we need to load this file and get the redshift z to get the distance for column
    # 3 below, and then calculate the apparent magnitude
    g_orig = np.loadtxt(stellar_cat_file, comments=['\\', '|'])
    temp_fit = np.argmin(np.abs(g_orig[:, 7] - 10000))
    g_orig = g_orig[temp_fit]

    seedo = np.random.randint(100000)
    obs = {'instrument': 'WFI',
           'filters': [p.upper() for p in filters],
           'detectors': 1,
           'distortion': False,
           'oversample': 5,
           'pupil_mask': '',
           'background': 'avg',
           'observations_id': 1,
           'exptime': exptime,
           'offsets': [{'offset_id': 1, 'offset_centre': False, 'offset_ra': 0.0, 'offset_dec': 0.0, 'offset_pa': 0.0}],
           'small_subarray': True, 'seed': seedo}
    obm_shifted = ObservationModule(obs, logger=logger, out_path='temp_files')
    for j in xrange(0, nfilts):
        # here the shifted galaxy is observed...
        obm_shifted.nextObservation()
        output_galaxy_catalogues_shifted = obm_shifted.addCatalogue(shifted_galaxy_cat_file)
        psf_file_shifted = obm_shifted.addError(parallel=True)
        fits_file_shifted, mosaic_file_shifted, params = obm_shifted.finalize(mosaic=False)
        f = pyfits.open(fits_file_shifted)
        images_without_sn.append(f[1].data)

    # as the inner loop, we need the filters in order [a, b, c, a, b, c, ...] in filt_list, so
    # that we can simply loop them as obs.nextObservation(); we also need the filters capitalised
    filt_list = [p.upper() for p in filters] * ntimes

    seedo = np.random.randint(100000)
    obs = {'instrument': 'WFI',
           'filters': filt_list,
           'detectors': 1,
           'distortion': False,
           'oversample': 5,
           'pupil_mask': '',
           'background': 'avg',
           'observations_id': 1,
           'exptime': exptime,
           'offsets': [{'offset_id': 1, 'offset_centre': False, 'offset_ra': 0.0, 'offset_dec': 0.0, 'offset_pa': 0.0}],
           'small_subarray': True, 'seed': seedo}

    obm = ObservationModule(obs, logger=logger, out_path='temp_files', noise_floor=0)

    for k in xrange(0, ntimes):
        images = []
        images_diff = []
        for j in xrange(0, nfilts):
            image_shifted = images_without_sn[j]
            # TODO: add exposure and readout time so that exposures are staggered in time
            time = times[k]

            # We need to change the distance and apparent magnitude, so edit (zero-indexed)
            # columns 3 and 12.
            g = g_orig.copy()
            # get the apparent magnitude of the supernova at a given time; first get the
            # appropriate filter for the observation
            bandpass = sncosmo.get_bandpass(filters[j])
            # time should be in days
            m_ia = sn_model.bandmag(bandpass, magsys='ab', time=time)

            # 'star' will have a distance based on the redshift of the galaxy, given by
            # m - M = \mu = 42.38 - 5 log10(h) + 5 log10(z) + 5 log10(1+z) where h = 0.7
            # (given by H0 = 100h km/s/Mpc), based on cz = H0d, \mu = 5 log10(dL) - 5, dL = (1+z)d,
            # and 5log10(c/100km/s/Mpc / pc) = 42.38.
            # h = 0.7
            # mu = 42.38 - 5 * np.log10(h) + 5 * np.log10(z) + 5 * np.log10(1+z)
            # dl = 10**(mu/5 + 1)
            # M_ia = -19
            # m_ia = M_ia + mu

            # if we need the 'star' of absolute magnitude M at distance dl then it has an apparent
            # magnitude of M + dl. thus after creating the source we need to move its distance
            # modulus and apparent magnitude by dM (the difference in absolute magnitudes)
            dmu = m_ia - g[12]
            g[12] = g[12] + dmu
            mu_s = 5 * np.log10(g[3]) - 5
            g[3] = 10**((mu_s + dmu)/5 + 1)

            new_stellar_cat_file = new_star_file_creation(stellar_cat_file, g)

            obm.nextObservation()
            output_galaxy_catalogues = obm.addCatalogue(new_galaxy_cat_file)
            output_stellar_catalogues = obm.addCatalogue(new_stellar_cat_file)
            psf_file = obm.addError(parallel=True)
            fits_file, mosaic_file, params = obm.finalize(mosaic=False)
            f = pyfits.open(fits_file)
            image = f[1].data
            images.append(image)

            image_diff = image - image_shifted
            images_diff.append(image_diff)

            time_array.append(time)
            band_array.append(filters[j])

            xind, yind = np.unravel_index(np.argmax(image_diff), image_diff.shape)
            N = 10

            # current naive sum the entire (box) 'aperture' flux of the Sn, correcting for
            # exposure time in both counts and uncertainty
            diff_sum = np.sum(image_diff[xind-N:xind+N+1, yind-N:yind+N+1]) / exptime
            diff_sum_err = np.sqrt(np.sum(image[xind-N:xind+N+1, yind-N:yind+N+1] +
                                   image_shifted[xind-N:xind+N+1, yind-N:yind+N+1])) / exptime
            flux_array.append(diff_sum)
            fluxerr_array.append(diff_sum_err)
            zp_array.append(filt_zp[j])  # filter-specific zeropoint
            # TODO: swap to STmag from the AB system
            zpsys_array.append('ab')

        images_with_sn.append(images)
        diff_images.append(images_diff)

    lc_data = [np.array(time_array), np.array(band_array), np.array(flux_array),
               np.array(fluxerr_array), np.array(zp_array), np.array(zpsys_array)]

    sn_params = [sn_model['z'], sn_model['t0'], sn_model['x0']]
    return images_with_sn, images_without_sn, diff_images, lc_data, sn_params


def fit_lc(lc_data, sn_types, directory, filters, counter, figtext):
    for sn_type in sn_types:
        params = ['z', 't0', 'x0']
        if sn_type == 'Ia':
            params += ['x1', 'c']
        sn_model = get_sn_model(sn_type, 0)
        # place upper limits on the redshift probeable, by finding the z at which each filter drops
        # out of being in overlap with the model
        z_uppers = np.empty(len(filters), float)
        for i in xrange(0, len(filters)):
            z = 0
            while sn_model.bandoverlap(filters[i], z=z):
                z += 0.01
            z_uppers[i] = z - 0.01
        # set the bounds on z to be at most the smallest of those available by the given filters in
        # the set being fit here
        bounds = {'z': (0.0, np.amin(z_uppers)), 'x1': (0, 1)}
        # x1 and c bounded by 6-sigma regions (x1: x0=0.4, sigma=0.9, c: x0=-0.04, sigma = 0.1)
        if sn_type == 'Ia':
            bounds.update({'x1': (-5, 5.8), 'c': (-0.64, 0.56)})
        result = None
        fitted_model = None
        for z_init in np.linspace(0, np.amin(z_uppers), 20):
            sn_model.set(z=z_init)
            result_temp, fitted_model_temp = sncosmo.fit_lc(lc_data, sn_model, params,
                                                            bounds=bounds, minsnr=3, guess_z=False)
            if result is None or result_temp.chisq < result.chisq:
                result = result_temp
                fitted_model = fitted_model_temp
        print("Message:", result.message)
        print("Number of chi^2 function calls:", result.ncall)
        print("Number of degrees of freedom in fit:", result.ndof)
        print("chi^2 value at minimum:", result.chisq)
        print("model parameters:", result.param_names)
        print("best-fit values:", result.parameters)
        ncol = 3
        fig = sncosmo.plot_lc(lc_data, model=fitted_model, errors=result.errors, xfigsize=15*ncol,
                              tighten_ylim=True, ncol=ncol, figtext=figtext)
        fig.tight_layout()
        fig.savefig('{}/fit_{}_{}.pdf'.format(directory, counter, sn_type))


# run_mins, run_n, run_obs, n_runs = 2, 6, 5, 100
# model_number(run_mins, run_n, run_obs, n_runs)

# sys.exit()

# TODO: track down the zero point changes in STIPS? one seems to be in pandeia somewhere
import warnings
warnings.simplefilter('ignore', RuntimeWarning)

ngals = 1
pixel_scale = 0.11  # arcsecond/pixel
directory = 'out_gals'

# TODO: vary these parameters
filters = ['z087', 'y106', 'w149', 'j129', 'h158', 'f184']
# 1 count/s for infinite aperture, hounsell17, AB magnitudes
filt_zp = [26.39, 26.41, 27.50, 26.35, 26.41, 25.96]

for j in xrange(0, len(filters)):
    f = pyfits.open('../../pandeia_data-1.0/wfirst/wfirstimager/filters/{}.fits'.format(filters[j]))
    data = f[1].data
    dispersion = [d[0] for d in data]
    transmission = [d[1] for d in data]
    bandpass = sncosmo.Bandpass(dispersion, transmission, wave_unit=u.micron, name=filters[j])
    sncosmo.register(bandpass)

# TODO: vary exptime to explore the effects of exposure cadence on observation
exptime = 1000  # seconds
sn_type = 'Ia'

times = [-10, 0, 10, 20, 30]

mog_galaxy_test(filters, pixel_scale, exptime, filt_zp)

sys.exit()

import timeit
# TODO: see about downloading the jwst_backgrounds cache and putting it somewhere for offline use?
for i in xrange(0, ngals):
    start = timeit.default_timer()
    images_with_sn, images_without_sn, diff_images, lc_data, sn_params = \
        make_images(filters, pixel_scale, sn_type, times, exptime, filt_zp)
    print "make", timeit.default_timer()-start
    start = timeit.default_timer()
    make_figures(filters, images_with_sn, images_without_sn, diff_images, exptime, directory, i+1,
                 times)
    print "plot", timeit.default_timer()-start
    start = timeit.default_timer()
    lc_data_table = Table(data=lc_data, names=['time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'])
    print lc_data_table['flux']
    figtext = 'z = {:.3f}, t0 = {:.1f}, x0 = {:.5f}'.format(*sn_params)
    # TODO: expand to include all types of Sne
    fit_lc(lc_data_table, [sn_type], directory, filters, i+1, figtext)
    print "fit", timeit.default_timer()-start
    print sn_params
