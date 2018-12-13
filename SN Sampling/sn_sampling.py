import os
import sys
import matplotlib.gridspec as gridspec
import numpy as np

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

import logging

from astropy.visualization import simple_norm
from scipy.special import gammaincinv
from astropy.table import Table
import sncosmo
import astropy.units as u
import timeit
from scipy.optimize import fmin_l_bfgs_b

import psf_mog_fitting as pmf


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

    print("{} choices, {:.0f}/{:.0f}/{:.0f} approximate minutes/hours/days".format(n_tot, time, time/60, time/60/24))


def gaussian_2d(x, x_t, mu, mu_t, sigma):
    det_sig = np.linalg.det(sigma)
    p = np.matmul(x_t - mu_t, np.linalg.inv(sigma))
    # if we don't take the 0, 0 slice we accidentally propagate to shape (len, len, len, len) by
    # having (len, len, 1, 1) shape passed through
    mal_dist_sq = np.matmul(p, (x - mu))[:, :, 0, 0]
    gauss_pdf = np.exp(-0.5 * mal_dist_sq) / (2 * np.pi * np.sqrt(det_sig))
    return gauss_pdf


# flat and dark can be loaded from the stips fits file or found elsewhere, they are simply input
# files to be multipled/added to the original data.
def add_dark(image, d):
    # choice returns a random choice from np.arange(a) if just given a single integer a
    x_i, y_j = (np.random.choice(d.shape[i]-image.shape[i]) for i in [0, 1])
    image += d[x_i:x_i + image.shape[0], y_j:y_j + image.shape[1]]
    return image


def mult_flat(image, d):
    # choice returns a random choice from np.arange(a) if just given a single integer a
    x_i, y_j = (np.random.choice(d.shape[i]-image.shape[i]) for i in [0, 1])
    image *= d[x_i:x_i + image.shape[0], y_j:y_j + image.shape[1]]
    return image


# read noise is just a constant single read value
def add_read(image, readnoise):
    image += readnoise
    return image


def set_exptime(image, exptime):
    image *= exptime
    return image


def add_background(image, bkg):
    image += bkg
    return image


# if lambda is a numpy array then size is ignored and each value is used creating a new array of
# the original shape. we could instead, for large lambda, generate a gaussian of mean 0 and
# variance lambda; this is the more general formula allowing for low counts, however.
def add_poisson(image):
    return np.random.poisson(lam=image).astype(float)


def mog_galaxy(pixel_scale, filt_zp, psf_c, gal_params):
    mu_0, n_type, e_disk, pa_disk, half_l_r, offset_r, Vgm_unit, cms, vms, mag, offset_ra_pix, \
        offset_dec_pix = gal_params

    cm_exp = np.array([0.00077, 0.01077, 0.07313, 0.37188, 1.39727, 3.56054, 4.74340, 1.78732])
    vm_exp_sqrt = np.array([0.02393, 0.06490, 0.13580, 0.25096, 0.42942, 0.69672, 1.08879,
                            1.67294])
    cm_dev = np.array([0.00139, 0.00941, 0.04441, 0.16162, 0.48121, 1.20357, 2.54182, 4.46441,
                       6.22821, 6.15393])
    vm_dev_sqrt = np.array([0.00087, 0.00296, 0.00792, 0.01902, 0.04289, 0.09351, 0.20168, 0.44126,
                            1.01833, 2.74555])

    # this requires re-normalising as Hogg & Lang (2013) created profiles with unit intensity at
    # their half-light radius, with total flux for the given profile simply being the sum of the
    # MoG coefficients, cm, so we ensure that sum(cm) = 1 for normalisation purposes
    cms = cm_dev / np.sum(cm_dev) if n_type == 4 else cm_exp / np.sum(cm_exp)
    # Vm is always circular so this doesn't need to be a full matrix, but PSF m/V do need to
    vms = np.array(vm_dev_sqrt)**2 if n_type == 4 else np.array(vm_exp_sqrt)**2

    mks = psf_c[:, [0, 1]].reshape(-1, 2, 1)
    pks = psf_c[:, 5]  # what is referred to as 'c' in psf_mog_fitting is p_k in H&L13
    sx, sy, r = psf_c[:, 2], psf_c[:, 3], psf_c[:, 4]
    Vks = np.array([[[sx[q]**2, r[q]*sx[q]*sy[q]], [r[q]*sx[q]*sy[q], sy[q]**2]] for
                    q in range(0, len(sx))])
    # covariance matrix and mean positions given in pixels, but need converting to half-light
    mks *= (pixel_scale / half_l_r)
    Vks *= (pixel_scale / half_l_r)**2

    len_image = np.ceil(2.2*offset_r / pixel_scale).astype(int)
    len_image = len_image + 1 if len_image % 2 == 0 else len_image
    len_image = max(25, len_image)
    image = np.zeros((len_image, len_image), float)
    x_cent, y_cent = (image.shape[0]-1)/2, (image.shape[1]-1)/2

    # positons should be in dimensionless but physical coordinates in terms of Re; first the
    # Xg vector needs converting from its given (ra, dec) to pixel coordiantes, to be placed
    # in the xy grid correctly (currently this just defaults to the central pixel, but it may
    # not in the future)
    xg = np.array([[(offset_ra_pix + x_cent) * pixel_scale / half_l_r],
                   [(offset_dec_pix + y_cent) * pixel_scale / half_l_r]])
    x_pos = (np.arange(0, image.shape[0])) * pixel_scale / half_l_r
    y_pos = (np.arange(0, image.shape[1])) * pixel_scale / half_l_r
    x, y = np.meshgrid(x_pos, y_pos, indexing='xy')
    # n-D gaussians have mahalnobis distance (x - mu)^T Sigma^-1 (x - mu) so coords_t and m_t
    # should be *row* vectors, and thus be shape (1, x) while coords and m should be column
    # vectors and shape (x, 1). starting with coords, we need to add the grid of data, so if
    # this array has shape (1, 2, y, x), and if we transpose it it'll have shape (x, y, 2, 1)
    coords = np.transpose(np.array([[x, y]]))
    # the "transpose" of the vector x turns from being a column vector (shape = (2, 1)) to a
    # row vector (shape = (1, 2)), but should still have external shape (x, y), so we start
    # with vector of (2, 1, y, x) and transpose again
    coords_t = np.transpose(np.array([[x], [y]]))
    # total flux in galaxy -- ensure that all units end up in flux as counts/s accordingly
    Sg = 10**(-1/2.5 * (mag - filt_zp))
    for k in range(0, len(mks)):
        pk = pks[k]
        Vk = Vks[k]
        mk = mks[k]
        for m in range(0, len(vms)):
            cm = cms[m]
            vm = vms[m]
            # Vgm = RVR^T = vm RR^T given that V = vmI
            Vgm = vm * Vgm_unit
            # reshape m and m_t to force propagation of arrays, remembering row vectors are
            # (1, x) and column vectors are (x, 1) in shape
            m = (mk + xg).reshape(1, 1, 2, 1)
            m_t = m.reshape(1, 1, 1, 2)
            V = Vgm + Vk
            g_2d = gaussian_2d(coords, coords_t, m, m_t, V)
            # having converted the covariance matrix to half-light radii, we need to account for a
            # corresponding reverse correction so that the PSF dimensions are correct, which are
            # defined in pure pixel scale
            image += Sg * cm * pk * g_2d / (half_l_r / pixel_scale)**2
    return image


def mog_add_psf(image, psf_params, filt_zp, psf_c):
    image = np.copy(image)
    offset_ra_pix, offset_dec_pix, mag = psf_params
    x_cent, y_cent = (image.shape[0]-1)/2, (image.shape[1]-1)/2
    xg = np.array([[(offset_ra_pix + x_cent) * pixel_scale],
                   [(offset_dec_pix + y_cent) * pixel_scale]])
    x_pos = (np.arange(0, image.shape[0])) * pixel_scale
    y_pos = (np.arange(0, image.shape[1])) * pixel_scale
    x, y = np.meshgrid(x_pos, y_pos, indexing='xy')
    # n-D gaussians have mahalnobis distance (x - mu)^T Sigma^-1 (x - mu) so coords_t and m_t
    # should be *row* vectors, and thus be shape (1, x) while coords and m should be column
    # vectors and shape (x, 1). starting with coords, we need to add the grid of data, so if
    # this array has shape (1, 2, y, x), and if we transpose it it'll have shape (x, y, 2, 1)
    coords = np.transpose(np.array([[x, y]]))
    # the "transpose" of the vector x turns from being a column vector (shape = (2, 1)) to a
    # row vector (shape = (1, 2)), but should still have external shape (x, y), so we start
    # with vector of (2, 1, y, x) and transpose again
    coords_t = np.transpose(np.array([[x], [y]]))

    mks = psf_c[:, [0, 1]].reshape(-1, 2, 1)
    pks = psf_c[:, 5]  # what is referred to as 'c' in psf_mog_fitting is p_k in H&L13
    sx, sy, r = psf_c[:, 2], psf_c[:, 3], psf_c[:, 4]
    Vks = np.array([[[sx[q]**2, r[q]*sx[q]*sy[q]], [r[q]*sx[q]*sy[q], sy[q]**2]] for
                    q in range(0, len(sx))])
    # convert PSF position and covariance matrix to arcseconds, from pixels
    mks *= pixel_scale
    Vks *= pixel_scale**2

    # total flux in source -- ensure that all units end up in flux as counts/s accordingly
    Sg = 10**(-1/2.5 * (mag - filt_zp))
    count = np.sum(image)
    for k in range(0, len(mks)):
        pk = pks[k]
        V = Vks[k]
        mk = mks[k]
        # reshape m and m_t to force propagation of arrays, remembering row vectors are
        # (1, x) and column vectors are (x, 1) in shape
        m = (mk + xg).reshape(1, 1, 2, 1)
        m_t = m.reshape(1, 1, 1, 2)
        g_2d = gaussian_2d(coords, coords_t, m, m_t, V)
        # equivalent to the mog_galaxy version, we converted the covariance matrix to arcseconds
        # so need to undo the unit change to get the correct dimensions, having fit for the PSF
        # in pure pixels
        image += Sg * pk * g_2d * pixel_scale**2
    return image


def make_figures(filters, img_sn, img_no_sn, diff_img, exptime, directory, counter, times):
    nfilts = len(filters)
    ntimes = len(times)
    gs = gridcreate('111', 3*ntimes, nfilts, 0.8, 15)
    for k in range(0, ntimes):
        for j in range(0, nfilts):
            image = img_sn[k][j]
            image_shifted = img_no_sn[j]
            image_diff = diff_img[k][j]
            norm = simple_norm(image / exptime, 'linear', percent=99.9)

            ax = plt.subplot(gs[0 + 3*k, j])
            img = ax.imshow(image.T / exptime, origin='lower', cmap='viridis', norm=norm)
            cb = plt.colorbar(img, ax=ax, use_gridspec=True)
            cb.set_label('Count rate / e$^-$ s$^{-1}$')
            ax.set_xlabel('x / pixel')
            if j == 0:
                ax.set_ylabel('Sn Observation, t = {} days\ny / pixel'.format(times[k]))
            else:
                ax.set_ylabel('y / pixel')
            ax.set_title(filters[j].upper())

            ax = plt.subplot(gs[1 + 3*k, j])
            img = ax.imshow(image_shifted.T / exptime, origin='lower', cmap='viridis', norm=norm)
            cb = plt.colorbar(img, ax=ax, use_gridspec=True)
            cb.set_label('Count rate / e$^-$ s$^{-1}$')
            ax.set_xlabel('x / pixel')
            if j == 0:
                ax.set_ylabel('Sn Reference\ny / pixel')
            else:
                ax.set_ylabel('y / pixel')

            norm = simple_norm(image_diff / exptime, 'linear', percent=99.9)

            ax = plt.subplot(gs[2 + 3*k, j])
            img = ax.imshow(image_diff.T / exptime, origin='lower', cmap='viridis', norm=norm)
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
            x1, c = np.random.normal(0.4, 0.9), np.random.normal(-0.04, 0.1)
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


def make_images(filters, pixel_scale, sn_type, times, exptime, filt_zp, psf_comp_filename,
                dark_img, flat_img, readnoise, t0):
    nfilts = len(filters)
    ntimes = len(times)

    # assuming surface brightnesses vary between roughly mu_e = 18-23 mag/arcsec^2 (mcgaugh
    # 1995, driver 2005, shen 2003 -- assume shen 2003 gives gaussian with mu=20.94, sigma=0.74)

    mu_0 = np.random.normal(20.94, 0.74)
    # elliptical galaxies approximated as de vaucouler (n=4) sersic profiles, spirals as
    # exponentials (n=1). axial ratios vary 0.5-1 for ellipticals and 0.1-1 for spirals
    rand_num = np.random.uniform()
    n_type = 4 if rand_num < 0.5 else 1
    # randomly draw the ellipcity from 0.5/0.1 to 1, depending on sersic index
    e_disk = np.random.uniform(0.5 if n_type == 4 else 0.1, 1.0)
    # position angle can be uniformly drawn [0, 360) as we convert to radians elsewhere
    pa_disk = np.random.uniform(0, 360)
    # half-light radius can be uniformly drawn between two reasonable radii
    lr_low, lr_high = 0.3, 2.5
    half_l_r = np.random.uniform(lr_low, lr_high)
    # L(< R) / Ltot = \gamma(2n, x) / \Gamma(2n); scipy.special.gammainc is lower incomplete over
    # regular gamma function. Thus gammaincinv is the inverse to gammainc, solving
    # L(< r) / Ltot = Y, where Y is a large fraction
    y_frac = 0.75
    x_ = gammaincinv(2*n_type, y_frac)
    # however, x = bn * (R/Re)**(1/n), so we have to solve for R now, approximating bn; in arcsec
    offset_r = (x_ / (2*n_type - 1/3))**n_type * half_l_r
    # redshift randomly drawn between two values uniformly
    z_low, z_high = 0.2, 1.0
    z = np.random.uniform(z_low, z_high)

    cm_exp = np.array([0.00077, 0.01077, 0.07313, 0.37188, 1.39727, 3.56054, 4.74340, 1.78732])
    vm_exp_sqrt = np.array([0.02393, 0.06490, 0.13580, 0.25096, 0.42942, 0.69672, 1.08879,
                            1.67294])
    cm_dev = np.array([0.00139, 0.00941, 0.04441, 0.16162, 0.48121, 1.20357, 2.54182, 4.46441,
                       6.22821, 6.15393])
    vm_dev_sqrt = np.array([0.00087, 0.00296, 0.00792, 0.01902, 0.04289, 0.09351, 0.20168, 0.44126,
                            1.01833, 2.74555])

    psf_comp = np.load(psf_comp_filename)

    # this requires re-normalising as Hogg & Lang (2013) created profiles with unit intensity at
    # their half-light radius, with total flux for the given profile simply being the sum of the
    # MoG coefficients, cm, so we ensure that sum(cm) = 1 for normalisation purposes
    cms = cm_dev / np.sum(cm_dev) if n_type == 4 else cm_exp / np.sum(cm_exp)
    # Vm is always circular so this doesn't need to be a full matrix, but PSF m/V do need to
    vms = np.array(vm_dev_sqrt)**2 if n_type == 4 else np.array(vm_exp_sqrt)**2

    # 0.75 mag is really 2.5 * log10(2), for double flux, given area is half-light radius
    mag = mu_0 - 2.5 * np.log10(np.pi * half_l_r**2 * e_disk) - 2.5 * np.log10(2)

    # since everything is defined in units of half-light radius, the "semi-major axis" is always
    # one with the semi-minor axis simply being the eccentricity (b/a, not to be confused with
    # the ellipicity = sqrt((a**2 - b**2)/a**2) = 1 - b/a) of the ellipse
    a, b = 1, e_disk
    t = np.radians(pa_disk)
    Rg = np.array([[-a * np.sin(t), b * np.cos(t)], [a * np.cos(t), b * np.sin(t)]])
    Vgm_unit = np.matmul(Rg, np.transpose(Rg))

    endflag = 0
    while endflag == 0:
        # random offsets for star should be in arcseconds; pixel scale is 0.11 arcsecond/pixel
        rand_ra = -offset_r + np.random.random_sample() * 2 * offset_r
        rand_dec = -offset_r + np.random.random_sample() * 2 * offset_r
        # the full equation for a shifted, rotated ellipse, with semi-major axis
        # originally aligned with the y-axis, is given by:
        # ((x-p)cos(t)-(y-q)sin(t))**2/b**2 + ((x-p)sin(t) + (y-q)cos(t))**2/a**2 = 1
        p = 0
        q = 0
        x = rand_ra
        y = rand_dec
        t = np.radians(pa_disk)
        a = offset_r
        b = e_disk * offset_r
        if ((((x - p) * np.cos(t) - (y - q) * np.sin(t)) / b)**2 +
                (((x - p) * np.sin(t) + (y - q) * np.cos(t)) / a)**2 <= 1):
            endflag = 1

    sn_model = get_sn_model(sn_type, 1, t0=t0, z=z)
    # pretending that F125W on WFC3/IR is 2MASS J, we set the absolute magnitude of a
    # type Ia supernova to J = -19.0 (meikle 2000). set supernova to a star of the closest
    # blackbody (10000K; Zheng 2017) -- code uses Johnson I magnitude but Phillips (1993) says that
    # is also ~M = -19 -- currently just setting absolute magnitudes to -19, but could change
    # if needed
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

    # given some zodiacal light flux, in ergcm^-2s^-1A^-1arcsec^-2, flip given the ST ZP,
    # then convert back to flux
    zod_flux = 2e-18  # erg/cm^2/s/A/arcsec^2
    zod_mag = -2.5 * np.log10(zod_flux) - 21.1  # st mag system
    zod_count = 10**(-1/2.5 * (zod_mag - filt_zp[0]))  # currently using an AB ZP...
    gal_params = [mu_0, n_type, e_disk, pa_disk, half_l_r, offset_r, Vgm_unit, cms, vms, mag]
    # currently assuming a simple half-pixel dither; TODO: check if this is right and update
    second_gal_offets = np.empty((nfilts, 2), float)
    for j in range(0, nfilts):
        # define a random pixel offset ra/dec
        offset_ra, offset_dec = np.random.uniform(0.01, 0.99), np.random.uniform(0.01, 0.99)
        sign = -1 if np.random.uniform(0, 1) < 0.5 else 1
        # non-reference image should be offset by half a pixel, wrapped around [0, 1]
        second_gal_offets[j, 0] = (offset_ra + sign * 0.5 + 1) % 1
        second_gal_offets[j, 1] = (offset_dec + sign * 0.5 + 1) % 1
        image = mog_galaxy(pixel_scale, filt_zp[j], psf_comp[j], gal_params +
                           [offset_ra, offset_dec])
        image = add_background(image, zod_count)
        image = set_exptime(image, exptime)
        image = add_poisson(image)
        image = mult_flat(image, flat_img)
        image = add_dark(image, dark_img)
        image = add_read(image, readnoise)
        images_without_sn.append(image)

    true_flux = []
    for k in range(0, ntimes):
        images = []
        images_diff = []
        for j in range(0, nfilts):
            image_shifted = images_without_sn[j]
            # TODO: add exposure and readout time so that exposures are staggered in time
            time = times[k] + t0

            # get the apparent magnitude of the supernova at a given time; first get the
            # appropriate filter for the observation
            bandpass = sncosmo.get_bandpass(filters[j])
            # time should be in days
            m_ia = sn_model.bandmag(bandpass, magsys='ab', time=time)
            if np.isnan(m_ia):
                m_ia = -2.5 * np.log10(0.01) + filt_zp[j]
            # 'star' will have a distance based on the redshift of the galaxy, given by
            # m - M = \mu = 42.38 - 5 log10(h) + 5 log10(z) + 5 log10(1+z) where h = 0.7
            # (given by H0 = 100h km/s/Mpc), based on cz = H0d, \mu = 5 log10(dL) - 5, dL = (1+z)d,
            # and 5log10(c/100km/s/Mpc / pc) = 42.38.
            # h = 0.7
            # mu = 42.38 - 5 * np.log10(h) + 5 * np.log10(z) + 5 * np.log10(1+z)
            # dl = 10**(mu/5 + 1)
            # M_ia = -19
            # m_ia = M_ia + mu

            # TODO: add background noise.
            # background comes from jwst_backgrounds.background, converted from MJy/sr to
            # mJy/pixel, converted to counts through the filter and zp I guess?
            # if cosmicrays are needed then figure out what stips does for that...
            offset_ra, offset_dec = second_gal_offets[j, :]
            image = mog_galaxy(pixel_scale, filt_zp[j], psf_comp[j],
                               gal_params + [offset_ra, offset_dec])
            countgal, galf = np.sum(image), 10**(-1/2.5 * (mag - filt_zp[j]))
            image = mog_add_psf(image, [rand_ra / pixel_scale, rand_dec / pixel_scale, m_ia],
                                filt_zp[j], psf_comp[j])
            countsn, snf = np.sum(image) - countgal, 10**(-1/2.5 * (m_ia - filt_zp[j]))
            image = add_background(image, zod_count)
            image = set_exptime(image, exptime)
            image = add_poisson(image)
            image = mult_flat(image, flat_img)
            image = add_dark(image, dark_img)
            image = add_read(image, readnoise)

            images.append(image)
            image_diff = image - image_shifted
            images_diff.append(image_diff)

            time_array.append(time)
            band_array.append(filters[j])

            x_cent, y_cent = (image.shape[0]-1)/2, (image.shape[1]-1)/2
            xind, yind = np.floor(rand_ra / pixel_scale + x_cent).astype(int), np.floor(rand_dec / pixel_scale + y_cent).astype(int)
            N = 4

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

            true_flux.append(10**(-1/2.5 * (m_ia - filt_zp[j])))

        images_with_sn.append(images)
        diff_images.append(images_diff)

    lc_data = [np.array(time_array), np.array(band_array), np.array(flux_array),
               np.array(fluxerr_array), np.array(zp_array), np.array(zpsys_array)]

    sn_params = [sn_model['z'], sn_model['t0'], sn_model['x0'], sn_model['x1'], sn_model['c']]
    return images_with_sn, images_without_sn, diff_images, lc_data, sn_params, true_flux


def fit_lc(lc_data, sn_types, directory, filters, counter, figtext, ncol, minsnr):
    for sn_type in sn_types:
        start = timeit.default_timer()
        params = ['z', 't0', 'x0']
        if sn_type == 'Ia':
            params += ['x1', 'c']
        sn_model = get_sn_model(sn_type, 0)
        # place upper limits on the redshift probeable, by finding the z at which each filter drops
        # out of being in overlap with the model
        z_uppers = np.empty(len(filters), float)
        for i in range(0, len(filters)):
            z = 0
            while sn_model.bandoverlap(filters[i], z=z):
                z += 0.01
            z_uppers[i] = min(2.5, z - 0.01)
        # set the bounds on z to be at most the smallest of those available by the given filters in
        # the set being fit here
        bounds = {'z': (0.0, np.amin(z_uppers))}
        # x1 and c bounded by 3.5-sigma regions (x1: mu=0.4, sigma=0.9, c: mu=-0.04, sigma = 0.1)
        if sn_type == 'Ia':
            bounds.update({'x1': (-2.75, 3.55), 'c': (-0.39, 0.31)})
        result = None
        fitted_model = None
        for z_init in np.linspace(0, np.amin(z_uppers), 5):
            sn_model.set(z=z_init)
            result_temp, fitted_model_temp = sncosmo.fit_lc(lc_data, sn_model, params,
                                                            bounds=bounds, minsnr=minsnr,
                                                            guess_z=False)
            if result is None or result_temp.chisq < result.chisq:
                result = result_temp
                fitted_model = fitted_model_temp
        print('Fit: {:.2f}s'.format(timeit.default_timer()-start))
        if 'success' not in result.message:
            print("Message:", result.message)

        figtext = [figtext[0], figtext[1] + '\n' + r'$\chi^2_{{\nu={}}}$ = {:.3f}'.format(result.ndof, result.chisq/result.ndof)]

        fig = sncosmo.plot_lc(lc_data, model=fitted_model, errors=result.errors, xfigsize=15*ncol,
                              tighten_ylim=True, ncol=ncol, figtext=figtext)
        fig.tight_layout()
        fig.savefig('{}/fit_{}_{}.pdf'.format(directory, counter, sn_type))

        return result


if __name__ == '__main__':
    # run_mins, run_n, run_obs, n_runs = 10/60, 6, 5, 100
    # model_number(run_mins, run_n, run_obs, n_runs)

    # sys.exit()

    ngals = 10
    pixel_scale = 0.11  # arcsecond/pixel
    directory = 'out_gals'

    # TODO: vary these parameters
    filters = ['z087', 'y106', 'w149', 'j129', 'h158', 'f184']
    # 1 count/s for infinite aperture, hounsell17, AB magnitudes
    filt_zp = [26.39, 26.41, 27.50, 26.35, 26.41, 25.96]

    for j in range(0, len(filters)):
        f = pyfits.open('../../pandeia_data-1.0/wfirst/wfirstimager/filters/{}.fits'
                        .format(filters[j]))
        data = f[1].data
        dispersion = [d[0] for d in data]
        transmission = [d[1] for d in data]
        bandpass = sncosmo.Bandpass(dispersion, transmission, wave_unit=u.micron, name=filters[j])
        sncosmo.register(bandpass)

    # TODO: vary exptime to explore the effects of exposure cadence on observation
    exptime = 1000  # seconds
    sn_type = 'Ia'

    t_low, t_high, t_interval = -10, 30, 5
    times = np.arange(t_low, t_high+1e-10, t_interval)
    psf_comp_filename = 'psf_comp.npy'

    # psf_names = ['../../pandeia_data-1.0/wfirst/wfirstimager/psfs/wfirstimager_any_{}.fits'.format(num) for num in [0.8421, 1.0697, 1.4464, 1.2476, 1.5536, 1.9068]]
    psf_names = ['../../../Buffalo/PSFSTD_WFC3IR_F{}W.fits'.format(q) for q in [105, 125, 160]]
    oversampling, noise_removal, N_comp, cut = 4, 0, 7, 0.01
    psf_comp_filename = 'psf_comp.npy'
    # pmf.psf_mog_fitting(psf_names, oversampling, noise_removal, psf_comp_filename, cut, N_comp)

    filters = ['F160W']  # ['F105W', 'F125W', 'F160W']
    filt_zp = [25.95]  # [27.69, 28.02, 28.19] - st; [26.27, 26.23, 25.95] - ab
    pixel_scale = 0.13

    f = pyfits.open('../err_rdrk_wfi.fits')
    # dark current is in counts/s, so requires correcting by the exosure time
    dark_img = f[1].data * exptime
    f = pyfits.open('../err_flat_wfi.fits')
    flat_img = f[1].data
    # currently what is in stips, claimed 'max ramp, lowest noise'
    readnoise = 12
    t0 = 50000
    minsnr = 5

    ncol = min(3, len(filters))

    gs_ = gridcreate('asjhfs', 1, 5, 0.8, 15)
    axs = [plt.subplot(gs_[i]) for i in range(0, 5)]
    true_params = np.empty((ngals, 5), float)
    fit_params = np.empty((ngals, 5, 2), float)

    i = 0
    while i < ngals:
        start = timeit.default_timer()
        images_with_sn, images_without_sn, diff_images, lc_data, sn_params, true_flux = \
            make_images(filters, pixel_scale, sn_type, times, exptime, filt_zp, psf_comp_filename,
                        dark_img, flat_img, readnoise, t0)
        print("Make: {:.2f}s".format(timeit.default_timer()-start))
        lc_data_table = Table(data=lc_data,
                              names=['time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'])
        if not np.amax(lc_data_table['flux'].data / lc_data_table['fluxerr'].data) >= minsnr:
            continue

        figtext = 'z = {:.3f}\nt0 = {:.1f}\nx0 = {:.5f}x1 = {:.5f}\nc = {:.5f}'.format(
                  *sn_params)
        figtext_split = figtext.split('x1')
        figtext = [figtext_split[0], 'x1' + figtext_split[1]]
        # TODO: expand to include all types of Sne
        result = fit_lc(lc_data_table, [sn_type], directory, filters, i+1, figtext, ncol, minsnr)
        make_figures(filters, images_with_sn, images_without_sn, diff_images, exptime,
                     directory, i+1, times)

        gs = gridcreate('09', 1, 1, 0.8, 15)
        ax = plt.subplot(gs[0])
        ax.errorbar(lc_data_table['time'], lc_data_table['flux'] - true_flux,
                    yerr=lc_data_table['fluxerr'], fmt='k.')
        ax.axhline(0, c='k', ls='--')
        ax.set_xlabel('Time')
        ax.set_ylabel('Flux difference (fit - true)')
        plt.tight_layout()
        plt.savefig('{}/flux_ratio_{}.pdf'.format(directory, i+1))

        true_params[i, :] = sn_params
        fit_params[i, :, 0] = result.parameters
        fit_params[i, :, 1] = [result.errors[q] for q in ['z', 't0', 'x0', 'x1', 'c']]

        i += 1

    plt.figure('asjhfs')
    for i, (ax, name) in enumerate(zip(axs, ['z', 't0', 'x0', 'x1', 'c'])):
        ax.errorbar(np.arange(1, ngals+1), fit_params[:, i, 0] - true_params[:, i],
                    yerr=fit_params[:, i, 1], fmt='k.')
        ax.axhline(0, c='k', ls='--')
        ax.set_xlabel('Count')
        ax.set_ylabel('{} difference (fit - true)'.format(name))
    plt.tight_layout()
    plt.savefig('{}/derived_parameter_ratio.pdf'.format(directory))
