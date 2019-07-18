import os
import sys
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import gammaincinv
from astropy.table import Table
import sncosmo
from scipy.ndimage import shift
import glob
import timeit

try:
    dummy = profile
except:
    profile = lambda x: x
np.set_printoptions(edgeitems=10, linewidth=500, precision=4, floatmode='maxprec')
import galsim.wfirst as wfirst

import sn_sampling_extras as sse
import psf_mog_fitting as pmf

# things to add to detector to create accurate noise model:
# sources, counting as poissonian noise
# dark
# read
# background (zodiacal light)
# thermal background
# reciprocity failure
# non-linearity
# interpixel capacitance
# persistence
# charge diffusion

# nonlinearity_beta - The coefficient of the (counts)^2 term in the detector nonlinearity
# function.  This will not ordinarily be accessed directly by users; instead, it will be accessed
# by the convenience function in this module that defines the nonlinearity function as
# counts_out = counts_in + beta*counts_in^2.

# reciprocity_alpha - The normalization factor that determines the effect of reciprocity failure
# of the detectors for a given exposure time. - use the algorithm galsim uses, in which
# pR/p = ((p/t)/(p'/t'))^(alpha/log(10)). p'/t' is the flux for which the relation holds - with
# wfirst using base_flux = 1.0, p is response in electrons, t is time; pR is the response if the
# relation fails to hold

# thermal background currently 0.023 e/pix/s except F184 which is 0.179 e/pix/s - add catch for
# R062 and default it to z087 otherwise

# ipc_kernel - The 3x3 kernel to be used in simulations of interpixel capacitance (IPC), using
# galsim.wfirst.applyIPC().

# persistence_coefficients - The retention fraction of the previous eight exposures in a simple,
# linear model for persistence.

# charge_diffusion - The per-axis sigma to use for a Gaussian representing charge diffusion for
# WFIRST.  Units: pixels.

# read noise goes as sqrt(sig_floor**2 + 12 * sig_RN**2 * (N-1) / (N+1) / N) where
# N = t_exp / t_read; if N=100 we get read noise of ~8.5 e- for a floor of 5 e- single RN of 20 e-.


def gridcreate(name, y, x, ratio, z, **kwargs):
    # Function that creates a blank axis canvas; each figure gets a name (or alternatively a number
    # if none is given), and gridspec creates an N*M grid onto which you can create axes for plots.
    # This returns a gridspec "instance" so you can specific which figure to put the axis on if you
    # have several on the go.
    plt.figure(name, figsize=(z*x, z*ratio*y))
    gs = gridspec.GridSpec(y, x, **kwargs)
    return gs


# flat and dark can be loaded from a fits file or found elsewhere, they are simply input
# files to be multipled/added to the original data.
def add_dark(image, dark_current):
    image += dark_current
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


def get_sn_model(sn_type, setflag, t0=0.0, z=0.0):
    # salt2 for Ia, s11-* where * is 2004hx for IIL/P, 2005hm for Ib, and 2006fo for Ic
    # draw salt2 x1 and c from salt2_parameters (gaussian, x1: x0=0.4, sigma=0.9, c: x0=-0.04,
    # sigma = 0.1)
    # Hounsell 2017 gives SALT2 models over a wider wavelength range, given as sncosmo source
    # salt2-h17. both salt2 models have phases -20 to +50 days.
    # above non-salt2 models don't give coverage, so trying new ones from the updated builtin
    # source list...

    if sn_type == 'Ia':
        sn_model = sncosmo.Model('salt2-extended-h17')
        if setflag:
            x1, c = np.random.normal(0.4, 0.9), np.random.normal(-0.04, 0.1)
            sn_model.set(t0=t0, z=z, x1=x1, c=c)
    elif sn_type == 'Ib':
        sn_model = sncosmo.Model('snana-2007nc')
        if setflag:
            sn_model.set(t0=t0, z=z)
    elif sn_type == 'Ic':
        sn_model = sncosmo.Model('snana-2006lc')
        if setflag:
            sn_model.set(t0=t0, z=z)
    elif sn_type == 'IIP' or sn_type == 'II':
        sn_model = sncosmo.Model('snana-2007nv')
        if setflag:
            sn_model.set(t0=t0, z=z)
    elif sn_type == 'IIL':
        sn_model = sncosmo.Model('nugent-sn2l')
        if setflag:
            sn_model.set(t0=t0, z=z)
    # TODO: add galaxy dust via smcosmo.F99Dust([r_v])

    return sn_model


def make_images(filters, pixel_scale, sn_type, times, exptime, filt_zp, psf_comp_filename,
                dark_current, readnoise, t0, lambda_eff):
    nfilts = len(filters)
    ntimes = len(times)

    # assuming surface brightnesses vary between roughly mu_e = 18-23 mag/arcsec^2 (mcgaugh
    # 1995, driver 2005, shen 2003 -- assume shen 2003 gives gaussian with mu=20.94, sigma=0.74)

    mu_0 = np.random.normal(20.94, 0.74)
    # elliptical galaxies approximated as de vaucouler (n=4) sersic profiles, spirals as
    # exponentials (n=1). axial ratios vary 0.5-1 for ellipticals and 0.1-1 for spirals
    rand_num = np.random.uniform()
    n_type = 4 if rand_num < 0.5 else 1
    # randomly draw the eccentricity from 0.5/0.1 to 1, depending on sersic index
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

    psf_comp = np.load(psf_comp_filename)

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
        # random offsets for star should be in arcseconds
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
        if (((((x - p) * np.cos(t) - (y - q) * np.sin(t)) / b)**2 +
             (((x - p) * np.sin(t) + (y - q) * np.cos(t)) / a)**2 <= 1) and
            ((((x - p) * np.cos(t) - (y - q) * np.sin(t)) / b)**2 +
             (((x - p) * np.sin(t) + (y - q) * np.cos(t)) / a)**2 > 0.05)):
            endflag = 1

    sn_model = get_sn_model(sn_type, 1, t0=t0, z=z)
    # pretending that F125W on WFC3/IR is 2MASS J, we set the absolute magnitude of a
    # type Ia supernova to J = -19.0 (meikle 2000). Phillips (1993) also says that ~M_I = -19 --
    # currently just setting absolute magnitudes to -19, but could change if needed
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

    # TODO: see if we can replace this with the galsim.wfirst version
    # given some zodiacal light flux, in ergcm^-2s^-1A^-1arcsec^-2, flip given the ST ZP,
    # then convert back to flux;
    # see http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c09_exposuretime08.html
    zod_flux_st = 2e-18  # erg/cm^2/s/A/arcsec^2; Fl = c/l^2 Fv
    # erg/cm^2/s/Hz/arcsec^2; l_eff in um; final conversion of 1e-10 makes it 1/(s^-1 ang^-1)
    zod_flux = zod_flux_st / (3e8 / (lambda_eff*1e-6)**2 * 1e-10)
    zod_flux *= pixel_scale**2  # erg/cm^2/s/Hz[/pixel]
    zod_mag = -2.5 * np.log10(zod_flux) - 48.6  # AB mag system
    zod_count = 10**(-1/2.5 * (zod_mag - filt_zp[0]))
    # correct the zodiacal light counts for the stray light fraction of the telescope
    zod_count *= (1.0 + wfirst.stray_light_fraction)
    gal_params = [mu_0, n_type, e_disk, pa_disk, half_l_r, offset_r, Vgm_unit, mag]
    # TODO: check if simple half-pixel dither is right and update if not
    second_gal_offets = np.empty((nfilts, 2), float)
    for j in range(0, nfilts):
        # define a random pixel offset ra/dec
        offset_ra, offset_dec = np.random.uniform(0.01, 0.99), np.random.uniform(0.01, 0.99)
        sign = -1 if np.random.uniform(0, 1) < 0.5 else 1
        # non-reference image should be offset by half a pixel, wrapped around [0, 1]
        second_gal_offets[j, 0] = (offset_ra + sign * 0.5 + 1) % 1
        second_gal_offets[j, 1] = (offset_dec + sign * 0.5 + 1) % 1
        image = pmf.mog_galaxy(pixel_scale, filt_zp[j], psf_comp[j], gal_params +
                               [offset_ra, offset_dec])
        q = np.where(image < 0)
        image[q] = 1e-8
        image = add_background(image, zod_count[j])
        image = add_dark(image, dark_current)
        image = set_exptime(image, exptime)
        image = add_poisson(image)
        image = add_read(image, readnoise)

        # second_gal_offset is the pixel offset, relative to the central pixel, of the observation,
        # onto which we should shift the 'reference' frame. we therefore are asking given
        # reference pixel x', what dx do we add such that x' + dx = x? Thus, dx = x - x', or
        # observation pixel minus reference pixel; assume scipy.ndimage.shift correctly spline
        # interpolates the shift as required.
        dx_, dy_ = second_gal_offets[j, 0] - offset_ra, second_gal_offets[j, 1] - offset_dec
        image = shift(image, [dx_, dy_], mode='nearest')
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

            # if cosmicrays are needed then figure out what stips does for that...
            offset_ra, offset_dec = second_gal_offets[j, :]
            image = pmf.mog_galaxy(pixel_scale, filt_zp[j], psf_comp[j],
                                   gal_params + [offset_ra, offset_dec])
            image = pmf.mog_add_psf(image, [rand_ra / pixel_scale, rand_dec / pixel_scale, m_ia],
                                    filt_zp[j], psf_comp[j])
            q = np.where(image < 0)
            image[q] = 1e-8
            image = add_background(image, zod_count[j])
            image = add_dark(image, dark_current)
            image = set_exptime(image, exptime)
            image = add_poisson(image)
            image = add_read(image, readnoise)

            images.append(image)
            image_diff = image - image_shifted
            images_diff.append(image_diff)

            time_array.append(time)
            band_array.append(filters[j])

            x_cent, y_cent = (image.shape[0]-1)/2, (image.shape[1]-1)/2
            xind, yind = np.floor(rand_ra / pixel_scale + x_cent).astype(int), np.floor(rand_dec / pixel_scale + y_cent).astype(int)

            N = 5
            delta = np.arange(-N, N+1e-10, 1)
            p = psf_comp[j].reshape(-1)
            # rand_* is (fractional) pixel offset from centre, so we just modulo 1 to get single
            # pixel fraction
            dx, dy = (rand_ra/pixel_scale) % 1, (rand_dec/pixel_scale) % 1
            psf_box_sum = np.sum(pmf.psf_fit_fun(p, delta+dx, delta+dy))
            # current naive sum the entire (box) 'aperture' flux of the Sn, correcting for
            # exposure time in both counts and uncertainty; also have to correct for the lost flux
            # outside of the box
            xind0, xind1 = max(0, xind-N), min(image_diff.shape[0], xind+N+1)
            yind0, yind1 = max(0, yind-N), min(image_diff.shape[1], yind+N+1)
            diff_sum = np.sum(image_diff[xind0:xind1, yind0:yind1]) / exptime / psf_box_sum
            diff_sum_err = np.sqrt(np.sum(image[xind0:xind1, yind0:yind1] +
                                          image_shifted[xind0:xind1, yind0:yind1])) / \
                exptime/psf_box_sum
            flux_array.append(diff_sum)
            fluxerr_array.append(diff_sum_err)
            zp_array.append(filt_zp[j])  # filter-specific zeropoint
            zpsys_array.append('ab')

            true_flux.append(10**(-1/2.5 * (m_ia - filt_zp[j])))
        images_with_sn.append(images)
        diff_images.append(images_diff)

    lc_data = [np.array(time_array), np.array(band_array), np.array(flux_array),
               np.array(fluxerr_array), np.array(zp_array), np.array(zpsys_array)]
    true_flux = np.array(true_flux)

    param_names = ['z', 't0']
    if sn_type == 'Ia':
        param_names += ['x0', 'x1', 'c']
    else:
        param_names += ['amplitude']
    sn_params = [sn_model[q] for q in param_names]

    return images_with_sn, images_without_sn, diff_images, lc_data, sn_params, true_flux


def make_fluxes(filters, sn_type, times, filt_zp, t0, exptime, psf_r):
    nfilts = len(filters)
    ntimes = len(times)

    # redshift randomly drawn between two values uniformly
    z_low, z_high = 0.2, 1.0
    z = np.random.uniform(z_low, z_high)

    sn_model = get_sn_model(sn_type, 1, t0=t0, z=z)
    # pretending that F125W on WFC3/IR is 2MASS J, we set the absolute magnitude of a
    # type Ia supernova to J = -19.0 (meikle 2000). Phillips (1993) also says that ~M_I = -19 --
    # currently just setting absolute magnitudes to -19, but could change if needed
    sn_model.set_source_peakabsmag(-19.0, 'f125w', 'ab')

    # things that are needed to create the astropy.table.Table for use in fit_lc:
    # time, band (name, see registered bandpasses), flux, fluxerr [both just derived from an
    # image somehow], zp, zpsys [zeropoint and name of system]

    time_array = []
    band_array = []
    flux_array = []
    fluxerr_array = []
    zp_array = []
    zpsys_array = []

    true_flux = []
    for k in range(0, ntimes):
        for j in range(0, nfilts):
            if filters[j] == 'F184':
                bkg = np.random.uniform(1, 3)
            else:
                bkg = np.random.uniform(0.3, 0.7)
            # TODO: add exposure and readout time so that exposures are staggered in time
            time = times[k] + t0

            # get the apparent magnitude of the supernova at a given time; first get the
            # appropriate filter for the observation
            bandpass = sncosmo.get_bandpass(filters[j])
            # time should be in days
            m_ia = sn_model.bandmag(bandpass, magsys='ab', time=time)
            if np.isnan(m_ia):
                m_ia = -2.5 * np.log10(0.01) + filt_zp[j]

            t_f = 10**(-1/2.5 * (m_ia - filt_zp[j]))

            # noise floor of, say, 0.5% photometry in quadrature with shot noise and background
            # counts in e/s/pixel, assuming a WFIRST aperture size of psf_r pix, or ~pi r^2 pixels,
            # remembering to correct for the fact that uncertainties in fluxes are really done in
            # photon counts, so multiply then divide by exptime
            flux_err = np.sqrt(np.sqrt(t_f * exptime)**2 + (0.005 * t_f * exptime)**2 +
                               np.sqrt(bkg * exptime * np.pi * psf_r**2)**2) / exptime
            flux = np.random.normal(loc=t_f, scale=flux_err)
            time_array.append(time)
            band_array.append(filters[j])
            flux_array.append(flux)
            fluxerr_array.append(flux_err)
            zp_array.append(filt_zp[j])  # filter-specific zeropoint
            zpsys_array.append('ab')

            true_flux.append(t_f)

    lc_data = [np.array(time_array), np.array(band_array), np.array(flux_array),
               np.array(fluxerr_array), np.array(zp_array), np.array(zpsys_array)]
    true_flux = np.array(true_flux)

    param_names = ['z', 't0']
    if sn_type == 'Ia':
        param_names += ['x0', 'x1', 'c']
    else:
        param_names += ['amplitude']
    sn_params = np.array([sn_model[q] for q in param_names])

    return lc_data, sn_params, true_flux


@profile
def fit_lc(lc_data, sn_types, directory, filters, figtext, ncol, minsnr, sn_priors,
           filt_zp, make_fit_figs, multi_z_fit, type_ind, sn_params):
    x2s = np.empty(len(sn_types), float)
    bestfit_models = []
    bestfit_results = []
    largest_z = 1.5
    dz = 0.01
    min_counts = 0.0001

    for i, sn_type in enumerate(sn_types):
        params = ['t0']
        if sn_type == 'Ia':
            params += ['x0', 'x1', 'c']
        else:
            params += ['amplitude']
        sn_model = get_sn_model(sn_type, 0)

        # place upper limits on the redshift probeable, by finding the z at which each filter drops
        # out of being in overlap with the model
        z_upper_band = np.empty(len(filters), float)
        for p in range(0, len(filters)):
            z = 0
            while sn_model.bandoverlap(filters[p], z=z):
                z += dz
                if z > largest_z:
                    break  # otherwise this will just keep going forever for very red filters
            z_upper_band[p] = min(largest_z, z - dz)
        z_upper_count = np.empty(len(filters), float)
        z_lower_count = np.empty(len(filters), float)
        # the lower limits on z -- for this model -- are, assuming a minsnr detection in that
        # filter, a model flux in the given system of, say, 0.0001 counts/s; a very low goal, but
        # one that avoids bluer SNe being selected when they would drop out of the detection. Also
        # avoids models from failing to calculate an amplitude... Similarly, we can calculate the
        # maximum redshift for a blue filter to have a "detection". If there is no detection in
        # this filter, we set the redshift range to its maximum to remove the filter from
        # consideration.
        for p in range(0, len(filters)):
            z_array = np.arange(0, z_upper_band[p]+1e-10, dz)
            snr_filt = lc_data['flux'].data[p] / lc_data['fluxerr'].data[p]
            if snr_filt < minsnr:
                z_upper_count[p] = z_array[-1]
                z_lower_count[p] = z_array[0]
                continue
            countrate = np.empty_like(z_array)
            for q, z_init in enumerate(z_array):
                sn_model.set(z=z_init)
                countrate[q] = sn_model.bandflux(filters[p], time=0, zp=filt_zp[p], zpsys='ab')
            z_upper_count[p] = z_array[np.where(countrate > min_counts)[0][-1]]
            z_lower_count[p] = z_array[np.where(countrate > min_counts)[0][0]]
        # set the bounds on z to be at most the smallest of those available by the given filters in
        # the set being fit here
        z_min = np.amax(z_lower_count)
        z_max = min(np.amin(z_upper_band), np.amin(z_upper_count))
        bounds = {}
        # x1 and c bounded by 3.5-sigma regions (x1: mu=0.4, sigma=0.9, c: mu=-0.04, sigma = 0.1)
        if sn_type == 'Ia':
            bounds.update({'x1': (-2.75, 3.55), 'c': (-0.39, 0.31)})
        bounds.update({'z': (z_min, z_max)})
        params += ['z']

        if multi_z_fit:
            result = None
            fitted_model = None
            for z_init in np.linspace(z_min, z_max, 10):
                sn_model.set(z=z_init)
                result_temp, fitted_model_temp = sncosmo.fit_lc(lc_data, sn_model, params,
                                                                bounds=bounds, minsnr=minsnr,
                                                                guess_z=False)
                if result is None or result_temp.chisq < result.chisq:
                    result = result_temp
                    fitted_model = fitted_model_temp
        else:
            fitted_model = sn_model

        # after a round of minimising the lightcurve at fixed redshifts, add redshift to allow a
        # final fit of the model to the data
        guess_z = True if fitted_model is sn_model else False
        result, fitted_model = sncosmo.fit_lc(lc_data, fitted_model, params, bounds=bounds,
                                              minsnr=minsnr, guess_z=guess_z)
        bestfit_models.append(fitted_model)
        bestfit_results.append(result)
        try:
            x2s[i] = result.chisq
        except AttributeError:
            x2s[i] = sncosmo.chisq(lc_data, fitted_model)

    # TODO: add a fire extinguisher null hypothesis probability properly
    # if this was p = p(m) * l(d) then ln(p) = ln(p(m)) - x2/2; fire extinguisher probability
    # becomes -2 ln(f) = 32 for p_f ~ 1e-7
    log_fire = -2 * np.log(1e-7)
    # probs = np.append(sn_priors*np.exp(-0.5 * x2s), 1e-7)
    # probs /= np.sum(probs)
    # prob = probs[0] if sn_types[type_ind] == 'Ia' else 1 - probs[0]
    # lnprob = np.log(prob)
    


    if make_fit_figs:
        sse.make_fit_fig(directory, sn_types, probs, x2s, lc_data, ncol, bestfit_results,
                         bestfit_models, figtext)

    return lnprob


@profile
def run_filt_cadence_combo(directory, sn_types, filters, pixel_scale, times, exptime, filt_zp,
                           psf_comp_filename, dark_current, readnoise, t0, lambda_eff,
                           make_sky_figs, make_fit_figs, make_flux_figs, image_flag,
                           multi_z_fit, psf_r, t_interval, n_obs):
    # only consider sources out to ~100 days
    t_low, t_high = 0, (n_obs - 1) * t_interval
    times = np.arange(t_low, min(100, t_high)+1e-10, t_interval) + \
        np.random.uniform(-t_interval, t_interval)
    if len(filters) * len(times) <= 5:
        return -np.inf

    type_ind = np.random.choice(len(sn_types))
    if image_flag:
        images_with_sn, images_without_sn, diff_images, lc_data, sn_params, true_flux = \
            make_images(filters, pixel_scale, sn_types[type_ind], times, exptime, filt_zp,
                        psf_comp_filename, dark_current, readnoise, t0, lambda_eff)
    else:
        lc_data, sn_params, true_flux = make_fluxes(filters, sn_types[type_ind], times,
                                                    filt_zp, t0, exptime, psf_r)

    if make_sky_figs and image_flag:
        sse.make_figures(images_with_sn, images_without_sn, diff_images, filters, times,
                         exptime)

    lc_data_table = Table(data=lc_data,
                          names=['time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'])
    if not np.amax(lc_data_table['flux'].data / lc_data_table['fluxerr'].data) >= minsnr:
        return -np.inf

    figtext = []
    if sn_types[type_ind] == 'Ia':
        z_, t_, x0_, x1_, c_ = sn_params
        figtext.append('Type {}: $z = {:.3f}$\n$t_0 = {:.1f}$\n'
                       '$x_0 = {:.5f}$'.format(sn_types[type_ind], z_, t_, x0_))
        figtext.append('$x_1 = {:.5f}$\n$c = {:.5f}$'.format(x1_, c_))
    else:
        z_ = sn_params[0]
        t_ = sn_params[1]
        A_ = sn_params[2]
        A_sig = int(np.floor(np.log10(abs(A_))))
        figtext.append('Type {}: $z = {:.3f}$\n$t_0 = {:.1f}$'.format(
                       sn_types[type_ind], z_, t_))
        figtext.append('$A = {:.3f} \\times 10^{{{}}}$'.format(A_/10**A_sig, A_sig))

    lnprob = fit_lc(lc_data_table, sn_types, directory, filters, figtext, ncol, minsnr,
                    sn_priors, filt_zp, make_fit_figs, multi_z_fit, type_ind, sn_params)

    if make_flux_figs:
        gs = gridcreate('09', 1, 1, 0.8, 5)
        ax = plt.subplot(gs[0])
        for c, filter_ in zip(colours, filters):
            q = lc_data_table['band'] == filter_
            ax.errorbar(lc_data_table['time'][q], (lc_data_table['flux'][q] - true_flux[q]) /
                        true_flux[q], yerr=lc_data_table['fluxerr'][q]/true_flux[q],
                        fmt='{}.'.format(c), label=filter_)
        ax.legend(shadow=False, framealpha=0)
        ax.axhline(0, c='k', ls='--')
        ax.set_xlabel('Time')
        ax.set_ylabel('Flux difference (fit - true)/true')
        plt.tight_layout()
        plt.savefig('{}/flux_ratio.pdf'.format(directory))

    # if we're fitting a true Ia then we want the probability to be high, otherwise it should
    # be low; the probability is P(Ia|D) = P(Ia) * p(D|Ia) / sum_j P(j) p(D|j) or
    # P(CC|D) = 1 - P(Ia|D)
    return lnprob


if __name__ == '__main__':
    # run_mins = 20/60
    # sse.model_number(run_mins, ngals)

    # sys.exit()

    directory = 'out_gals'
    psf_comp_filename = '../PSFs/wfirst_psf_comp.npy'

    filters_master = np.array(['z087', 'y106', 'w149', 'j129', 'h158', 'f184'])  # 'r062'
    colours_master = np.array(['k', 'r', 'b', 'g', 'c', 'm', 'orange'])

    # 1 count/s for infinite aperture, hounsell17, AB magnitudes
    # get r062 ZP if added; microsit disagrees on h158 by ~0.03 mags - additional microsit ZPs are
    # [26.39 r062, 27.50 w149 mask, 27.61 no mask w149, 25.59 k208] (26.30 -- j or h?)
    filt_zp_master = np.array([26.39, 26.41, 27.50, 26.35, 26.41, 25.96])
    lambda_eff_master = np.array([0.601, 0.862, 1.045, 1.251, 1.274, 1.555, 1.830])
    sse.register_filters(filters_master)

    # dark current and read noise from the GalSim instrument; read noise is in pure e-, but
    # the current is e-/pixel/s, so requires correcting by exposure time
    readnoise, dark_current = wfirst.read_noise, wfirst.dark_current
    pixel_scale = wfirst.pixel_scale  # arcsecond/pixel

    t0, minsnr, ncol = 50000, 5, min(3, len(filters_master))

    sn_priors = sse.get_sn_priors()
    sn_types = ['Ia', 'Ib', 'Ic', 'II']

    if len(glob.glob('{}/*/*.pdf'.format(directory))) > 0:
        os.system('rm -r {}'.format(directory))
        os.makedirs(directory)

    t_intervals, n_obss = [20], [3]
    sub_inds_combos = [[0, 3, 4]]

    dark = 0.015  # e/s/pixel
    psf_r = 3  # pixel
    snr_det = 5  # minimum SNR to consider a detection, given source and background noise
    rnoise = 20  # readout noise RMS

    # sse.faintest_sn(sn_types, filters_master, exptime, filt_zp_master, snr_det, psf_r, rnoise,
    #                 dark)
    # sys.exit()

    make_sky_figs, make_fit_figs, make_flux_figs, image_flag = False, True, False, False
    multi_z_fit = True

    exptime, t_interval, n_obs = 100, 20, 3

    for sub_inds in sub_inds_combos:

        filters = filters_master[sub_inds]
        filt_zp = filt_zp_master[sub_inds]
        colours = colours_master[sub_inds]
        lambda_eff = lambda_eff_master[sub_inds]

        if not os.path.exists(directory):
            os.makedirs(directory)

        start = timeit.default_timer()
        ln_prob = run_filt_cadence_combo(directory, sn_types, filters,
                                         pixel_scale, exptime, filt_zp, psf_comp_filename,
                                         dark_current, readnoise, t0, lambda_eff, make_sky_figs,
                                         make_fit_figs, make_flux_figs, image_flag,
                                         multi_z_fit, psf_r, t_interval, n_obs)
        time = '{:.0f}s'.format(timeit.default_timer()-start)
