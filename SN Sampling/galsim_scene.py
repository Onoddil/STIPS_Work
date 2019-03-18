import sys
import os
import math
import logging
import galsim
import galsim.wfirst as wfirst
from webbpsf import wfirst as wfirst_wpsf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.visualization import simple_norm
import numpy as np
from scipy.special import gammaincinv
import sncosmo
import astropy.io.fits as pyfits
import astropy.units as u


def gridcreate(name, y, x, ratio, z, **kwargs):
    # Function that creates a blank axis canvas; each figure gets a name (or alternatively a number
    # if none is given), and gridspec creates an N*M grid onto which you can create axes for plots.
    # This returns a gridspec "instance" so you can specific which figure to put the axis on if you
    # have several on the go.
    plt.figure(name, figsize=(z*x, z*ratio*y))
    gs = gridspec.GridSpec(y, x, **kwargs)
    return gs


def get_sn_model(sn_type, setflag, t0=0.0, z=0.0):
    # salt2 for Ia, s11-* where * is 2004hx for IIL/P, 2005hm for Ib, and 2006fo for Ic
    # draw salt2 x1 and c from salt2_parameters (gaussian, x1: x0=0.4, sigma=0.9, c: x0=-0.04,
    # sigma = 0.1)
    # Hounsell 2017 gives SALT2 models over a wider wavelength range, given as sncosmo source
    # salt2-h17. both salt2 models have phases -20 to +50 days.
    # above non-salt2 models don't give coverage, so trying new ones from the updated builtin
    # source list...

    if sn_type == 'Ia':
        sn_model = sncosmo.Model('salt2-h17')
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
        sn_model = sncosmo.Model('nugent-sn21')
        if setflag:
            sn_model.set(t0=t0, z=z)
    # TODO: add galaxy dust via smcosmo.F99Dust([r_v])

    return sn_model


def main(argv):
    # Default is to use all filters.  Specify e.g. 'YJH' to only do Y106, J129, and H158.
    use_filters = None

    # Initialize (pseudo-)random number generator.
    random_seed = 123456
    rng = galsim.BaseDeviate(random_seed)

    # Generate a Poisson noise model.
    poisson_noise = galsim.PoissonNoise(rng)

    # Read in the WFIRST filters, setting an AB zeropoint appropriate for this telescope given its
    # diameter and (since we didn't use any keyword arguments to modify this) using the typical
    # exposure time for WFIRST images.  By default, this routine truncates the parts of the
    # bandpasses that are near 0 at the edges, and thins them by the default amount.
    wfirst_filters = wfirst.getBandpasses(AB_zeropoint=True)
    psfs = []
    filters = np.array(['z087', 'y106', 'w149', 'j129', 'h158', 'f184'])  # 'r062'
    use_SCA = 7  # This could be any number from 1...18
    remake_psfs = False
    for filter_ in filters:
        if remake_psfs or not os.path.exists('psf_fit/{}.fits'.format(filter_)):
            wfi = wfirst_wpsf.WFI()
            wfi.filter = filter_
            wfi.detector = 'SCA09'
            # position can vary 4 - 4092, allowing for a 4 pixel gap
            wfi.detector_position = (2048, 2048)
            wfi.options['parity'] = 'odd'
            wfi.options['output_mode'] = 'both'
            psf_ = wfi.calc_psf(oversample=10)
            psf_.writeto('psf_fit/{}.fits'.format(filter_), overwrite=True)
            psf_ = psf_[0]
        else:
            psf_ = pyfits.open('psf_fit/{}.fits'.format(filter_))[0]
        psfs.append(galsim.InterpolatedImage(galsim.Image(psf_.data),
                    scale=psf_.header['PIXELSCL']))

    # Define the size of the postage stamp that we use for each individual galaxy within the larger
    # image, and for the PSF images.
    stamp_size = 100

    # We choose a particular (RA, dec) location on the sky for our observation.
    ra_targ = 90.*galsim.degrees
    dec_targ = -10.*galsim.degrees
    targ_pos = galsim.CelestialCoord(ra=ra_targ, dec=dec_targ)
    # Get the WCS for an observation at this position.  We are not supplying a date, so the routine
    # will assume it's the vernal equinox.  We are also not supplying a position angle for the
    # observatory, which means that it will just find the optimal one (the one that has the solar
    # panels pointed most directly towards the Sun given this targ_pos and date).  The output of
    # this routine is a dict of WCS objects, one for each SCA.  We then take the WCS for the SCA
    # that we are using.
    wcs_list = wfirst.getWCS(world_pos=targ_pos, SCAs=use_SCA)
    wcs = wcs_list[use_SCA]
    # We need to find the center position for this SCA.  We'll tell it to give us a CelestialCoord
    # corresponding to (X, Y) = (wfirst.n_pix/2, wfirst.n_pix/2).
    SCA_cent_pos = wcs.toWorld(galsim.PositionD(wfirst.n_pix/2, wfirst.n_pix/2))

    bulge_n = 3.5          #
    bulge_re = 2.3         # arcsec
    disk_n = 1.5           #
    disk_re = 3.7          # arcsec
    bulge_frac = 0.3       #
    gal_q = 0.73           # (axis ratio 0 < q < 1)
    pa_disk = 23           # degrees (position angle on the sky)

    filt_zp = np.array([26.39, 26.41, 27.50, 26.35, 26.41, 25.96])
    # assuming surface brightnesses vary between roughly mu_e = 18-23 mag/arcsec^2 (mcgaugh
    # 1995, driver 2005, shen 2003 -- assume shen 2003 gives gaussian with mu=20.94, sigma=0.74)

    mu_0 = np.random.normal(20.94, 0.74)
    # elliptical galaxies approximated as de vaucouler (n=4) sersic profiles, spirals as
    # exponentials (n=1). axial ratios vary 0.5-1 for ellipticals and 0.1-1 for spirals
    rand_num = np.random.uniform()
    n_type = 4 if rand_num < 0.5 else 1
    # randomly draw the eccentricity from 0.5/0.1 to 1, depending on sersic index
    e_disk = np.random.uniform(0.5 if n_type == 4 else 0.1, 1.0)
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
    # 0.75 mag is really 2.5 * log10(2), for double flux, given area is half-light radius
    mag = mu_0 - 2.5 * np.log10(np.pi * half_l_r**2 * e_disk) - 2.5 * np.log10(2)

    # TODO: set properly
    exptime = 100

    # total flux in galaxy -- ensure that all units end up in flux as counts/s accordingly
    Sg = 10**(-1/2.5 * (mag - filt_zp[np.where(filters == 'f184')[0][0]]))

    # Define the galaxy profile.
    # Normally Sersic profiles are specified by half-light radius, the radius that
    # encloses half of the total flux.  However, for some purposes, it can be
    # preferable to instead specify the scale radius, where the surface brightness
    # drops to 1/e of the central peak value.
    bulge = galsim.Sersic(bulge_n, half_light_radius=bulge_re)
    disk = galsim.Sersic(disk_n, half_light_radius=disk_re)
    # Objects may be multiplied by a scalar (which means scaling the flux) and also
    # added to each other.
    gal = (bulge_frac * bulge + (1-bulge_frac) * disk).withFlux(Sg)
    # Set the shape of the galaxy according to axis ratio and position angle
    # Note: All angles in GalSim must have explicit units.  Options are:
    #       galsim.radians
    #       galsim.degrees
    #       galsim.arcmin
    #       galsim.arcsec
    #       galsim.hours
    gal_shape = galsim.Shear(q=gal_q, beta=pa_disk*galsim.degrees)
    gal = gal.shear(gal_shape)

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
    rand_dx = rand_ra / wfirst.pixel_scale
    rand_dy = rand_dec / wfirst.pixel_scale

    gs = gridcreate('1', 2, 3, 0.8, 5)
    # Calculate the sky level for each filter, and draw the PSF and the galaxies through the
    # filters.
    t0 = 0
    sn_type = 'Ia'
    sn_model = get_sn_model(sn_type, 1, t0=t0, z=z)
    # pretending that F125W on WFC3/IR is 2MASS J, we set the absolute magnitude of a
    # type Ia supernova to J = -19.0 (meikle 2000). Phillips (1993) also says that ~M_I = -19 --
    # currently just setting absolute magnitudes to -19, but could change if needed
    sn_model.set_source_peakabsmag(-19.0, 'f125w', 'ab')
    for j in range(0, len(filters)):
        f = pyfits.open('../../webbpsf-data/WFI/filters/{}_throughput.fits'.format(
                        filters[j].upper()))
        data = f[1].data
        dispersion = np.array([d[0] * 1e-4 for d in data])
        transmission = np.array([d[1] * 0.95 for d in data])
        # both F184 and W149 extend 0.004 microns into 2 microns, beyond the wavelength range of
        # the less extended models, 19990A, or 1.999 microns. Thus we slightly chop the ends off
        # these filters, and set the final 'zero' to 1.998 microns:
        if filters[j] == 'f184' or filters[j] == 'w149':
            ind_ = np.where(dispersion < 1.999)[0][-1]
            dispersion[ind_+1] = 1.9998
            dispersion[ind_+2] = 1.99985
        q_ = np.argmax(transmission)
        if transmission[q_] == transmission[q_+1]:
            q_ += 1
        imin = np.where(transmission[:q_] == 0)[0][-1]
        imax = np.where(transmission[q_:] == 0)[0][0] + q_ + 1
        bandpass = sncosmo.Bandpass(dispersion[imin:imax], transmission[imin:imax],
                                    wave_unit=u.micron, name=filters[j])
        sncosmo.register(bandpass)
    for i, (filter_name, filter_) in enumerate(wfirst_filters.items()):
        if use_filters is not None and filter_name[0] not in use_filters:
            continue
        print('Beginning filter {0}.'.format(filter_name))

        time = t0

        # get the apparent magnitude of the supernova at a given time; first get the
        # appropriate filter for the observation
        bandpass = sncosmo.get_bandpass(filter_name)
        # time should be in days
        m_ia = sn_model.bandmag(bandpass, magsys='ab', time=time)
        if np.isnan(m_ia):
            m_ia = -2.5 * np.log10(0.01) + filt_zp[i]
        sn_flux = 10**(-1/2.5 * (m_ia - filt_zp[i])) * wfirst.gain * exptime

        # Set up the full image that will contain all the individual galaxy images, with information
        # about WCS:
        final_image = galsim.ImageF(stamp_size, stamp_size, wcs=wcs)

        # Drawing PSF.  Note that the PSF object intrinsically has a flat SED, so if we convolve it
        # with a galaxy, it will properly take on the SED of the galaxy.  For the sake of the demo,
        # we will simply convolve with a 'star' that has a flat SED and unit flux in this band, so
        # that the PSF image will be normalized to unit flux. This does mean that the PSF image
        # being drawn here is not quite the right PSF for the galaxy.  Indeed, the PSF for the
        # galaxy effectively varies within it, since it differs for the bulge and the disk.  To
        # make a real image, one would have to choose SEDs for stars and convolve with a star that
        # has a reasonable SED, but we just draw with a flat SED for this demo.

        # Generate a point source.
        point = galsim.DeltaFunction(flux=1.)
        # Use a flat SED here, but could use something else.  A stellar SED for instance.
        # Or a typical galaxy SED.  Depending on your purpose for drawing the PSF. Give it unit
        # flux in this filter.
        star_sed = galsim.SED(lambda x: 1, 'nm', 'flambda').withFlux(1., filter_)
        star = galsim.Convolve(point*star_sed, psfs[i])
        img_psf = galsim.ImageF(stamp_size, stamp_size)
        star.drawImage(bandpass=filter_, image=img_psf, scale=wfirst.pixel_scale)
        ix = int(math.floor(stamp_size/2+rand_dx+0.5))
        iy = int(math.floor(stamp_size/2+rand_dy+0.5))
        stamp_bounds = galsim.BoundsI(ix-0.5*stamp_size, ix+0.5*stamp_size-1,
                                      iy-0.5*stamp_size, iy+0.5*stamp_size-1)
        bounds = stamp_bounds & final_image.bounds
        final_image[bounds] += sn_flux * img_psf[bounds]

        stamp = galsim.Convolve([psfs[i], gal])
        dx_gal, dy_gal = np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)
        # Account for the fractional part of the position:
        ix = int(math.floor(stamp_size/2+dx_gal+0.5))
        iy = int(math.floor(stamp_size/2+dy_gal+0.5))
        img_gal = galsim.ImageF(stamp_size, stamp_size)
        stamp.drawImage(image=img_gal, scale=wfirst.pixel_scale)  # bandpass=filter_,
        stamp_bounds = galsim.BoundsI(ix-0.5*stamp_size, ix+0.5*stamp_size-1,
                                      iy-0.5*stamp_size, iy+0.5*stamp_size-1)
        bounds = stamp_bounds & final_image.bounds
        final_image[bounds] += img_gal[bounds]

        # First we get the amount of zodaical light for a position corresponding to the center of
        # this SCA.  The results are provided in units of e-/arcsec^2, using the default WFIRST
        # exposure time since we did not explicitly specify one.  Then we multiply this by a factor
        # >1 to account for the amount of stray light that is expected.  If we didn't provide a
        # date for the observation, then it will assume that it's the vernal equinox (sun at (0,0)
        # in ecliptic coordinates) in 2025.
        sky_level = wfirst.getSkyLevel(wfirst_filters[filter_name], world_pos=SCA_cent_pos)
        sky_level *= (1.0 + wfirst.stray_light_fraction)
        # Make a image of the sky that takes into account the spatially variable pixel scale.  Note
        # that makeSkyImage() takes a bit of time.  If you do not care about the variable pixel
        # scale, you could simply compute an approximate sky level in e-/pix by multiplying
        # sky_level by wfirst.pixel_scale**2, and add that to final_image.
        sky_image = final_image.copy()
        wcs.makeSkyImage(sky_image, sky_level)
        # This image is in units of e-/pix.  Finally we add the expected thermal backgrounds in
        # this band.  These are provided in e-/pix/s, so we have to multiply by the exposure time.
        sky_image += wfirst.thermal_backgrounds[filter_name]*exptime
        # Adding sky level to the image.
        final_image += sky_image

        # Now that all sources of signal (from astronomical objects and background) have been added
        # to the image, we can start adding noise and detector effects.  There is a utility,
        # galsim.wfirst.allDetectorEffects(), that can apply ALL implemented noise and detector
        # effects in the proper order.  Here we step through the process and explain these in a bit
        # more detail without using that utility.

        # First, we include the expected Poisson noise:
        final_image.addNoise(poisson_noise)

        # The subsequent steps account for the non-ideality of the detectors.

        # 1) Reciprocity failure:
        # Reciprocity, in the context of photography, is the inverse relationship between the
        # incident flux (I) of a source object and the exposure time (t) required to produce a given
        # response(p) in the detector, i.e., p = I*t. However, in NIR detectors, this relation does
        # not hold always. The pixel response to a high flux is larger than its response to a low
        # flux. This flux-dependent non-linearity is known as 'reciprocity failure', and the
        # approximate amount of reciprocity failure for the WFIRST detectors is known, so we can
        # include this detector effect in our images.

        # If we had wanted to, we could have specified a different exposure time than the default
        # one for WFIRST, but otherwise the following routine does not take any arguments.
        wfirst.addReciprocityFailure(final_image)

        # At this point in the image generation process, an integer number of photons gets
        # detected, hence we have to round the pixel values to integers:
        final_image.quantize()

        # 2) Adding dark current to the image:
        # Even when the detector is unexposed to any radiation, the electron-hole pairs that
        # are generated within the depletion region due to finite temperature are swept by the
        # high electric field at the junction of the photodiode. This small reverse bias
        # leakage current is referred to as 'dark current'. It is specified by the average
        # number of electrons reaching the detectors per unit time and has an associated
        # Poisson noise since it is a random event.
        dark_current = wfirst.dark_current*wfirst.exptime
        dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(rng, dark_current))
        final_image.addNoise(dark_noise)

        # NOTE: Sky level and dark current might appear like a constant background that can be
        # simply subtracted. However, these contribute to the shot noise and matter for the
        # non-linear effects that follow. Hence, these must be included at this stage of the
        # image generation process. We subtract these backgrounds in the end.

        # 3) Applying a quadratic non-linearity:
        # In order to convert the units from electrons to ADU, we must use the gain factor. The gain
        # has a weak dependency on the charge present in each pixel. This dependency is accounted
        # for by changing the pixel values (in electrons) and applying a constant nominal gain
        # later, which is unity in our demo.

        # Apply the WFIRST nonlinearity routine, which knows all about the nonlinearity expected in
        # the WFIRST detectors.
        wfirst.applyNonlinearity(final_image)
        # Note that users who wish to apply some other nonlinearity function (perhaps for other NIR
        # detectors, or for CCDs) can use the more general nonlinearity routine, which uses the
        # following syntax:
        # final_image.applyNonlinearity(NLfunc=NLfunc)
        # with NLfunc being a callable function that specifies how the output image pixel values
        # should relate to the input ones.

        # 4) Including Interpixel capacitance:
        # The voltage read at a given pixel location is influenced by the charges present in the
        # neighboring pixel locations due to capacitive coupling of sense nodes. This interpixel
        # capacitance effect is modeled as a linear effect that is described as a convolution of a
        # 3x3 kernel with the image.  The WFIRST IPC routine knows about the kernel already, so the
        # user does not have to supply it.
        wfirst.applyIPC(final_image)

        # 5) Adding read noise:
        # Read noise is the noise due to the on-chip amplifier that converts the charge into an
        # analog voltage.  We already applied the Poisson noise due to the sky level, so read noise
        # should just be added as Gaussian noise:
        read_noise = galsim.GaussianNoise(rng, sigma=wfirst.read_noise)
        final_image.addNoise(read_noise)

        # We divide by the gain to convert from e- to ADU. Currently, the gain value in the WFIRST
        # module is just set to 1, since we don't know what the exact gain will be, although it is
        # expected to be approximately 1. Eventually, this may change when the camera is assembled,
        # and there may be a different value for each SCA. For now, there is just a single number,
        # which is equal to 1.
        final_image /= (wfirst.gain * exptime)

        # Finally, the analog-to-digital converter reads in an integer value.
        final_image.quantize()
        # Note that the image type after this step is still a float.  If we want to actually
        # get integer values, we can do new_img = galsim.Image(final_image, dtype=int)

        # Since many people are used to viewing background-subtracted images, we provide a
        # version with the background subtracted (also rounding that to an int).
        sky_image.quantize()
        tot_sky_image = (sky_image + round(dark_current))/wfirst.gain
        tot_sky_image.quantize()
        final_image -= tot_sky_image

        ax = plt.subplot(gs[i])
        norm = simple_norm(final_image.array, 'log', percent=100)
        img = ax.imshow(final_image.array, cmap='viridis', norm=norm, origin='lower')
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label(r'Flux / e$^-\,\mathrm{s}^{-1}$')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')
    plt.tight_layout()
    plt.savefig('galsim_test.pdf')


if __name__ == "__main__":
    main(sys.argv)
