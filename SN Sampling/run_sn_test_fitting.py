import numpy as np
import sncosmo
import warnings
from astropy.table import Table
from matplotlib import gridspec
import matplotlib.pyplot as plt
from scipy.stats import norm
import sn_sampling_extras as sse
from scipy.special import erf
from scipy.optimize import minimize

import signal
if signal.getsignal(signal.SIGHUP) != signal.SIG_DFL:
    import matplotlib
    matplotlib.use('Agg')  # change to non-interactive backend if nohup mode used


def fun_mle_gauss(p, x):
    u = p[0]
    c = np.abs(p[1])
    N = len(x)
    ln_prob = -N * np.log(np.sqrt(2*np.pi) * c) - np.sum((x - u)**2) / (2 * c**2)
    dlnfdu = np.sum(x - u) / c**2
    dlnfdc = -N / c + np.sum((x - u)**2) / c**3
    return -2 * ln_prob, -2 * np.array([dlnfdu, dlnfdc])


def hess_mle_gauss(p, x):
    u = p[0]
    c = np.abs(p[1])
    N = len(x)
    d2lnfdu2 = -N / c**2
    d2lnfdc2 = N / c**2 - 3 / c**4 * np.sum((x - u)**2)
    d2lnfdudc = -2 / c**3 * np.sum(x - u)
    return -2 * np.array([[d2lnfdu2, d2lnfdudc], [d2lnfdudc, d2lnfdc2]])


def sumgauss(p, x, y, o, dx):
    return np.sum((y - fitgauss(p, x, dx))**2 / o**2)


def fitgauss(p, x, dx):
    u = p[0]
    c = p[1]
    return 1/(2 * dx) * (erf((x+dx - u)/(np.sqrt(2) * c)) - erf((x - u)/(np.sqrt(2) * c)))


def gradgauss(p, x, y, o, dx):
    u = p[0]
    c = p[1]
    dzdu = -1/dx * 1/(np.sqrt(2) * np.pi) / c * (
        np.exp(-0.5 * (x + dx - u)**2 / c**2) - np.exp(-0.5 * (x - u)**2 / c**2))
    dzdc = -1/dx * 1/(np.sqrt(2) * np.pi) / c**2 * (
        (x + dx - u) * np.exp(-0.5 * (x + dx - u)**2 / c**2) -
        (x - u) * np.exp(-0.5 * (x - u)**2 / c**2))
    dfdu = np.sum(-2 * (y - fitgauss(p, x, dx)) / o**2 * dzdu)
    dfdc = np.sum(-2 * (y - fitgauss(p, x, dx)) / o**2 * dzdc)
    return np.array([dfdu, dfdc])


def gridcreate(name, y, x, ratio, z, **kwargs):
    # Function that creates a blank axis canvas; each figure gets a name (or alternatively a number
    # if none is given), and gridspec creates an N*M grid onto which you can create axes for plots.
    # This returns a gridspec "instance" so you can specific which figure to put the axis on if you
    # have several on the go.
    plt.figure(name, figsize=(z*x, z*ratio*y))
    gs = gridspec.GridSpec(y, x, **kwargs)
    return gs


def make_fluxes(filters, times, filt_zp, t0, exptime, psf_r, dark, readnoise):
    nfilts = len(filters)
    ntimes = len(times)

    # redshift randomly drawn between two values uniformly
    z_low, z_high = 0.2, 1.5
    z = np.random.uniform(z_low, z_high)

    sn_model = sncosmo.Model('hsiao')
    sn_model.set(t0=t0, z=z)
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
            time = times[k] + t0

            # get the apparent magnitude of the supernova at a given time; first get the
            # appropriate filter for the observation
            bandpass = sncosmo.get_bandpass(filters[j])
            # time should be in days
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='divide by zero encountered in log10')
                warnings.filterwarnings('ignore', message='invalid value encountered in log10')
                m_ia = sn_model.bandmag(bandpass, magsys='ab', time=time)
                if np.isnan(m_ia):
                    m_ia = -2.5 * np.log10(0.01) + filt_zp[j]

            t_f = 10**(-1/2.5 * (m_ia - filt_zp[j]))

            # noise floor of, say, 0.5% photometry in quadrature with shot noise and background
            # counts in e/s/pixel, assuming a WFIRST aperture size of psf_r pix, or ~pi r^2 pixels,
            # remembering to correct for the fact that uncertainties in fluxes are really done in
            # photon counts, so multiply then divide by exptime
            npix = np.pi * psf_r**2
            _f = t_f * exptime
            _d = dark * npix * exptime
            _b = bkg * npix * exptime
            _r = npix * readnoise**2
            flux_err = np.sqrt(_f + (0.005 * _f)**2 + _b + _d + _r) / exptime
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

    param_names = ['z', 't0', 'amplitude']
    sn_params = np.array([sn_model[q] for q in param_names])

    return lc_data, sn_params, true_flux


filters_master = np.array(['z087', 'y106', 'w149', 'j129', 'h158', 'f184'])
filt_zp_master = np.array([26.39, 26.41, 27.50, 26.35, 26.41, 25.96])

sse.register_filters(filters_master)

# for source in ['hsiao', 'nugent-sn91t', 'nugent-sn91bg', 'snana-2007y', 'snana-2004fe',
#                'snana-2007kw', 'nugent-sn2l', 'nugent-sn2n']:
#     for filt in ['z087', 'y106', 'w149', 'j129', 'h158', 'f184']:
#         sn_model = sncosmo.Model(source)
#         sn_model.set(t0=0, z=1)
#         print(source, filt, sn_model._source.peakphase(filt))
# sys.exit()


filt_ids = [0, 1, 3, 5]
filters = filters_master[filt_ids]
filt_zp = filt_zp_master[filt_ids]
min_offset, max_offset = -100, -5
n_obs = 15
t_interval = 5
t0, exptime = 0, 400
dark, readnoise, psf_r = 0.015, 20, 3

tot = 50

runs = 5
rows = np.ceil(np.sqrt(runs)).astype(int)
cols = np.ceil(runs / rows).astype(int)
gs = gridcreate('asdasd', cols, rows, 0.8, 5)

for j in range(runs):
    print(j)
    ax = plt.subplot(gs[j])

    early = np.empty(tot, np.bool)
    relative_offsets = np.empty(tot, float)

    multi_z_fit = False

    z_min, z_max = 0.2, 1.5

    i = 0
    while i < tot:
        t_low = np.random.uniform(min_offset, max_offset)
        t_high = (n_obs - 1) * t_interval + t_low
        times = np.arange(t_low, t_high, t_interval)
        sn_model = sncosmo.Model('hsiao')

        lc_data, sn_params, true_flux = make_fluxes(filters, times, filt_zp, t0, exptime, psf_r,
                                                    dark, readnoise)
        lc_data = Table(data=lc_data, names=['time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'])
        if not np.amax(lc_data['flux'].data / lc_data['fluxerr'].data) >= 5:
            continue

        bounds = {}
        bounds.update({'z': (0.2, 1.5)})
        params = ['t0', 'amplitude']
        params += ['z']

        if multi_z_fit:
            result = None
            fitted_model = None
            for z_init in np.linspace(z_min, z_max, 10):
                sn_model.set(z=z_init)
                result_temp, fitted_model_temp = sncosmo.fit_lc(lc_data, sn_model, params,
                                                                bounds=bounds, minsnr=5,
                                                                guess_z=False)
                if result is None or result_temp.chisq < result.chisq:
                    result = result_temp
                    fitted_model = fitted_model_temp
        else:
            fitted_model = sn_model

        result, fitted_model = sncosmo.fit_lc(lc_data, fitted_model, params, bounds=bounds,
                                              minsnr=5, guess_z=False if multi_z_fit else True)
        if np.any([message in result.message for message in ['error invalid',
                   'positive definite', 'No covar', 'Failed']]) or not result.success:
            continue

        fit_params = fitted_model.parameters
        fit_errors = result.errors
        dz_sigz = (fit_params[0] - sn_params[0]) / (fit_errors.get('z') + 1e-30)
        early[i] = 1 if t_high < 0 else 0
        relative_offsets[i] = dz_sigz
        i += 1
        if i % 25 == 0:
            print(i)

    for slicing, c in zip([np.ones(tot, np.bool), early, np.logical_not(early)], ['k', 'r', 'b']):
        cut = np.copy(relative_offsets[slicing & (np.abs(relative_offsets) <= 4)])
        hist, bins = np.histogram(cut, bins='auto')
        if bins[1] - bins[0] > 1:
            hist, bins = np.histogram(cut, bins=np.arange(bins[0], bins[-1]+1e-10, 1))
        sig_cut = np.abs(bins[:-1] + np.diff(bins)) <= 4
        _pdf = np.append(hist / np.diff(bins), 0) / np.sum(hist[sig_cut])
        _pdf_uncert = np.append(np.sqrt(hist) / np.diff(bins), 0) / np.sum(hist[sig_cut])
        ax.plot(bins, _pdf, ls='-', c=c, drawstyle='steps-post')
        # output1 = minimize(sumgauss, x0=[0, 1],
        #                    args=(bins[:-1], _pdf[:-1], _pdf_uncert[:-1]+1e-10, np.diff(bins)),
        #                    jac=gradgauss, method='L-BFGS-B', options = {'maxiter': 10000})
        # N = len(hist)
        output1 = minimize(fun_mle_gauss, x0=[0, 1], args=(cut), jac=True, method='newton-cg',
                           hess=hess_mle_gauss, options = {'maxiter': 10000})
        N = len(cut)

        mu, std = output1.x
        std = np.abs(std)
        label = r'$\mu$={:.2f}, $\sigma$={:.2f}, N={}, $\chi^2_\nu$={:.2f}, $\chi^2$={:.2f}'

        x = np.linspace(bins[0], bins[-1], 10000)
        ax.plot(x, norm.pdf(x, mu, std), ls='--', c=c,
                label=label.format(*output1.x, len(cut), output1.fun / (N - 2), output1.fun))

    x = np.linspace(*ax.get_xlim(), 1000)
    ax.plot(x, norm.pdf(x, 0, 1), ls='-', c='orange')
    ax.legend(fontsize=5)
    ax.set_xlabel(r'$\Delta$z/$\sigma_\mathrm{{z}}$')
    ax.set_ylabel('PDF')

plt.tight_layout()
plt.savefig('test_pdf_run_fitting.pdf')
