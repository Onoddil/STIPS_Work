import numpy as np
from astropy.visualization import simple_norm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sn_sampling as sn
import astropy.io.fits as pyfits
import sncosmo
import astropy.units as u
import glob
from scipy.stats import binned_statistic, binned_statistic_2d
from matplotlib.cm import ScalarMappable
from scipy.special import erfinv


def gridcreate(name, y, x, ratio, z, **kwargs):
    # Function that creates a blank axis canvas; each figure gets a name (or alternatively a number
    # if none is given), and gridspec creates an N*M grid onto which you can create axes for plots.
    # This returns a gridspec "instance" so you can specific which figure to put the axis on if you
    # have several on the go.
    plt.figure(name, figsize=(z*x, z*ratio*y))
    gs = gridspec.GridSpec(y, x, **kwargs)
    return gs


def mcmc_runtime(sec_per_filt, n_filts):
    t = 0
    combo_tot = 0
    for k in range(1, n_filts+1):
        n_combos = np.math.factorial(n_filts) / np.math.factorial(k) / np.math.factorial(n_filts -
                                                                                         k)
        t += n_combos * sec_per_filt*k
        combo_tot += n_combos
    print("{} choices, {:.0f}/{:.0f}/{:.0f} approximate minutes/hours/days".format(
        combo_tot, t/60, t/60/60, t/60/60/24))


def make_figures(images_with_sn, images_without_sn, diff_images, filters, times, exptime):
    n = np.random.choice(len(times))
    # TODO: fix these times to include the randomness of the times given in run_cadence above
    t = times[n]
    iws = images_with_sn[n]
    ds = diff_images[n]
    gs = gridcreate('1', 3, len(filters), 0.8, 5)
    for j, (iw, iwo, d, f) in enumerate(zip(iws, images_without_sn, ds, filters)):
        ax = plt.subplot(gs[0, j])
        norm = simple_norm(iw / exptime, 'linear', percent=100)
        img = ax.imshow(iw / exptime, cmap='viridis', norm=norm, origin='lower')
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label(r'Flux / e$^-\,\mathrm{s}^{-1}$')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel, {}, t = {:.0f}'.format(f, t))
        ax = plt.subplot(gs[1, j])
        norm = simple_norm(iwo / exptime, 'linear', percent=100)
        img = ax.imshow(iwo / exptime, cmap='viridis', norm=norm, origin='lower')
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label(r'Flux / e$^-\,\mathrm{s}^{-1}$')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')
        ax = plt.subplot(gs[2, j])
        norm = simple_norm(d / exptime, 'linear', percent=100)
        img = ax.imshow(d / exptime, cmap='viridis', norm=norm, origin='lower')
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label(r'Flux / e$^-\,\mathrm{s}^{-1}$')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')
    plt.tight_layout()
    plt.savefig('out_gals/image.pdf')
    plt.close()


def model_number(run_minutes, n_runs):
    # assuming a static time for each run; dominated by fit, not creation currently
    n_filt_choice = 0
    n = 7  # including R, eventually
    for k in np.arange(2, n+1):
        n_filt_choice += np.math.factorial(n) / np.math.factorial(k) / np.math.factorial(n - k)
    # cadence can vary from, say, 5 days to 40 days (5 days being the minimum needed, and 25 days
    # giving 2 data points per lightcurve), so cadence could be varied in 5s initially, and thus
    cadence_interval = 5
    cadences = np.arange(5, 40+1e-10, cadence_interval)
    n_cadence = len(cadences)

    n_tot = n_filt_choice * n_cadence

    time = n_tot * run_minutes * n_runs

    print("{} choices, {} runs, {:.0f}/{:.0f}/{:.0f} approximate minutes/hours/days".format(n_tot, n_runs, time, time/60, time/60/24))


def faintest_sn(sn_types, filters, filt_minmags, exptime, filt_zp, snr_det, psf_r, rnoise, dark):
    filt_minmags = np.empty(len(filters), float)
    for i in range(0, len(filt_minmags)):
        # background counts for WFIRST is ~0.3-0.7 e/s/pix blue of F184, 1-3 e/s/pix for F184
        if filters[i] == 'F184':
            bkg = np.random.uniform(1, 3)
        else:
            bkg = np.random.uniform(0.3, 0.7)
        # get f from S = c / sqrt(sqrt(c)**2 + A) where c = f*t; A = D + B + R;
        # D = (sqrt(dpt))**2 = dpt, B = (sqrt(bpt))**2 = bpt; R = (p[ * ndit] * ron)**2
        R = np.pi * psf_r**2 * rnoise**2
        A = np.pi * psf_r**2 * (bkg + dark) * exptime + R
        f = (snr_det**2 + np.sqrt(4 * A * snr_det**2 + snr_det**4)) / (2 * exptime)
        filt_minmags[i] = -2.5 * np.log10(f) + filt_zp[i]
    for sn_type in sn_types:
        sn_model = sn.get_sn_model(sn_type, 1)
        for filt, filt_minmag in zip(filters, filt_minmags):
            z = 0
            sn_model.set(z=z)
            while (sn_model.bandoverlap(filt) and sn_model.bandmag(filt, 'ab',
                   sn_model.source.peakphase(filt)) < filt_minmag):
                sn_model.set(z=z)
                z += 0.01
            if sn_model.bandoverlap(filt):
                print(sn_type, filt, 'z={:.2f}'.format(z), 'mag loss')
            else:
                print(sn_type, filt, 'z={:.2f}'.format(z), 'wavelength loss')


def register_filters(filters):
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


def get_sn_priors(kind='reduced'):
    # li et al., 2011, mnras, 412, 1441 give:
    # Ia 0.24, Ibc 0.19, II 0.57; Ib 0.21 Ic 0.54
    if kind == 'reduced':
        # Ia 0.24, II 0.57 Ibc 0.19; Ib 0.21 Ic 0.54; Ib = 0.19*0.21/0.75 = 0.053,
        # Ic = 0.19*0.54/0.75 = 0.137
        return np.array([0.24, 0.053, 0.137, 0.57])
    else:
        # IIP 0.7, IIL 0.1, IIn 0.09; Ia 0.7 Ia91T 0.09 Ia91bg 0.15, and thus:
        # fia = 0.24*0.7/0.94 ~ 0.179, fiat = 0.24*0.09/0.94 ~ 0.022,
        # fiabg = 0.24*0.15/0.94 ~ 0.038,
        # fib = 0.19*0.21/(0.21+0.54) ~ 0.053, fic = 0.19*0.54/(0.21+0.54) ~ 0.137,
        # fiip = 0.57*0.69/0.88 ~ 0.447, fiil = 0.57*0.1/0.88 ~ 0.065,
        # fiin = 0.57*0.09/0.88 ~ 0.058, ignoring 'peculiar' Ibc and IIbs
        return np.array([0.179, 0.022, 0.038, 0.053, 0.137, 0.447, 0.065, 0.058])


def make_fit_fig(directory, sn_types, probs, x2s, lc_data, ncol, bestfit_results, bestfit_models,
                 figtext):
    best_ind = np.argmax(probs[:-1])
    best_r = bestfit_results[best_ind]
    best_m = bestfit_models[best_ind]
    best_x2 = x2s[best_ind]
    figtext = [figtext[0], figtext[1] + '\n' + r'$\chi^2_{{\nu={}}}$ = {:.3f}'.format(
               best_r.ndof, best_x2/best_r.ndof)]
    errors = best_r.errors
    model_params = best_m.parameters

    z_format = sncosmo.utils.format_value(model_params[0], errors.get('z'), latex=True)
    t0_format = sncosmo.utils.format_value(model_params[1], errors.get('t0'), latex=True)
    A_format = sncosmo.utils.format_value(model_params[2], errors.get('amplitude'),
                                          latex=True)
    figtext.append('Type {}: $z = {}$\n$t_0 = {}$'.format(sn_types[best_ind],
                   z_format, t0_format))
    if probs[0] > 0:
        p_sig = int(np.floor(np.log10(abs(probs[0]))))
    else:
        p_sig = 0
    if p_sig > 3:
        figtext.append('$A = {}$\n$P(Ia|D) = {:.3f} \\times 10^{{{}}}$'.format(
            A_format, probs[0]/10**p_sig, p_sig))
    else:
        figtext.append('$A = {}$\n$P(Ia|D) = {:.3f}$'.format(A_format, probs[0]))

    ypad = 2
    fig = sncosmo.plot_lc(lc_data, model=bestfit_models, xfigsize=5*ncol, tighten_ylim=False,
                          ncol=ncol, figtext=figtext, figtextsize=ypad, model_label=sn_types)
    fig.tight_layout(rect=[0, 0.03, 1, 0.935])

    files = glob.glob('{}/fit_*.pdf'.format(directory))
    if len(files) == 0:
        i = -1
    else:
        f = [int(f.split('_')[-1].split('.')[0]) for f in files]
        i = np.amax(f)
    fig.savefig('{}/fit_{}.pdf'.format(directory, i+1))


def make_goodness_corner_fig(percentiles, names, ndim, params, axis_names, fracinds, flat_samples,
                             flat_blobs, labels, directory, subname, min_1d_count, min_2d_count,
                             max_interval, min_offset, max_offset):
    gs_outer = gridcreate('0', len(percentiles), len(names), 1, 4*ndim)
    for jj, (param, name, axis_name, fracind) in enumerate(zip(params, names,
                                                               axis_names, fracinds)):
        if jj == len(names) - 1:
            func = 'count'
        else:
            func = lambda x: np.nanpercentile(x, percentile)
        for ii, percentile in enumerate(percentiles):
            gs = gridspec.GridSpecFromSubplotSpec(ndim, ndim*3, subplot_spec=gs_outer[ii, jj],
                                                  hspace=0, wspace=0)
            ndim_bins = []
            for i in range(0, ndim):
                hist, bins, _ = binned_statistic(flat_samples[:, i], param,
                                                 statistic=func, bins=20)
                ndim_bins.append(bins)
                if jj < len(names) - 1:
                    count, _, _ = binned_statistic(flat_samples[:, i], param,
                                                   statistic='count', bins=bins)
                    hist[count < min_1d_count] = np.nan
                else:
                    hist[hist <= 0] = np.nan

                ax = plt.subplot(gs[i, 3*i:3*i+3], label=jj*2+ii*3+i*5)
                ax.plot(bins, np.append(hist, hist[-1]), ls='-', c='k', drawstyle='steps-post')
                if i == ndim - 1:
                    ax.set_xlabel(labels[i])
                else:
                    ax.set_xticklabels([])
                ax.set_xlim(bins[0], bins[-1])
                if i == 0:
                    ax.set_ylabel(axis_name)
                    if jj < len(names) - 4:
                        posfrac = np.sum(flat_blobs[:, fracind] == 1) / len(flat_blobs) * 100
                        negfrac = np.sum(flat_blobs[:, fracind] == -1) / len(flat_blobs) * 100
                        if ndim > 1:
                            ax.set_title(r'Overall sign: +ve {:.0f}%, -ve: {:.0f}%'.format(
                                posfrac, negfrac))
                        else:
                            ax.set_title(name + '\n' +
                                         r'Overall sign: +ve {:.0f}%, -ve: {:.0f}%' + '\n' +
                                         r'log$_{{10}}$(1D gaussian CDF): {:.2f})'
                                         .format(posfrac, negfrac,
                                                 np.log10(np.sqrt(2) * erfinv(percentile/100))))
                    else:
                        if ndim == 1:
                            ax.set_title(name)
                if i == 1 and jj < 3:  # just the first 3 variables, normalised by uncertainty
                    ax.set_title(r'log$_{{10}}$(1D gaussian CDF): {:.2f}'.format(
                                 np.log10(np.sqrt(2) * erfinv(percentile/100))))

                ax.yaxis.tick_right()
                for j in range(0, ndim):
                    if j >= i:
                        if ii == 0 and i == 0 and j == np.floor((ndim - 1) / 2) and ndim > 1:
                            ax = plt.subplot(gs[i, 3*j:3*j+3], label=jj*3+ii*1+i*2+j*3)
                            ax.set_frame_on(False)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.set_title(name)
                        continue
                    ax = plt.subplot(gs[i, 3*j:3*j+3], label=jj*30+ii*38+i*4+j*7)
                    hist, ybins, xbins, _ = binned_statistic_2d(flat_samples[:, i],
                                                                flat_samples[:, j],
                                                                param, statistic=func,
                                                                bins=[ndim_bins[i], ndim_bins[j]])
                    if jj < len(names) - 1:
                        count, _, _, _ = binned_statistic_2d(flat_samples[:, i],
                                                             flat_samples[:, j], param,
                                                             statistic='count',
                                                             bins=[ybins, xbins])
                        hist[count < min_2d_count] = np.nan
                    else:
                        hist[hist <= 0] = np.nan
                    _hist = np.ma.array(hist, mask=np.isnan(hist))

                    cmap = plt.get_cmap('viridis')
                    cmap.set_bad(color='r', alpha=1)

                    norm = simple_norm(hist, 'linear', max_cut=np.nanpercentile(hist, 99),
                                       min_cut=max(np.nanpercentile(hist, 1), -10), clip=False)
                    ax.pcolormesh(xbins, ybins, _hist, norm=norm, cmap=cmap,
                                  edgecolors='face', shading='flat')
                    ax.set_xlim(ndim_bins[j][0], ndim_bins[j][-1])
                    ax.set_ylim(ndim_bins[i][0], ndim_bins[i][-1])
                    if i == ndim - 1:
                        ax.set_xlabel(labels[j])
                    else:
                        ax.set_xticklabels([])
                    if j == 0:
                        if jj == 0 and i == np.floor((ndim - 1) / 2):
                            ending = ('st' if percentile % 10 == 1 else 'nd' if percentile %
                                      10 == 2 else 'th')
                            ax.set_ylabel(r'{}$^\mathrm{{{}}}$ percentile'.format(percentile,
                                          ending) + '\n' + labels[i])
                        else:
                            ax.set_ylabel(labels[i])
                    else:
                        ax.set_yticklabels([])
            if ndim > 1:
                ax = plt.subplot(gs[0, -2], label=jj*3+ii*104+i*99)
                cb = plt.colorbar(ScalarMappable(norm=norm, cmap='viridis'),
                                  cax=ax, ax=ax, use_gridspec=True, orientation='vertical')
                cb.set_label(axis_name)
    plt.tight_layout()
    plt.savefig('{}/{}_corner_fit.pdf'.format(directory, subname))
