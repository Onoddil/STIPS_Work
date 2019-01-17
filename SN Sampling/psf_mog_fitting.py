from scipy.optimize import basinhopping, minimize
import multiprocessing
import itertools
import timeit
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from astropy.visualization import simple_norm
from webbpsf import wfirst
import copy


def gridcreate(name, y, x, ratio, z, **kwargs):
    # Function that creates a blank axis canvas; each figure gets a name (or alternatively a number
    # if none is given), and gridspec creates an N*M grid onto which you can create axes for plots.
    # This returns a gridspec "instance" so you can specific which figure to put the axis on if you
    # have several on the go.
    plt.figure(name, figsize=(z*x, z*ratio*y))
    gs = gridspec.GridSpec(y, x, **kwargs)
    return gs


def create_psf_image(filter_, directory, oversamp):
    # see https://webbpsf.readthedocs.io/en/stable/wfirst.html for details of detector things
    wfi = wfirst.WFI()
    wfi.filter = filter_
    wfi.detector = 'SCA09'
    # position can vary 4 - 4092, allowing for a 4 pixel gap
    wfi.detector_position = (2048, 2048)
    wfi.options['parity'] = 'odd'
    wfi.options['output_mode'] = 'both'

    psf = wfi.calc_psf(oversample=oversamp)

    return psf


def create_effective_psf(psf_, oversamp):
    # you lose oversamp/2 pixels at each edge, so overall lose oversamp pixels
    N = int(oversamp/2)
    psf = copy.deepcopy(psf_)
    # only have to remove the first (oversampled) pixel and the final oversampled (minus end of
    # slie) pixels, as the "first" effective pixel -- the sum over the NxN finer resolution
    # pixels -- will use the information from the sides, but we can remove the information
    # subsequently
    reduced_psf = np.empty((psf[0].data.shape[0] - 2*oversamp+1,
                            psf[0].data.shape[1] - 2*oversamp+1), float)

    for i in range(oversamp, psf[0].data.shape[0]-oversamp+1):
        for j in range(oversamp, psf[0].data.shape[1]-oversamp+1):
            # because the "middle" of an NxN pixel grid is two lower but only one higher than the
            # specific pixel (i.e., p0 p1 [p2 is this pixel] p3), we only go +-N with python's
            # 'drop the last value' slicing; otherwise we'd sum 2N+1 data points for each
            # oversample, creating additional flux
            reduced_psf[i-oversamp, j-oversamp] = np.sum(psf[0].data[i-N:i+N, j-N:j+N])
    psf[0].data = reduced_psf
    psf[0].header['NAXIS1'] = reduced_psf.shape[0]
    psf[0].header['NAXIS2'] = reduced_psf.shape[1]
    psf[0].header['HISTORY'] = "Created oversampled ePSF response at original pixel resolution"
    return psf


# f = c/(2pi sx sy sqrt(1 - p**2)) *
# exp(-0.5/(1 - p**2)*((x - mux)**2/sx**2 + (y - muy)**2/sy**2 - 2 p (x - mux)*(y - muy)/(sx*sy)))
# = c/(2 pi sx sy sqrt(1- p**2)) exp(-0.5/(1 - p**2) A)
# A = (x - mux)**2/sx**2 + (y - muy)**2/sy**2 - B
# B = 2 p (x - mux) (y - muy) / (sx sy)
# also, C = (x - mux)/sx - p (y - muy) / sy; D = (y - muy) / sy - p (x - mux) / sx
# dfdmux = f/(sx(1 - p**2)) * C
# dfdmuy = f/(sy(1 - p**2)) * D
# dfdsx = f (x - mux)/(sx**2 * (1 - p**2)) * C - f / sx
# dfdsy = f (y - muy)/(sy**2 * (1 - p**2)) * D - f / sy
# dfdp = f p / (1 - p**2) * (1 - B/(1 - p**2) + A/(2*p))
# dfdc = f / c


# F = sum_i (sum_j f_ij(c_j, s_j, mux_j, muy_j, x_i, y_i) - z_i)**2
# dFda = sum_i 2 (sum_j f_ij - z_i) dfikda (assuming no parameters are shared between individual
# gaussians in a MoG -- notice the dropped j subscribe in dfikda; this is a specific gaussian only)
# d2Fdadb = sum_i 2 (sum_j f_ij - z_i) d2fikdadb + 2 dfikda dfildb (d2fikdadb is zero unless
# a and b are parameters of a single gaussian, given the non-sharing of parameters; however, the
# second term is always present across off-axis terms)


def psf_fit_min(p, x, y, z):
    mu_xs, mu_ys, s_xs, s_ys, rhos, cks = \
        np.array([p[0+i*6] for i in range(0, int(len(p)/6))]), \
        np.array([p[1+i*6] for i in range(0, int(len(p)/6))]), \
        np.array([p[2+i*6] for i in range(0, int(len(p)/6))]), \
        np.array([p[3+i*6] for i in range(0, int(len(p)/6))]), \
        np.array([p[4+i*6] for i in range(0, int(len(p)/6))]), \
        np.array([p[5+i*6] for i in range(0, int(len(p)/6))])

    model_zs = np.empty((len(p) // 6, len(y), len(x)), float)
    model_z = np.zeros((len(y), len(x)), float)
    dfdcs = np.empty_like(model_zs)
    As, Bs, Cs, Ds, x_s, y_s = [], [], [], [], [], []
    for i, (mu_x, mu_y, sx, sy, rho, ck) in enumerate(zip(mu_xs, mu_ys, s_xs, s_ys, rhos, cks)):
        omp2 = 1 - rho**2
        x_ = (x - mu_x).reshape(1, -1)
        y_ = (y - mu_y).reshape(-1, 1)
        A = x_ * y_ / (sx * sy)
        B = (x_/sx)**2 + (y_/sy)**2 - A * 2 * rho
        C = x_ / sx - rho * y_ / sy
        D = y_ / sy - rho * x_ / sx
        As.append(A)
        Bs.append(B)
        Cs.append(C)
        Ds.append(D)
        x_s.append(x_)
        y_s.append(y_)
        # as dfdc = f / c this definition allows for the avoidance of divide-by-zero errors
        dfdcs[i] = 1/(2 * np.pi * sx * sy * np.sqrt(omp2)) * np.exp(-0.5/omp2 * B)
        model_zs[i] = ck * dfdcs[i]
        model_z += model_zs[i]
    dz = model_z - z

    jac = np.empty(len(p), float)
    # model_z is sum_j f_ij above
    for i in range(0, len(p)):
        i_set = i // 6
        i_in = i % 6
        mu_x, mu_y, sx, sy, rho, ck = mu_xs[i_set], mu_ys[i_set], s_xs[i_set], s_ys[i_set], \
            rhos[i_set], cks[i_set]
        x_, y_, A, B, C, D = x_s[i_set], y_s[i_set], As[i_set], Bs[i_set], Cs[i_set], Ds[i_set]
        f = model_zs[i_set]
        omp2 = 1 - rho**2
        # each of our six parameters in turn are mux, muy, sx, sy, rho and c.
        if i_in == 0:
            dfda = f / (sx * omp2) * C
        elif i_in == 1:
            dfda = f / (sy * omp2) * D
        elif i_in == 2:
            dfda = f * x_ / (sx**2 * omp2) * C - f / sx
        elif i_in == 3:
            dfda = f * y_ / (sy**2 * omp2) * D - f / sy
        elif i_in == 4:
            dfda = f * rho / omp2 * (1 - B/omp2 + A)
        elif i_in == 5:
            dfda = dfdcs[i_set]
        # differential of sum_i (sum_j f_ij - z_i)**2 / o**2 is
        # sum_i (2 * (sum_j f_ij - z_i) * dfda / o**2)
        dFda = 2 * np.sum(dz * dfda)
        jac[i] = dFda
    return np.sum(dz * dz), jac


class MyTakeStep(object):
    def __init__(self, stepsize=0.5):
        self.stepsize = stepsize

    def __call__(self, x):
        s = self.stepsize
        # each six parameter set is mux, muy, sx, sy, p, c. Whatever the stepsize is accept for
        # mux and muy, but reduce by a factor for sigma, p and c
        stepper = np.arange(0, len(x)//6).astype(int)
        x[0 + stepper] += np.random.uniform(-s, s, len(stepper))
        x[1 + stepper] += np.random.uniform(-s, s, len(stepper))
        x[2 + stepper] += np.random.uniform(-min(s/20, 0.05), min(s/20, 0.05), len(stepper))
        x[3 + stepper] += np.random.uniform(-min(s/20, 0.05), min(s/20, 0.05), len(stepper))
        x[4 + stepper] += np.random.uniform(-min(s/5, 0.1), min(s/5, 0.1), len(stepper))
        x[5 + stepper] += np.random.uniform(-min(s/3, 0.5), min(s/3, 0.5), len(stepper))
        return x


def psf_fitting_wrapper(iterable):
    np.random.seed(seed=None)
    i, (x, y, psf_image, x_cent, y_cent, N, min_kwarg, niters, x0, s, t) = iterable
    if x0 is None:
        x0 = []
        for _ in range(0, N):
            x0 = x0 + [np.random.normal(x_cent, 3*s), np.random.normal(y_cent, 3*s),
                       np.random.uniform(0.05, 0.3), np.random.uniform(0.05, 0.3),
                       np.random.uniform(0, 0.3), np.random.random()]
    take_step = MyTakeStep(stepsize=s)
    res = basinhopping(psf_fit_min, x0, minimizer_kwargs=min_kwarg, niter=niters, T=t,
                       stepsize=s, take_step=take_step)

    return res


def psf_fit_fun(p, x, y):
    mu_xs, mu_ys, s_xs, s_ys, rhos, cks = \
        np.array([p[0+i*6] for i in range(0, int(len(p)/6))]), \
        np.array([p[1+i*6] for i in range(0, int(len(p)/6))]), \
        np.array([p[2+i*6] for i in range(0, int(len(p)/6))]), \
        np.array([p[3+i*6] for i in range(0, int(len(p)/6))]), \
        np.array([p[4+i*6] for i in range(0, int(len(p)/6))]), \
        np.array([p[5+i*6] for i in range(0, int(len(p)/6))])
    psf_fit = np.zeros((len(y), len(x)), float)
    for i, (mu_x, mu_y, sx, sy, rho, ck) in enumerate(zip(mu_xs, mu_ys, s_xs, s_ys, rhos, cks)):
        omp2 = 1 - rho**2
        x_ = (x - mu_x).reshape(1, -1)
        y_ = (y - mu_y).reshape(-1, 1)
        B = 2 * rho * x_ * y_ / (sx * sy)
        A = (x_/sx)**2 + (y_/sy)**2 - B
        psf_fit = psf_fit + ck/(2 * np.pi * sx * sy * np.sqrt(omp2)) * np.exp(-0.5/omp2 * A)
    return psf_fit


def eq_con(x, g):
    # since sum_k c_k = g, the equality constrain is sum_k c_k - g
    return np.sum([x[5+i*6] for i in range(0, int(len(x)/6))]) - g


def eq_con_jac(x, g):
    # if f = sum_k c_k - g, then dfdc_1 = 1 for all c_k; zero otherwise
    return np.array([0, 0, 0, 0, 0, 1]*int(len(x)/6))


def background_mog_fit(p, x, y):
    c = p[0]
    x = x.reshape(1, -1)
    y = y.reshape(-1, 1)
    x_ = np.exp(-0.5 * (x / c)**2)
    y_ = np.exp(-0.5 * (y / c)**2)
    exp_xy = x_ * y_ / (2 * np.pi * c**2)
    f = np.sum(exp_xy)
    dfdc = np.sum(exp_xy * ((x**2 + y**2) / c**3 - 2/c))
    return (f - 1)**2, np.array([2 * (f - 1) * dfdc])


def psf_mog_fitting(psf_names, oversamp, psf_comp_filename, N_comp, type_, max_pix_offset,
                    center, diff_edges):
    gs = gridcreate('adsq', 4, len(psf_names), 0.8, 5)
    # assuming each gaussian component has mux, muy, sigx, sigy, rho, c, and that we fit for
    # N_comp Gaussians in the central region, and fit each diffration spike separately
    psf_comp = np.empty((len(psf_names), N_comp + 12, 6), float)
    diff_angles = np.empty(12, float)
    x0, y0 = center
    for i, edge in enumerate(diff_edges):
        x1, y1 = edge
        grad = (y1 - y0) / (x1 - x0)
        angle = 180 - np.degrees(np.arctan(1/grad))
        diff_angles[[i, i+len(diff_edges)]] = angle

    for j in range(0, len(psf_names)):
        print(j)
        f = pyfits.open(psf_names[j])
        # #### WFC3 ####
        # as WFC3-2016-12 suggests that fortran reads these files (x, y, N) we most likely read
        # them as (N, y, x) with the transposition from f- to c-order, thus the psf is (y, x) shape
        # psf_image = f[0].data[4, :, :]
        # #### WFIRST ####
        psf_image = f[0].data
        x, y = np.arange(0, psf_image.shape[1])/oversamp, np.arange(0, psf_image.shape[0])/oversamp
        x_cent, y_cent = (x[-1]+x[0])/2, (y[-1]+y[0])/2
        over_index_middle = 1 / 2
        cut_int = ((x.reshape(1, -1) % 1.0 == over_index_middle) &
                   (y.reshape(-1, 1) % 1.0 == over_index_middle))
        total_flux = np.sum(psf_image[cut_int])
        x -= x_cent
        y -= y_cent
        x_cent, y_cent = 0, 0

        ax = plt.subplot(gs[0, j])
        norm = simple_norm(psf_image, 'log', percent=100)
        # with the psf being (y, x) we do not need to transpose it to correct for pcolormesh being
        # flipped, but our x and y need additional tweaking, as these are pixel centers, but
        # pcolormesh wants pixel edges. we thus subtract half a pixel off each value and add a
        # final value to the end
        dx, dy = np.mean(np.diff(x)), np.mean(np.diff(y))
        x_pc, y_pc = np.append(x - dx/2, x[-1] + dx/2), np.append(y - dy/2, y[-1] + dy/2)
        img = ax.pcolormesh(x_pc, y_pc, psf_image, cmap='viridis', norm=norm, edgecolors='face',
                            shading='flat')
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('PSF Response')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')

        x_w = np.where((x >= -1 * max_pix_offset) & (x <= max_pix_offset))[0]
        y_w = np.where((y >= -1 * max_pix_offset) & (y <= max_pix_offset))[0]
        y_w0, y_w1, x_w0, x_w1 = np.amin(y_w), np.amax(y_w), np.amin(x_w), np.amax(x_w)
        psf_image_c = np.copy(psf_image[y_w0:y_w1+1, x_w0:x_w1+1])
        x_c, y_c = x[x_w0:x_w1+1], y[y_w0:y_w1+1]
        cut_int = ((x_c.reshape(1, -1) % 1.0 == over_index_middle) &
                   (y_c.reshape(-1, 1) % 1.0 == over_index_middle))
        cut_flux = np.sum(psf_image_c[cut_int])
        ax.set_title(r'Cut flux is {:.3f}\% of total flux'.format(cut_flux/total_flux*100))
        ax.axvline(x_c[0], c='k', ls='-')
        ax.axvline(x_c[-1], c='k', ls='-')
        ax.axhline(y_c[0], c='k', ls='-')
        ax.axhline(y_c[-1], c='k', ls='-')

        temp = 0.01

        start = timeit.default_timer()
        N_pools = 12
        N_overloop = 4
        niters = 250
        pool = multiprocessing.Pool(N_pools)
        counter = np.arange(0, N_pools*N_overloop)
        xy_step = max_pix_offset/3
        x0 = None
        method = 'SLSQP'  # 'L-BFGS-B'
        # we must constrain sum_k c_k = cut_flux, to ensure flux preservation in convolution
        min_kwarg = {'method': method, 'args': (x_c, y_c, psf_image_c),
                     'jac': True, 'bounds': [(x_c[0], x_c[-1]), (y_c[0], y_c[-1]),
                                             (1e-1, 3), (1e-1, 3), (-0.9, 0.9),
                                             (None, None)]*N_comp,
                     'constraints': {'type': 'eq', 'fun': eq_con, 'jac': eq_con_jac,
                                     'args': [cut_flux]}}
        iter_rep = itertools.repeat([x_c, y_c, psf_image_c, x_cent, y_cent, N_comp, min_kwarg,
                                     niters, x0, xy_step, temp])
        iter_group = zip(counter, iter_rep)
        res = None
        min_val = None
        for stuff in pool.imap_unordered(psf_fitting_wrapper, iter_group, chunksize=N_overloop):
            if min_val is None or stuff.fun < min_val:
                res = stuff
                min_val = stuff.fun
        p = res.x

        # if we want the integral -- or sum -- over pixels r < 20 to be 1 - cut_flux then we need
        # to figure out what the sigma for that must be. the easiest way to try this is to just
        # pick an uncertainty at which the integral out to the PSF edge is some large, but not
        # quite unity, value. \int_0^20 r/c^2 exp(-0.5 r^2/c^2) dr = d solves as
        # c = 10 sqrt(2) sqrt(-1 / ln(1 - d))
        # x_int, y_int = np.arange(-20, 20.1, 1), np.arange(-20, 20.1, 1)
        # int_lim = 0.99
        # c_ = 10 * np.sqrt(2) * np.sqrt(-1 / np.log(1 - int_lim))
        # new_g = [0, 0, c_, c_, 0, 1 - np.sum(psf_fit_fun(p, x_int, y_int))]
        # p = np.append(p, new_g)

        psf_comp[j, :, :] = p.reshape(N_comp+12, 6)

        print(timeit.default_timer()-start)
        print(psf_fit_min(p, x, y, psf_image)[0])
        psf_fit = psf_fit_fun(p, x, y)
        ax = plt.subplot(gs[1, j])
        norm = simple_norm(psf_fit, 'log', percent=100)
        img = ax.pcolormesh(x_pc, y_pc, psf_fit, cmap='viridis', norm=norm, edgecolors='face', shading='flat')
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('PSF Response')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')
        ax.set_title('Model PSF sum: {:.5f}'.format(np.sum(psf_fit_fun(p, x_int, y_int))))
        ax.axvline(x_c[0], c='k', ls='-')
        ax.axvline(x_c[-1], c='k', ls='-')
        ax.axhline(y_c[0], c='k', ls='-')
        ax.axhline(y_c[-1], c='k', ls='-')

        ax = plt.subplot(gs[2, j])
        ratio = np.zeros_like(psf_fit)
        ratio[psf_image != 0] = (psf_fit[psf_image != 0] - psf_image[psf_image != 0]) / \
            psf_image[psf_image != 0]
        ratio_ma = np.ma.array(ratio, mask=(psf_image == 0) & (psf_image > 1e-3))
        norm = simple_norm(ratio[(ratio != 0) & (psf_image > 1e-3)], 'linear', percent=100)
        cmap = plt.get_cmap('viridis')
        cmap.set_bad('w', 0)
        img = ax.pcolormesh(x_pc, y_pc, ratio_ma, cmap=cmap, norm=norm, edgecolors='face', shading='flat')
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('Relative Difference')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')
        ax.axvline(x_c[0], c='k', ls='-')
        ax.axvline(x_c[-1], c='k', ls='-')
        ax.axhline(y_c[0], c='k', ls='-')
        ax.axhline(y_c[-1], c='k', ls='-')

        ax = plt.subplot(gs[3, j])
        ratio = (psf_fit - psf_image)
        ratio_ma = np.ma.array(ratio, mask=(psf_image == 0) & (psf_image > 1e-3))
        norm = simple_norm(ratio[(ratio != 0) & (psf_image > 1e-3)], 'linear', percent=100)
        cmap = plt.get_cmap('viridis')
        cmap.set_bad('w', 0)
        img = ax.pcolormesh(x_pc, y_pc, ratio_ma, cmap=cmap, norm=norm, edgecolors='face', shading='flat')
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('Absolute Difference')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')
        ax.axvline(x_c[0], c='k', ls='-')
        ax.axvline(x_c[-1], c='k', ls='-')
        ax.axhline(y_c[0], c='k', ls='-')
        ax.axhline(y_c[-1], c='k', ls='-')

    plt.tight_layout()
    plt.savefig('psf_fit/test_psf_mog_{}.pdf'.format(type_))

    np.save(psf_comp_filename, psf_comp)


if __name__ == '__main__':
    filters = ['z087', 'y106', 'w149', 'j129', 'h158', 'f184']
    # psfs is a list of HDULists
    psfs = []
    reduced_psfs = []
    oversamp = 4

    # with output_model set to 'both' HDUList [0] is the oversampled data and [1] is the
    # detector-binned data -- i.e., the created ePSF but sampled at pixel centers, which is thus
    # propagated into reduced_psf with [1] unchaged but [0] now the ePSF (the detector-pixel
    # sampled fraction at oversampling levels of pixel positions).
    for filter_ in filters:
        psf = create_psf_image(filter_, 'psf_fit', oversamp)
        psfs.append(psf)
        reduced_psf = create_effective_psf(psf, oversamp)
        rp_hdulist = pyfits.HDUList([a for a in reduced_psf])
        rp_hdulist.writeto('../PSFs/{}.fits'.format(filter_), overwrite=True)
        reduced_psfs.append(reduced_psf)

    gs = gridcreate('a', 3, len(filters), 0.8, 5)
    for i in range(0, len(filters)):
        ax = plt.subplot(gs[0, i])
        norm = simple_norm(psfs[i][1].data, 'log', percent=100)
        img = ax.imshow(psfs[i][1].data, origin='lower', cmap='viridis', norm=norm)
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('{} PSF Detector Response'.format(filters[i]))
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')

        ax = plt.subplot(gs[1, i])
        norm = simple_norm(psfs[i][0].data, 'log', percent=100)
        img = ax.imshow(psfs[i][0].data, origin='lower', cmap='viridis', norm=norm)
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('{} PSF Supersampled Response'.format(filters[i]))
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')

        x, y = np.arange(0, reduced_psfs[i][0].data.shape[1])/oversamp, \
            np.arange(0, reduced_psfs[i][0].data.shape[0])/oversamp
        over_index_middle = 1 / 2
        cut_int = ((x.reshape(1, -1) % 1.0 == over_index_middle) &
                   (y.reshape(-1, 1) % 1.0 == over_index_middle))
        print(filters[i], np.sum(psfs[i][1].data), np.amax(psfs[i][1].data),
              np.sum(psfs[i][0].data), np.amax(psfs[i][0].data),
              np.sum(reduced_psfs[i][0].data[cut_int]), np.amax(reduced_psfs[i][0].data))
        ax = plt.subplot(gs[2, i])
        norm = simple_norm(reduced_psfs[i][0].data, 'log', percent=100)
        img = ax.imshow(reduced_psfs[i][0].data, origin='lower', cmap='viridis', norm=norm)
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('{} PSF Response'.format(filters[i]))
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')
    plt.tight_layout()
    plt.savefig('{}/wfirst_psfs.pdf'.format('psf_fit'))
