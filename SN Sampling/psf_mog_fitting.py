from scipy.optimize import basinhopping
import multiprocessing
import itertools
import timeit
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from astropy.visualization import simple_norm
from webbpsf import wfirst


def gridcreate(name, y, x, ratio, z, **kwargs):
    # Function that creates a blank axis canvas; each figure gets a name (or alternatively a number
    # if none is given), and gridspec creates an N*M grid onto which you can create axes for plots.
    # This returns a gridspec "instance" so you can specific which figure to put the axis on if you
    # have several on the go.
    plt.figure(name, figsize=(z*x, z*ratio*y))
    gs = gridspec.GridSpec(y, x, **kwargs)
    return gs


def create_psf_image(filter, directory, oversamp):
    # see https://webbpsf.readthedocs.io/en/stable/wfirst.html for details of detector things
    pixelscale = 110e-3
    wfi = wfirst.WFI(pixelscale=pixelscale/oversamp)
    wfi.filter = filter
    wfi.detector = 'SCA09'
    # position can vary 4 - 4092, allowing for a 4 pixel gap
    wfi.detector_position = (2048, 2048)
    wfi.options['parity'] = 'odd'
    psf = wfi.calc_psf()
    print(psf)
    print(psf[1].data.shape, np.sum(psf[1].data))
    # webbpsf creates a FITS HDUList, and for now we just want the data from the image...
    return psf[1].data


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


def psf_fit_min(p, x, y, z, o_inv_sq):
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
    group_terms = 2 * dz * o_inv_sq

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
        dFda = np.sum(group_terms * dfda)
        jac[i] = dFda
    # group_terms includes a two for the jacobian; remove subsequently, but computationally cheaper
    return 0.5 * np.sum(group_terms * dz), jac


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
        x[2 + stepper] += np.random.uniform(-min(s/100, 0.05), min(s/100, 0.05), len(stepper))
        x[3 + stepper] += np.random.uniform(-min(s/100, 0.05), min(s/100, 0.05), len(stepper))
        x[4 + stepper] += np.random.uniform(-min(s/50, 0.1), min(s/50, 0.1), len(stepper))
        x[5 + stepper] += np.random.uniform(-min(s/10, 0.5), min(s/10, 0.5), len(stepper))
        return x


def psf_fitting_wrapper(iterable):
    np.random.seed(seed=None)
    i, (x, y, psf_image, psf_inv_var, x_cent, y_cent, N, min_kwarg, niters, x0, s, t) = iterable
    if x0 is None:
        x0 = []
        for _ in range(0, N):
            x0 = x0 + [x_cent - s + np.random.random()*2*s, y_cent - s + np.random.random()*2*s,
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


def psf_mog_fitting(psf_names, os, noise_removal, psf_comp_filename, cut, N_comp):
    gs = gridcreate('adsq', 4, len(psf_names), 0.8, 15)
    # assuming each gaussian component has mux, muy, sigx, sigy, rho, c
    psf_comp = np.empty((len(psf_names), N_comp, 6), float)
    for j in range(0, len(psf_names)):
        print(j)
        f = pyfits.open(psf_names[j])
        # as WFC3-2016-12 suggests that fortran reads these files (x, y, N) we most likely read
        # them as (N, y, x) with the transposition from f- to c-order, thus the psf is (y, x) shape
        psf_image = f[0].data[4, :, :]

        x, y = np.arange(0, psf_image.shape[0])/os, np.arange(0, psf_image.shape[1])/os
        x_cent, y_cent = (x[-1]+x[0])/2, (y[-1]+y[0])/2
        over_index_middle = 1 / 2
        cut_int = ((x.reshape(1, -1) % 1.0 == over_index_middle) &
                   (y.reshape(-1, 1) % 1.0 == over_index_middle))
        # just ignore anything below cut*np.amax(image) to only fit central psf
        total_flux, cut_flux = np.sum(psf_image[cut_int]), \
            np.sum(psf_image[cut_int & (psf_image >= cut * np.amax(psf_image))])
        x -= x_cent
        y -= y_cent
        x_cent, y_cent = 0, 0

        y_w, x_w = np.where(psf_image >= cut * np.amax(psf_image))
        y_w0, y_w1, x_w0, x_w1 = np.amin(y_w), np.amax(y_w), np.amin(x_w), np.amax(x_w)

        x_, y_ = x[x_w0:x_w1+1], y[y_w0:y_w1+1]
        ax = plt.subplot(gs[3, j])
        psf_ratio = np.log10((np.abs(psf_image) / np.amax(psf_image)) + 1e-8)
        norm = simple_norm(psf_ratio, 'linear', percent=100)
        # with the psf being (y, x) we do not need to transpose it to correct for pcolormesh being
        # flipped, but our x and y need additional tweaking, as these are pixel centers, but
        # pcolormesh wants pixel edges. we thus subtract half a pixel off each value and add a
        # final value to the end
        dx, dy = np.mean(np.diff(x)), np.mean(np.diff(y))
        x_pc, y_pc = np.append(x - dx/2, x[-1] + dx/2), np.append(y - dy/2, y[-1] + dy/2)
        img = ax.pcolormesh(x_pc, y_pc, psf_ratio, cmap='viridis', norm=norm, edgecolors='face', shading='flat')
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('Log Absolute Relative PSF Response')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')
        ax.axvline(x_[0], c='k', ls='-')
        ax.axvline(x_[-1], c='k', ls='-')
        ax.axhline(y_[0], c='k', ls='-')
        ax.axhline(y_[-1], c='k', ls='-')

        psf_image = psf_image[y_w0:y_w1+1, x_w0:x_w1+1]
        # remove any edge features with a blanket zeroing of 'noise'
        if noise_removal:
            psf_image[psf_image < cut * np.amax(psf_image)] = 0
        x, y = x[x_w0:x_w1+1], y[y_w0:y_w1+1]

        ax = plt.subplot(gs[0, j])
        ax.set_title(r'Cut flux is {:.3f}\% of total flux'.format(cut_flux/total_flux*100))
        norm = simple_norm(psf_image, 'log', percent=99.9)
        # with the psf being (y, x) we do not need to transpose it to correct for pcolormesh being
        # flipped, but our x and y need additional tweaking, as these are pixel centers, but
        # pcolormesh wants pixel edges. we thus subtract half a pixel off each value and add a
        # final value to the end
        dx, dy = np.mean(np.diff(x)), np.mean(np.diff(y))
        x_pc, y_pc = np.append(x - dx/2, x[-1] + dx/2), np.append(y - dy/2, y[-1] + dy/2)
        img = ax.pcolormesh(x_pc, y_pc, psf_image, cmap='viridis', norm=norm, edgecolors='face', shading='flat')
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('PSF Response')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')

        temp = 0.01

        start = timeit.default_timer()
        N_pools = 12
        N_overloop = 4
        niters = 200
        pool = multiprocessing.Pool(N_pools)
        counter = np.arange(0, N_pools*N_overloop)
        xy_step = 2
        x0 = None
        min_kwarg = {'method': 'L-BFGS-B', 'args': (x, y, psf_image, 1),
                     'jac': True, 'bounds': [(x[0], x[-1]), (y[0], y[-1]),
                                             (1e-5, 5), (1e-5, 5), (-0.999, 0.999),
                                             (None, None)]*N_comp}
        iter_rep = itertools.repeat([x, y, psf_image, 1, x_cent, y_cent, N_comp, min_kwarg, niters,
                                     x0, xy_step, temp])
        iter_group = zip(counter, iter_rep)
        res = None
        min_val = None
        for stuff in pool.imap_unordered(psf_fitting_wrapper, iter_group, chunksize=N_overloop):
            if min_val is None or stuff.fun < min_val:
                res = stuff
                min_val = stuff.fun

        print(timeit.default_timer()-start)
        psf_comp[j, :, :] = res.x.reshape(N_comp, 6)
        p = res.x
        print(psf_fit_min(p, x, y, psf_image, 1)[0])
        psf_fit = psf_fit_fun(p, x, y)
        ax = plt.subplot(gs[1, j])
        norm = simple_norm(psf_fit, 'log', percent=99.9)
        img = ax.pcolormesh(x_pc, y_pc, psf_fit, cmap='viridis', norm=norm, edgecolors='face', shading='flat')
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('PSF Response')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')

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

    plt.tight_layout()
    plt.savefig('psf_fit/test_psf_mog.pdf')

    np.save(psf_comp_filename, psf_comp)


if __name__ == '__main__':
    filters = ['z087']  # , 'y106', 'w149', 'j129', 'h158', 'f184']
    psfs = []
    oversamp = 4
    for filter_ in filters:
        psfs.append(create_psf_image(filter_, 'psf_fit', oversamp))

    # TODO: now that we have hacked webbpsf to allow the creation of supersampled images, we need
    # to fix the normalisation. to do this, we need to produce a 'running sum' over the
    # oversamp**2 pixels

    gs = gridcreate('a', 1, 1, 0.8, 15)
    for i in range(0, len(filters)):
        print(filters[i], np.sum(psfs[i]), np.amax(psfs[i]))
        ax = plt.subplot(gs[i])
        norm = simple_norm(psfs[i], 'log', percent=99.9)
        img = ax.imshow(psfs[i], origin='lower', cmap='viridis', norm=norm)
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('{} PSF Response'.format(filters[i]))
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')
    plt.tight_layout()
    plt.savefig('{}/wfirst_psfs.pdf'.format('psf_fit'))
