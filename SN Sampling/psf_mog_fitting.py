from scipy.optimize import basinhopping
import multiprocessing
import itertools
import timeit
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from astropy.visualization import simple_norm

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


@profile
def psf_fit_min(p, x, y, z, o_inv_sq):
    mu_xs, mu_ys, s_xs, s_ys, rhos, cks = \
        np.array([p[0+i*6] for i in range(0, int(len(p)/6))]), \
        np.array([p[1+i*6] for i in range(0, int(len(p)/6))]), \
        np.array([p[2+i*6] for i in range(0, int(len(p)/6))]), \
        np.array([p[3+i*6] for i in range(0, int(len(p)/6))]), \
        np.array([p[4+i*6] for i in range(0, int(len(p)/6))]), \
        np.array([p[5+i*6] for i in range(0, int(len(p)/6))])

    model_zs = np.empty((len(p) // 6, len(x), len(y)), float)
    dfdcs = np.empty_like(model_zs)
    As, Bs, Cs, Ds, x_s, y_s = [], [], [], [], [], []
    for i, (mu_x, mu_y, sx, sy, rho, ck) in enumerate(zip(mu_xs, mu_ys, s_xs, s_ys, rhos, cks)):
        omp2 = 1 - rho**2
        x_ = (x - mu_x).reshape(-1, 1)
        y_ = (y - mu_y).reshape(1, -1)
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
    model_z = np.sum(model_zs, axis=0)
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
        x[2 + stepper] += np.random.uniform(-min(5, s/5), min(5, s/5), len(stepper))
        x[3 + stepper] += np.random.uniform(-min(5, s/5), min(5, s/5), len(stepper))
        x[4 + stepper] += np.random.uniform(-0.5, 0.5, len(stepper))
        x[5 + stepper] += np.random.uniform(-0.5, 0.5, len(stepper))
        return x


def psf_fitting_wrapper(iterable):
    np.random.seed(seed=None)
    i, (x, y, psf_image, psf_inv_var, x_cent, y_cent, N, min_kwarg, niters) = iterable
    x0 = []
    for _ in range(0, N):
        x0 = x0 + [x_cent - 20 + np.random.random()*40, y_cent - 20 + np.random.random()*40,
                   np.random.random()*0.5, np.random.random()*0.5, np.random.random(),
                   np.random.random()]
    take_step = MyTakeStep()
    res = basinhopping(psf_fit_min, x0, minimizer_kwargs=min_kwarg, niter=niters, T=5,
                       stepsize=50, take_step=take_step)

    return res


@profile
def psf_mog_fitting(psf_names):
    psf_names = ['../../../Buffalo/PSFSTD_WFC3IR_F{}W.fits'.format(q) for q in [105, 125, 160]]
    gs = gridcreate('adsq', 4, len(psf_names), 0.8, 15)
    for j in range(0, len(psf_names)):
        print(j)
        f = pyfits.open(psf_names[j])
        psf_image = f[0].data[4, :, :]

        ax = plt.subplot(gs[0, j])
        norm = simple_norm(psf_image, 'log', percent=99.9)
        img = ax.imshow(psf_image, origin='lower', cmap='viridis', norm=norm)
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('PSF Response')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')

        x, y = np.arange(0, psf_image.shape[0]), np.arange(0, psf_image.shape[1])
        x_cent, y_cent = np.ceil((psf_image.shape[0]-1)/2), np.ceil((psf_image.shape[1]-1)/2)
        psf_uncert = np.zeros_like(psf_image)
        psf_uncert[psf_image > 0] = np.sqrt(psf_image[psf_image > 0]) + 0.001
        psf_uncert[psf_image <= 0] = 1
        psf_inv_var = 1 / psf_uncert**2
        N = 8
        min_kwarg = {'method': 'L-BFGS-B', 'args': (x, y, psf_image, psf_inv_var), 'jac': True,
                     'bounds': [(0, len(x)-1), (0, len(y)-1), (1e-3, 10), (1e-3, 10),
                                (1e-5, 0.999), (None, None)]*N}
        N_pools = 8
        niters = 150
        pool = multiprocessing.Pool(N_pools)
        counter = np.arange(0, N_pools)
        iter_rep = itertools.repeat([x, y, psf_image, psf_inv_var, x_cent, y_cent, N, min_kwarg,
                                     niters])
        iter_group = zip(counter, iter_rep)
        res = None
        min_val = None
        start = timeit.default_timer()
        for stuff in pool.imap_unordered(psf_fitting_wrapper, iter_group, chunksize=1):
            if min_val is None or stuff.fun < min_val:
                res = stuff
                min_val = stuff.fun
        print(res)
        print(timeit.default_timer()-start)

        p = res.x
        mu_xs, mu_ys, s_xs, s_ys, rhos, cks = \
            np.array([p[0+i*6] for i in range(0, int(len(p)/6))]), \
            np.array([p[1+i*6] for i in range(0, int(len(p)/6))]), \
            np.array([p[2+i*6] for i in range(0, int(len(p)/6))]), \
            np.array([p[3+i*6] for i in range(0, int(len(p)/6))]), \
            np.array([p[4+i*6] for i in range(0, int(len(p)/6))]), \
            np.array([p[5+i*6] for i in range(0, int(len(p)/6))])
        psf_fit = np.zeros((len(x), len(y)), float)
        for i, (mu_x, mu_y, sx, sy, rho, ck) in enumerate(zip(mu_xs, mu_ys, s_xs, s_ys, rhos, cks)):
            omp2 = 1 - rho**2
            x_ = (x - mu_x).reshape(-1, 1)
            y_ = (y - mu_y).reshape(1, -1)
            B = 2 * rho * x_ * y_ / (sx * sy)
            A = (x_/sx)**2 + (y_/sy)**2 - B
            psf_fit = psf_fit + ck/(2 * np.pi * sx * sy * np.sqrt(omp2)) * np.exp(-0.5/omp2 * A)
        ax = plt.subplot(gs[1, j])
        norm = simple_norm(psf_fit, 'log', percent=99.9)
        img = ax.imshow(psf_fit, origin='lower', cmap='viridis', norm=norm)
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('PSF Response')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')
        ax = plt.subplot(gs[2, j])
        ratio = np.zeros_like(psf_fit)
        ratio[psf_image != 0] = (psf_fit[psf_image != 0] - psf_image[psf_image != 0]) / \
            psf_image[psf_image != 0]
        ratio_ma = np.ma.array(ratio, mask=psf_image == 0)
        norm = simple_norm(ratio[(ratio != 0) & (psf_image > 1e-3)], 'linear', percent=100)
        cmap = plt.get_cmap('viridis')
        cmap.set_bad('w')
        img = ax.imshow(ratio_ma, origin='lower', cmap=cmap, norm=norm)
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('Relative Difference')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')

        ax = plt.subplot(gs[3, j])
        ratio = psf_fit - psf_image
        norm = simple_norm(ratio, 'linear', percent=100)
        img = ax.imshow(ratio, origin='lower', cmap='viridis', norm=norm)
        cb = plt.colorbar(img, ax=ax, use_gridspec=True)
        cb.set_label('Absolute Difference')
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')

    plt.tight_layout()
    plt.savefig('out_gals/test_psf_mog.pdf')


if __name__ == '__main__':
    psf_names = ['../../pandeia_data-1.0/wfirst/wfirstimager/psfs/wfirstimager_any_{}.fits'.format(num) for num in [0.8421, 1.0697, 1.4464, 1.2476, 1.5536, 1.9068]]
    psf_mog_fitting(psf_names)
