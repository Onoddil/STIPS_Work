import numpy as np
import psf_mog_fitting as pmf
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm


def gridcreate(name, y, x, ratio, z, **kwargs):
    # Function that creates a blank axis canvas; each figure gets a name (or alternatively a number
    # if none is given), and gridspec creates an N*M grid onto which you can create axes for plots.
    # This returns a gridspec "instance" so you can specific which figure to put the axis on if you
    # have several on the go.
    plt.figure(name, figsize=(z*x, z*ratio*y))
    gs = gridspec.GridSpec(y, x, **kwargs)
    return gs


filters = ['z087', 'y106', 'w149', 'j129', 'h158', 'f184']
psf_comp = np.load('../PSFs/wfirst_psf_comp.npy')
dx, dy = np.arange(-0.5, 0.5+1e-10, 0.01), np.arange(-0.5, 0.5+1e-10, 0.01)
dx_pc, dy_pc = np.append(dx - 0.005, dx[-1] + 0.005), np.append(dy - 0.005, dy[-1] + 0.005)
x_int, y_int = np.arange(-20, 20.1, 1), np.arange(-20, 20.1, 1)
gs = gridcreate('a', 1, 6, 1, 5)
for k in range(0, len(filters)):
    ax = plt.subplot(gs[k])
    psf_sum = np.empty((len(dx), len(dy)), float)
    p = psf_comp[k].reshape(-1)
    for i in range(0, len(dx)):
        for j in range(0, len(dy)):
            psf_sum[i, j] = np.sum(pmf.psf_fit_fun(p, x_int+dx[i], y_int+dy[j]))
    norm = simple_norm(psf_sum, 'linear', percent=100)
    img = ax.pcolormesh(dx_pc, dy_pc, psf_sum.T, cmap='viridis', norm=norm, edgecolors='face', shading='flat')
    cb = plt.colorbar(img, ax=ax, use_gridspec=True)
    cb.set_label('PSF Normalisation')
    ax.set_xlabel('dx')
    ax.set_ylabel('dy')
plt.tight_layout()
plt.savefig('test_normalisation.pdf')
