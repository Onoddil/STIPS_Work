import matplotlib.gridspec as gridspec
import numpy as np

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt


def gridcreate(name, y, x, ratio, z, **kwargs):
    # Function that creates a blank axis canvas; each figure gets a name (or alternatively a number
    # if none is given), and gridspec creates an N*M grid onto which you can create axes for plots.
    # This returns a gridspec "instance" so you can specific which figure to put the axis on if you
    # have several on the go.
    plt.figure(name, figsize=(z*x, z*ratio*y))
    gs = gridspec.GridSpec(y, x, **kwargs)
    return gs


gs = gridcreate('111', 1, 1, 0.8, 15)
ax = plt.subplot(gs[0])
filters_master = ['r062', 'f184', 'h158', 'j129', 'w149', 'y106', 'z087']
colours_master = ['k', 'r', 'b', 'g', 'c', 'm', 'orange']
for j, (filt, c) in enumerate(zip(filters_master, colours_master)):
    f = pyfits.open('../../webbpsf-data/WFI/filters/{}_throughput.fits'.format(filt.upper()))
    data = f[1].data
    dispersion = np.array([d[0] * 1e-4 for d in data])
    transmission = np.array([d[1] * 0.95 for d in data])
    if filters_master[j] == 'f184' or filters_master[j] == 'w149':
        ind_ = np.where(dispersion < 1.999)[0][-1]
        dispersion[ind_+1] = 1.9998
        dispersion[ind_+2] = 1.99985
    q_ = np.argmax(transmission)
    if transmission[q_] == transmission[q_+1]:
        q_ += 1
    imin = np.where(transmission[:q_] == 0)[0][-1]
    imax = np.where(transmission[q_:] == 0)[0][0] + q_ + 1
    ax.plot(dispersion[imin:imax], transmission[imin:imax], c=c, label=filt)

ax.legend()
ax.set_xlabel(r'Wavelength / $\mu$m')
ax.set_ylabel('Throughput')
plt.tight_layout()
plt.savefig('wfirst_filters.pdf')
