from __future__ import division
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
filters = ['f184', 'h158', 'j129', 'w149', 'y106', 'z087']
colours = ['k', 'r', 'b', 'g', 'c', 'm', 'orange']
for filt, c in zip(filters, colours):
    f = pyfits.open('../pandeia_data-1.0/wfirst/wfirstimager/filters/{}.fits'.format(filt))
    data = f[1].data
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    ax.plot(x, y, c=c, label=filt)

ax.legend()
ax.set_xlabel('Wavelength / $\mu$m')
ax.set_ylabel('Throughput')
plt.tight_layout()
plt.savefig('wfirst_filters.pdf')
