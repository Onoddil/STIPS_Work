from __future__ import division
import matplotlib.gridspec as gridspec
import numpy as np

import matplotlib.pyplot as plt
import scipy.optimize
from scipy.special import erf


def gridcreate(name, y, x, ratio, z, **kwargs):
    # Function that creates a blank axis canvas; each figure gets a name (or alternatively a number
    # if none is given), and gridspec creates an N*M grid onto which you can create axes for plots.
    # This returns a gridspec "instance" so you can specific which figure to put the axis on if you
    # have several on the go.
    plt.figure(name, figsize=(z*x, z*ratio*y))
    gs = gridspec.GridSpec(y, x, **kwargs)
    return gs


def fitg(p, x, dx):
    x0 = p[0]
    c = p[1]
    f = 0.5 * (erf((x+dx - x0)/(np.sqrt(2)*c)) - erf((x - x0)/(np.sqrt(2)*c)))
    return f


def sumg(p, x, y, dx, o):
    return np.sum((y - fitg(p, x, dx))**2 / o**2)


def gradg(p, x, y, dx, o):
    f = fitg(p, x, dx)
    x0 = p[0]
    c = p[1]
    dfdx0 = (np.exp(-0.5*(x-x0)**2/c**2) - np.exp(-0.5 * (x+dx-x0)**2/c**2)) / (np.sqrt(2*np.pi)*c)
    dfdc = ((x-x0) * np.exp(-0.5*(x-x0)**2/c**2) -
            (x+dx-x0) * np.exp(-0.5 * (x+dx-x0)**2/c**2)) / (np.sqrt(2*np.pi)*c**2)
    return np.array([np.sum(-2 * (y - f) * i / o**2) for i in [dfdx0, dfdc]])


gs = gridcreate('111', 1, 2, 0.8, 15)
ax = plt.subplot(gs[0])

a = np.genfromtxt('salt2-guy2010-parameters.txt', usecols=4, comments='#')
hist, bins = np.histogram(a)

x = bins[:-1]
dx = np.diff(bins)
o = np.sqrt(hist) / np.sum(hist)
o[o == 0] = 10
y = hist / np.sum(hist)

output2 = scipy.optimize.minimize(sumg, x0=np.array([-0.1, 1]), args=(x, y, dx, o),
                                  jac=gradg, method='newton-cg',
                                  options={'maxiter': 10000, 'xtol': 1e-6})
x0, c = output2.x
ax.plot(bins, np.append(np.sum(hist) * fitg([x0, c], x, dx), 0), 'r-',
        label=r'x$_0$ = {:.2f}, $\sigma$ = {:.2f}'.format(x0, c), drawstyle='steps-post')
x_ = np.linspace(bins[0], bins[-1], 500)
dx_ = np.mean(np.diff(bins))

ax.plot(x_, np.sum(hist)/(np.sqrt(2*np.pi)*c) * np.exp(-0.5 * (x_ - x0)**2 / c**2)*dx_, 'r--')

ax.plot(bins, np.append(hist, 0), 'k-', drawstyle='steps-post')
ax.legend()
ax.set_xlabel('X1')
ax.set_ylabel('Counts')

ax = plt.subplot(gs[1])

a = np.genfromtxt('salt2-guy2010-parameters.txt', usecols=6, comments='#')
hist, bins = np.histogram(a)

x = bins[:-1]
dx = np.diff(bins)
o = np.sqrt(hist) / np.sum(hist)
o[o == 0] = 10
y = hist / np.sum(hist)

output2 = scipy.optimize.minimize(sumg, x0=np.array([0, 1]), args=(x, y, dx, o),
                                  jac=gradg, method='newton-cg',
                                  options={'maxiter': 10000, 'xtol': 1e-6})
x0, c = output2.x
ax.plot(bins, np.append(np.sum(hist) * fitg([x0, c], x, dx), 0), 'r-',
        label=r'x$_0$ = {:.3f}, $\sigma$ = {:.3f}'.format(x0, c), drawstyle='steps-post')
x_ = np.linspace(bins[0], bins[-1], 500)
dx_ = np.mean(np.diff(bins))

ax.plot(x_, np.sum(hist)/(np.sqrt(2*np.pi)*c) * np.exp(-0.5 * (x_ - x0)**2 / c**2)*dx_, 'r--')

ax.plot(bins, np.append(hist, 0), 'k-', drawstyle='steps-post')
ax.set_xlabel('C')
ax.set_ylabel('Counts')
ax.legend()

plt.tight_layout()
plt.savefig('salt2_parameters.pdf')
