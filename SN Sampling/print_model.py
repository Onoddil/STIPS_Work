import numpy as np
from astropy.table import Table
import sn_sampling as sns
import sncosmo
import astropy.io.fits as pyfits
import astropy.units as u
import matplotlib.pyplot as plt

filters = ['z087', 'y106', 'w149', 'j129', 'h158', 'f184']
filt_zp = [26.39, 26.41, 27.50, 26.35, 26.41, 25.96]
pixel_scale = 0.11
for j in range(0, len(filters)):
    f = pyfits.open('../../pandeia_data-1.0/wfirst/wfirstimager/filters/{}.fits'
                    .format(filters[j]))
    data = f[1].data
    dispersion = np.array([d[0] for d in data])
    transmission = np.array([d[1] for d in data])
    # both F184 and W149 extend 0.004 microns into 2 microns, beyond the wavelength range of
    # the less extended models, 19990A, or 1.999 microns. Thus we slightly chop the ends off
    # these filters, and set the final 'zero' to 1.998 microns:
    if filters[j] == 'f184' or filters[j] == 'w149':
        ind_ = np.where(dispersion < 1.999)[0][-1]
        transmission[ind_:] = 0
    q_ = np.argmax(transmission)
    if transmission[q_] == transmission[q_+1]:
        q_ += 1
    imin = np.where(transmission[:q_] == 0)[0][-1]
    imax = np.where(transmission[q_:] == 0)[0][0] + q_ + 1
    bandpass = sncosmo.Bandpass(dispersion[imin:imax], transmission[imin:imax], wave_unit=u.micron,
                                name=filters[j])
    sncosmo.register(bandpass)

sn_type = 'Ib'
t0 = 0
z = np.random.uniform(0.2, 0.7)
sn_model = sns.get_sn_model(sn_type, 1, t0=t0, z=z)
# sn_model.set_source_peakabsmag(-19.0, 'f125w', 'ab')
print(z)
wave = np.arange(3000, 20000, 0.1)

p = 0
gs = sns.gridcreate('1', 1, 1, 0.8, 15)
ax = plt.subplot(gs[0])
ax2 = ax.twinx()
for t, c in zip([-5, 0, 5, 10], ['k', 'r', 'g', 'b']):
    fluxes = sn_model.flux(t+t0, wave)
    ax.plot(wave, fluxes, ls='-', c=c)
    p = max(np.amax(fluxes), p)
    for filter_, zp, m in zip(filters, filt_zp, ['x', '*', '+', '^', '.', 'o']):
        b = sncosmo.get_bandpass(filter_)
        print(filter_, t, sn_model.bandflux(filter_, t+t0, 25, 'ab'))
        ax2.plot(b.wave_eff, sn_model.bandflux(filter_, t+t0, 25, 'ab'), marker=m, c=c)

for dx, filter_, c in zip([-0.1, -0.05, 0, 0.05, 0.1, 0.15], filters,
                          ['k', 'r', 'g', 'b', 'c', 'orange']):
    b = sncosmo.get_bandpass(filter_)
    ax.plot(b.wave, b.trans*(0.8+dx)/np.amax(b.trans)*p, ls='-', c=c)

ax.set_xlabel(r'$\lambda$ / $\AA$')
ax.set_ylabel('Flux')
plt.tight_layout()
plt.savefig('spectrum_test.pdf')
