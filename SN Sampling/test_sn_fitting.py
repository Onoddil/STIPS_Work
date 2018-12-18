import numpy as np
from astropy.table import Table
import sn_sampling as sns
import sncosmo
import astropy.io.fits as pyfits
import astropy.units as u

filters = ['z087', 'y106', 'w149', 'j129', 'h158', 'f184']
filt_zp = [26.39, 26.41, 27.50, 26.35, 26.41, 25.96]
pixel_scale = 0.11
for j in range(0, len(filters)):
    f = pyfits.open('../../pandeia_data-1.0/wfirst/wfirstimager/filters/{}.fits'
                    .format(filters[j]))
    data = f[1].data
    dispersion = np.array([d[0] for d in data])
    transmission = np.array([d[1] for d in data])
    # both F184 and W149 extend 0.004 microns into 2 microns, beyond the wavelength range of the
    # less extended models, 19990A, or 1.999 microns. Thus we slightly chop the ends off these
    # filters, and set the final 'zero' to 1.998 microns:
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

# filters = ['F160W']
# filt_zp = [25.95]  # [27.69, 28.02, 28.19] - st; [26.27, 26.23, 25.95] - ab
# pixel_scale = 0.13

exptime = 1000  # seconds
sn_type = 'Ia'
sn_types = ['Ia', 'Ib', 'Ic', 'II']

t0 = 50000

t_low, t_high, t_interval = -0, 10, 5
times = np.arange(t_low, t_high+1e-10, t_interval)

z_low, z_high = 0.2, 0.7
z = np.random.uniform(z_low, z_high)

sn_model = sns.get_sn_model(sn_type, 1, t0=t0, z=z)
sn_model.set_source_peakabsmag(-19.0, 'f125w', 'ab')

time_array = []
band_array = []
flux_array = []
fluxerr_array = []
zp_array = []
zpsys_array = []

NiaNcc, NibcNii, NicNib = 0.44, 0.36, 2.12
fia, fcc = NiaNcc / (1 + NiaNcc), 1 - NiaNcc / (1 + NiaNcc)
fibc, fii = fcc * NibcNii / (1 + NibcNii), fcc * (1 - NibcNii / (1 + NibcNii))
fib, fic = fibc * (1 - NicNib / (1 + NicNib)), fibc * NicNib / (1 + NicNib)
sn_priors = np.array([fia, fib, fic, fii])

directory = 'out_gals'
for j in range(0, len(filters)):
    for k in range(0, len(times)):
        time = times[k] + t0
        bandpass = sncosmo.get_bandpass(filters[j])
        m_ia = sn_model.bandmag(bandpass, magsys='ab', time=time)
        if np.isnan(m_ia):
            m_ia = -2.5 * np.log10(0.01) + filt_zp[j]
        time_array.append(time)
        band_array.append(filters[j])
        f = 10**(-1/2.5 * (m_ia - filt_zp[j]))
        if f < 0:
            f = 1e-5
        flux_array.append(f)
        fluxerr_array.append(np.sqrt(f))
        zp_array.append(filt_zp[j])
        zpsys_array.append('ab')

lc_data = [np.array(time_array), np.array(band_array), np.array(flux_array),
           np.array(fluxerr_array), np.array(zp_array), np.array(zpsys_array)]

param_names = ['z', 't0']
if sn_type == 'Ia':
    param_names += ['x0', 'x1', 'c']
else:
    param_names += ['amplitude']
sn_params = [sn_model[q] for q in param_names]

lc_data_table = Table(data=lc_data,
                      names=['time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'])

figtext = []
if sn_type == 'Ia':
    z_, t_, x0_, x1_, c_ = sn_params
    figtext.append('Type {}: $z = {:.3f}$\n$t_0 = {:.1f}$\n'
                   '$x_0 = {:.5f}$'.format(sn_type, z_, t_, x0_))
    figtext.append('$x_1 = {:.5f}$\n$c = {:.5f}$'.format(x1_, c_))
else:
    z_ = sn_params[0]
    t_ = sn_params[1]
    A_ = sn_params[2]
    A_sig = int(np.floor(np.log10(abs(A_))))
    figtext.append('Type {}: $z = {:.3f}$\n$t_0 = {:.1f}$'.format(
                   sn_type, z_, t_))
    figtext.append('$A = {:.3f} \\times 10^{{{}}}$'.format(A_/10**A_sig, A_sig))
sns.fit_lc(lc_data_table, sn_types, directory, filters, 'test', figtext, min(3, len(filters)),
           3, sn_priors, filt_zp)
