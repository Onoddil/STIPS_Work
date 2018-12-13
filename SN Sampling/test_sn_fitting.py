import numpy as np
from astropy.table import Table
import sn_sampling as sns
import sncosmo

filters = ['F160W']
filt_zp = [25.95]  # [27.69, 28.02, 28.19] - st; [26.27, 26.23, 25.95] - ab
pixel_scale = 0.13

exptime = 1000  # seconds
sn_type = 'Ia'

times = np.arange(-15, 41, 5)


z_low, z_high = 0.2, 1.0
z = np.random.uniform(z_low, z_high)

sn_model = sns.get_sn_model(sn_type, 1, t0=0.0, z=z)
sn_model.set_source_peakabsmag(-19.0, 'f125w', 'ab')

time_array = []
band_array = []
flux_array = []
fluxerr_array = []
zp_array = []
zpsys_array = []

j = 0
i = 0
directory = 'out_gals'
for k in range(0, len(times)):
    time = times[k]
    bandpass = sncosmo.get_bandpass(filters[j])
    m_ia = sn_model.bandmag(bandpass, magsys='ab', time=time)

    time_array.append(time)
    band_array.append(filters[j])
    f = 10**(-1/2.5 * (m_ia - filt_zp[j]))
    flux_array.append(f + np.random.normal(0, np.sqrt(f)))
    fluxerr_array.append(np.sqrt(f))
    zp_array.append(filt_zp[j])  # filter-specific zeropoint
    # TODO: swap to STmag from the AB system
    zpsys_array.append('ab')
lc_data = [np.array(time_array), np.array(band_array), np.array(flux_array),
           np.array(fluxerr_array), np.array(zp_array), np.array(zpsys_array)]

sn_params = [sn_model['z'], sn_model['t0'], sn_model['x0'], sn_model['x1'], sn_model['c']]

lc_data_table = Table(data=lc_data,
                      names=['time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'])
print(lc_data_table['flux'])
figtext = 'z = {:.3f}\nt0 = {:.1f}\nx0 = {:.5f}x1 = {:.5f}\nc = {:.5f}'.format(*sn_params)
figtext_split = figtext.split('x1')
figtext = [figtext_split[0], 'x1' + figtext_split[1]]
sns.fit_lc(lc_data_table, [sn_type], directory, filters, 'test', figtext, min(3, len(filters)))
