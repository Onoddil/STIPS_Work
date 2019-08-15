import sncosmo
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import sn_sampling as sns
import sn_sampling_extras as snse


def gridcreate(name, y, x, ratio, z, **kwargs):
    # Function that creates a blank axis canvas; each figure gets a name (or alternatively a number
    # if none is given), and gridspec creates an N*M grid onto which you can create axes for plots.
    # This returns a gridspec "instance" so you can specific which figure to put the axis on if you
    # have several on the go.
    plt.figure(name, figsize=(z*x, z*ratio*y))
    gs = gridspec.GridSpec(y, x, **kwargs)
    return gs


z = 0.2
bands = np.array(['z087', 'y106', 'w149', 'j129', 'h158', 'f184'])
snse.register_filters(bands)
a = sncosmo.models._SOURCES.get_loaders_metadata()

gs = gridcreate('123123', 2, 3, 0.8, 5)
axs = [plt.subplot(gs[i]) for i in range(0, 6)]
cs = ['k', 'r', 'b', 'g', 'c', 'm', 'orange', 'brown']

typings = ['Ia', 'Ib', 'Ic', 'IIP', 'IIL', 'IIn', 'Iat', 'Iabg']
set_models = [0, 1, 1, 0, 1, 1, 1, 1]

mintime, maxtime = 999, -999
for typing in typings:
    sn_model = sns.get_sn_model(typing, 1, t0=0, z=z)
    sn_model.set_source_peakabsmag(-19.0, 'f125w', 'ab')
    mintime, maxtime = min(mintime, sn_model.mintime()), max(maxtime, sn_model.maxtime())
mintime -= 50
maxtime += 50
t_max = np.linspace(mintime, maxtime, 1000)

for typing, c, set_model in zip(typings, cs, set_models):
    sn_model = sns.get_sn_model(typing, 1, t0=0, z=z)
    sn_model.set_source_peakabsmag(-19.0, 'f125w', 'ab')
    t = np.linspace(sn_model.mintime(), sn_model.maxtime(), 1000)
    for band, ax in zip(bands, axs):
        f = sn_model.bandflux(band, time=t)
        ax.plot(t, f, ls='-', c=c, label=typing + '*' if set_model else typing)
        f = sn_model.bandflux(band, time=t_max)
        ax.plot(t_max, f, ls='--', c=c)
        ax.axvline(sn_model.maxtime() if 'L' not in typing and 'n' not in typing else
                   150*(1+z)+np.random.uniform(-5, 5), ls='-', c=c)
    if not set_model:
        print(typing)
        print('===============')
        for i in a:
            if typing in i['type']:
                m = sncosmo.Model(source=i['name'])
                m.set(z=z)
                if m.maxtime() >= 50 and m.maxwave() >= 20000:
                    try:
                        print(m._source.name, i['type'], m.mintime(), m.maxtime(),
                              m.minwave(), m.maxwave(), [m._source.peakphase(band) for band in bands])
                    except ValueError:
                        pass
sn_model = sncosmo.Model('nugent-sn2p')
sn_model.set(t0=0, z=z)
sn_model.set_source_peakabsmag(-19.0, 'f125w', 'ab')
t = np.linspace(sn_model.mintime(), sn_model.maxtime(), 1000)
for band, ax in zip(bands, axs):
    f = sn_model.bandflux(band, time=t)
    ax.plot(t, f, ls='-', c='orange', label='IIP2')
    f = sn_model.bandflux(band, time=t_max)
    ax.plot(t_max, f, ls='--', c='orange')
axs[0].legend()
for ax in axs:
    ax.axvline(150, ls=':', c='k', lw=1.5)
    ax.axhline(0, ls=':', c='k', lw=1.5)
    ax.set_xlabel('time')
    ax.set_ylabel('flux')
plt.tight_layout()
plt.savefig('test_model_phases.pdf')
