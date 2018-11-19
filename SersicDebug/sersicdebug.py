path = '../../STScI-STIPS'
import sys
sys.path.insert(1, path)
import stips
import logging
from stips.scene_module import SceneModule
from stips.observation_module import ObservationModule
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.io.fits as pyfits
from astropy.visualization import simple_norm
import numpy as np

def gridcreate(name, y, x, ratio, z, **kwargs):
    # Function that creates a blank axis canvas; each figure gets a name (or alternatively a number
    # if none is given), and gridspec creates an N*M grid onto which you can create axes for plots.
    # This returns a gridspec "instance" so you can specific which figure to put the axis on if you
    # have several on the go.
    plt.figure(name, figsize=(z*x, z*ratio*y))
    gs = gridspec.GridSpec(y, x, **kwargs)
    return gs

file_ = open('galaxy_creation_test.log', 'w+')
stream_handler = logging.StreamHandler(file_)
stream_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)

scm = SceneModule(logger=logger)

seedg = 9999
galaxy = {'n_gals': 50,
      'z_low': 0.0, 'z_high': 1.0,
      'rad_low': 0.5, 'rad_high': 2.0,
      'sb_v_low': 23.0, 'sb_v_high': 20.0,
      'distribution': 'uniform', 'clustered': False,
      'radius': 5, 'radius_units': 'arcsec',
      'offset_ra': 0.00001, 'offset_dec': 0.00001, 'seed': seedg}
galaxy_cat_file = scm.CreateGalaxies(galaxy)
seedo = 12345
obs = {'instrument': 'WFI',
       'filters': ['Z087'],
       'detectors': 1,
       'distortion': False,
       'oversample': 1,
       'pupil_mask': '',
       'background': 'avg',
       'observations_id': 1,
       'exptime': 1000,
       'offsets': [{'offset_id': 1, 'offset_centre': False, 'offset_ra': 0.0, 'offset_dec': 0.0, 'offset_pa': 0.0}],
       'small_subarray': True, 'seed': seedo}

obm = ObservationModule(obs, logger=logger)
obm.nextObservation()

output_galaxy_catalogues = obm.addCatalogue(galaxy_cat_file)
psf_file = obm.addError()
fits_file, mosaic_file, params = obm.finalize(mosaic=False)

gs = gridcreate('m', 1, 1, 0.8, 15)
ax = plt.subplot(gs[0])

f = pyfits.open(fits_file)
image = f[1].data
print "total in final image:", np.sum(image)
norm = simple_norm(image, 'log', min_percent=50, max_percent=99.95)

ax = plt.subplot(gs[0])
img = ax.imshow(image, origin='lower', cmap='viridis', norm=norm)
plt.colorbar(img, ax=ax, use_gridspec=True)
ax.set_xlabel('x / pixel')
ax.set_ylabel('y / pixel')
plt.tight_layout()
plt.savefig('galaxies.pdf')
plt.close()