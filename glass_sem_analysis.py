"""
from http://scipy-lectures.github.com/
"""

import numpy as np
import pylab as pl
import matplotlib.cm as cm
from scipy import ndimage

img_id = "MV_HFV_012"

def remove_axes():
    ax = pl.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

dat = pl.imread('%s.jpg' % img_id)

# Crop the image to remove the lower panel with measure information.
dat = dat[60:]

# Slightly filter the image with a median filter in order to refine its histogram.
filtdat = ndimage.median_filter(dat, size=(7,7))
hi_dat = np.histogram(dat, bins=np.arange(256), new=False)
hi_filtdat = np.histogram(filtdat, bins=np.arange(256), new=False)
pl.plot(hi_dat[1], hi_dat[0], 'b-')
pl.plot(hi_filtdat[1], hi_filtdat[0], 'g-')
pl.xlim((-5, 260))
pl.ylim((-1000, 145000))
pl.savefig("Data/%s_histogram.png" % img_id)

# Define masks for sand pixels, glass pixels and bubble pixels
void = filtdat <= 50
sand = np.logical_and(filtdat > 50, filtdat <= 114)
glass = filtdat > 114

# Create image with each phase a different colour
phases = void.astype(np.int) + 2*glass.astype(np.int) + 3*sand.astype(np.int)
pl.clf()
pl.imshow(phases, cmap=cm.copper, origin="lower")
remove_axes()
pl.colorbar()
pl.savefig("Data/%s_phases.png" % img_id)

# Clean the phases
sand_op = ndimage.binary_opening(sand, iterations=2)
sand_labels, sand_nb = ndimage.label(sand_op)
sand_areas = np.array(ndimage.sum(sand_op, sand_labels, np.arange(sand_labels.max()+1)))
mask = sand_areas > 100
remove_small_sand = mask[sand_labels.ravel()].reshape(sand_labels.shape)

pl.clf()
pl.subplot(1, 2, 1)
pl.imshow(sand, cmap=cm.gist_gray, origin="lower")
remove_axes()
pl.subplot(1, 2, 2)
pl.imshow(remove_small_sand, cmap=cm.gist_gray, origin="lower")
remove_axes()
pl.savefig("Data/%s_sand.png" % img_id)

# Compute the mean size of bubbles.
bubbles_labels, bubbles_nb = ndimage.label(void)
bubbles_areas = np.bincount(bubbles_labels.ravel())[1:]
mean_bubble_size = bubbles_areas.mean()
median_bubble_size = np.median(bubbles_areas)
print mean_bubble_size, median_bubble_size
