import numpy, scipy
from scipy import ndimage
a = scipy.misc.imread("rgb.jpg")
b = ((a[:,:,0] == 255) * (a[:,:,1] == 0) * (a[:,:,2] == 0))*1 # Imports RGB to numpy array where a[0] is red, a[1] is blue, a[2] is green...
num_red = numpy.sum((a[:,:,0] == 255) * (a[:,:,1] == 0) * (a[:,:,2] == 0)) # Counts the number of pure red pixels
labeled_array, num_features = scipy.ndimage.measurements.label(b.astype('Int8'))