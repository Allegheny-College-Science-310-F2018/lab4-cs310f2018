# USAGE
# python blurring.py --image images/imagefile

# Import the necessary packages
import numpy as np
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image")
args = vars(ap.parse_args())

# Load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

#######################start here#################################

# Let's apply standard averaging blurring first. Average
# blurring (as the name suggests), takes the average of all
# pixels in the surrounding area and replaces the central
# element of the output image with the average. Thus, in
# order to have a central element, the area surrounding the
# central must be odd. Here are a few examples with varying
# kernel sizes. Notice how the larger the kernel gets, the
# more blurred the image becomes

# we are going to define a k x k sliding window on top of
#our image, where k is always an odd
#number. This window is going to slide from left-to-right
#and from top-to-bottom. The pixel at the center of this
#matrix (we have to use an odd number, otherwise there would
#not be a true center) is then set to be the average of all
#other pixels surrounding it.
#sliding window is called kernel, as kernel size increases - more blurred
#cv2.blur function requires two arguments: the image
#we want to blur and the size of the kernel.
#we blur our image with increasing-sized kernels '''
blurred = np.hstack([ # np.stack function to horiz stack output images together
	cv2.blur(image, (3, 3)),
	cv2.blur(image, (5, 5)),
	cv2.blur(image, (7, 7))])
cv2.imshow("Averaged", blurred)
cv2.waitKey(0)

# We can also apply Gaussian blurring, where the relevant
# parameters are the image we want to blur and the standard
# deviation in the X and Y direction. Again, as the standard
# deviation size increases, the image becomes progressively
# more blurred
''' Gaussian blurring is similar to average blurring, but instead of
using a simple mean, we are now using a weighted mean,
where neighborhood pixels that are closer to the central
pixel contribute more weight to the average.
The end result is that our image is less blurred, but more
naturally blurred, than using the average method'''
blurred = np.hstack([
## cv2.GaussianBlur function: the first argument
# to the function is the image we want to blur, then,
# similar to cv2.blur, we provide a tuple representing our kernel size
# last parameter is the standard deviation (amount of variation) in the
# x-axis direction. By setting this value to 0, we are instructing OpenCV
# to automatically compute them based on our kernel size.
	cv2.GaussianBlur(image, (3, 3), 0),
	cv2.GaussianBlur(image, (5, 5), 0),
	cv2.GaussianBlur(image, (7, 7), 0)])
cv2.imshow("Gaussian", blurred)
cv2.waitKey(0)

# The cv2.medianBlur function is mainly used for removing
# what is called "salt-and-pepper" noise. Unlike the Average
# method mentioned above, the median method (as the name
# suggests), calculates the median pixel value amongst the
# surrounding area.
''' When applying a median blur, we first define our kernel
size k. Then, as in the averaging blurring method, we
consider all pixels in the neighborhood of size k x k. But, unlike
the averaging method, instead of replacing the central pixel
with the average of the neighborhood, we instead replace
the central pixel with the median of the neighborhood.
Median blurring is more effective at removing salt-and-
pepper style noise from an image because each central pixel
is always replaced with a pixel intensity that exists in the
image.'''
blurred = np.hstack([
# cv2.medianBlur function takes two parameters:
# the image we want to blur and the size of our kernel.
	cv2.medianBlur(image, 3),
	cv2.medianBlur(image, 5),
	cv2.medianBlur(image, 7)])
cv2.imshow("Median", blurred)
cv2.waitKey(0)

# You may have noticed that blurring can help remove noise,
# but also makes edge less sharp. In order to keep edges
# sharp, we can use bilateral filtering. We need to specify
# the diameter of the neighborhood (as in examples above),
# along with sigma values for color and coordinate space.
# The larger these sigma values, the more pixels will be
# considered within the neighborhood.
'''Thus far, the intention of our blurring methods has been
to reduce noise and detail in an image; however, we tend to
lose edges in the image.
In order to reduce noise while still maintaining edges, we
can use bilateral blurring. Bilateral blurring accomplishes
this by introducing two Gaussian distributions.
The first Gaussian function only considers spatial
neighbors, that is, pixels that appear close together in the ( x, y )
coordinate space of the image. The second Gaussian then
models the pixel intensity of the neighborhood, ensuring
that only pixels with similar intensity are included in the
actual computation of the blur.
Overall, this method is able to preserve edges of an im-
age, while still reducing noise. The largest downside to this
method is that it is considerably slower than its averaging,
Gaussian, and median blurring counterparts.'''
blurred = np.hstack([
# The first parameter we supply
# is the image we want to blur. Then, we need to define the
# diameter of our pixel neighborhood. The third argument
# is our color. A larger value for color means that more
# colors in the neighborhood will be considered when computing the blur.
# Finally, we need to supply the space. A
# larger value of space means that pixels farther out from
# the central pixel will influence the blurring calculation, provided
# that their colors are similar enough.
	cv2.bilateralFilter(image, 5, 21, 21),
	cv2.bilateralFilter(image, 7, 31, 31),
	cv2.bilateralFilter(image, 9, 41, 41)])
cv2.imshow("Bilateral", blurred)
cv2.waitKey(0)
