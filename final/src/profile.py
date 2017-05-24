"""
Authors: Alejandro Rodriguez, Fernando Collado
See:
    [1] http://ac.els-cdn.com/0167865587900213/1-s2.0-0167865587900213-main.pdf?_tid=02180bde-4000-11e7-84ae-00000aab0f02&acdnat=1495575624_4b2abdfe3b391de15331c5d884462bcf
    [2] http://download.springer.com/static/pdf/658/bok%253A978-3-642-54851-2.pdf?originUrl=http%3A%2F%2Flink.springer.com%2Fbook%2F10.1007%2F978-3-642-54851-2&token2=exp=1495577732~acl=%2Fstatic%2Fpdf%2F658%2Fbok%25253A978-3-642-54851-2.pdf%3ForiginUrl%3Dhttp%253A%252F%252Flink.springer.com%252Fbook%252F10.1007%252F978-3-642-54851-2*~hmac=cbf933680654186fafd57bed1da3ed80fa5ffef7dea1de3f47d61ac90f9fe6f9
"""
import numpy as np
import math

WIDTH = 10 # Number of pixels for the GreyLevelModel

class Profile(object):
    """
    This class represents a profile, used for calculating
    the Grey Level Model
    """

    def __init__(self, index, shape, img, enhanced_img):
        self._point = shape.data()[index]
        self._img = img
        self._enhanced_img = enhanced_img
        self._norm = self._compute_normal(shape.data()[(index - 1) % 40], # take the mod to avoid overflow
                                          shape.data()[(index + 1) % 40])
        self._points, self._samples = self._sample()

    def _sample(self):
        """
        For each landmark, we can sample k pixels on either side of the landmark along a
        profile. Then we obtain a gray-level profile of 2k+1 (include the
        landmark itself) length. We describe it by a vector g. To reduce the effect of global
        intensity changes, we do not use the actual vector g but use the normalized
        derivative instead. (From [2])
        """
        pos_values, pos_enhanced = self._get_intensity(-self._norm)
        neg_values, neg_enhanced = self._get_intensity(self._norm)
        pos_points = self._get_coordinates(self._norm).T
        neg_points = self._get_coordinates(-self._norm).T

        print neg_values, pos_values
        values = np.append(neg_values[::-1], pos_values[1:])
        grads = np.append(neg_enhanced[::-1], pos_enhanced[1:])
        points = np.vstack((neg_points[::-1], pos_points[1:]))

        # Finally, normalize
        factor = sum([math.fabs(v) for v in values])
        samples = [float(g)/factor for g in grads]

        return points, samples

    def get_samples(self):
        """
        """
        return self._samples

    def get_points(self):
        """
        """
        return self._points

    def _compute_normal(self, p1, p2):
        """
        See: http://stackoverflow.com/a/1243676/1397152
        receives the previous point in the model (p1) and the
        next point (p2) and calculates the normal between those two points
        to get the normal that pass through self._point
        """
        # normal (-dy, dx)
        norm_1 = np.array([p1[1] - self._point[1], self._point[0] - p1[0]])
        norm_2 = np.array([self._point[1] - p2[1], p2[0] - self._point[0]])

        norm = (norm_1 + norm_2) / 2
        # normalize the vector
        return norm / np.linalg.norm(norm)

    def _get_coordinates(self, normal):
        """
        np.newaxis - https://stackoverflow.com/a/29868617/1397152
        """

        p1 = self._point
        p2 = self._point + normal * WIDTH
        coordinates = np.array(p1[:, np.newaxis] * np.linspace(1, 0, WIDTH + 1) +
                               p2[:, np.newaxis] * np.linspace(0, 1, WIDTH + 1))
        return coordinates

    def _get_intensity(self, coords):
        """
        cast to int - https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.ndarray.astype.html
        """
        values = self._img[coords[1].astype(np.int), coords[0].astype(np.int)]
        enhanced_values = self._enhanced_img[coords[1].astype(np.int), coords[0].astype(np.int)]
        return values, enhanced_values
