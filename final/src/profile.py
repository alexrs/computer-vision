"""
Authors: Alejandro Rodriguez, Fernando Collado
See:
    [1] http://ac.els-cdn.com/0167865587900213/1-s2.0-0167865587900213-main.pdf?_tid=02180bde-4000-11e7-84ae-00000aab0f02&acdnat=1495575624_4b2abdfe3b391de15331c5d884462bcf
    [2] http://download.springer.com/static/pdf/658/bok%253A978-3-642-54851-2.pdf?originUrl=http%3A%2F%2Flink.springer.com%2Fbook%2F10.1007%2F978-3-642-54851-2&token2=exp=1495577732~acl=%2Fstatic%2Fpdf%2F658%2Fbok%25253A978-3-642-54851-2.pdf%3ForiginUrl%3Dhttp%253A%252F%252Flink.springer.com%252Fbook%252F10.1007%252F978-3-642-54851-2*~hmac=cbf933680654186fafd57bed1da3ed80fa5ffef7dea1de3f47d61ac90f9fe6f9
"""
import math
import numpy as np
from utils import K

class Profile(object):
    """
    This class represents a profile, used for calculating
    the Grey Level Model
    """

    def __init__(self, index, shape, img, enhanced_img, width=K):
        self._point = shape.data()[index]
        self._img = img
        self._enhanced_img = enhanced_img
        self._width = width
        self._normal = self._compute_normal(shape.data()[(index - 1) % 40], # mod to avoid overflow
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
        # sample k pixels on either side of the landmarks
        pos_points, pos_values, pos_enhanced = self._get_intensity(-self._normal)
        neg_points, neg_values, neg_enhanced = self._get_intensity(self._normal)

        points = np.vstack((neg_points[::-1], pos_points[1:]))
        values = np.append(neg_values[::-1], pos_values[1:])

        # 2k+1
        grey_values = np.append(neg_enhanced[::-1], pos_enhanced[1:])
        # derivatives, length 2k
        grey_values = [grey_values[i] - grey_values[i - 1] for i in range(1, len(grey_values))]

        # Finally, normalize
        factor = sum([math.fabs(v) for v in values])
        samples = [g/factor for g in grey_values]

        return points, samples

    def samples(self):
        """
        return samples
        """
        return self._samples

    def points(self):
        """
        return points
        """
        return self._points

    def _compute_normal(self, point1, point2):
        """
        See: http://stackoverflow.com/a/1243676/1397152
        receives the previous point in the model (point1) and the
        next point (point2) and calculates the normal between those two points
        to get the normal that pass through self._point
        """
        # normal (-dy, dx)
        norm_1 = np.array([point1[1] - self._point[1], self._point[0] - point1[0]])
        norm_2 = np.array([self._point[1] - point2[1], point2[0] - self._point[0]])

        norm = (norm_1 + norm_2) / 2
        # normalize the vector
        return norm / np.linalg.norm(norm)

    def _get_coordinates(self, normal):
        """
        See:
        np.newaxis - https://stackoverflow.com/a/29868617/1397152
        """
        point1 = self._point
        point2 = self._point + normal * self._width
        # get width + 1 points equally spaced between 1 and 0, and  0 and 1
        coordinates = np.array(point1[:, np.newaxis] * np.linspace(1, 0, self._width + 1) +
                               point2[:, np.newaxis] * np.linspace(0, 1, self._width + 1))
        return coordinates

    def _get_intensity(self, normal):
        """
        See:
        cast to int -
             https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.ndarray.astype.html
        """
        coords = self._get_coordinates(normal)
        values = self._img[coords[1].astype(int), coords[0].astype(int)]
        enhanced_values = self._enhanced_img[coords[1].astype(int), coords[0].astype(int)]
        return coords.T, values, enhanced_values
