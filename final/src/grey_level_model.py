"""
Authors: Alejandro Rodriguez, Fernando Collado
See:
    [1] http://www.bmva.org/bmvc/1993/bmvc-93-064.pdf
    [2] http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4462625
    [3] https://pdfs.semanticscholar.org/eb6a/f7f2c40d5f7f86e6acf140fc3040135fd43c.pdf
"""

import numpy as np

class GreyLevelModel(object):
    """
    For every landmark point j in the image i of the
    training set, we extract a gray level profile gij , of length
    np pixels, centered around the landmark point. We do
    not use the actual gray level profile but its normalized
    derivative. This gives invariance to the offsets and
    uniform scaling of the gray levels. (From [3])
    """

    def __init__(self):
        pass

    def get_modes(self):
        """
        TODO
        """
        pass

    def mahalanobis(self, samples):
        """
        mahalanobis distance returns the quality of the fit
        See:
            https://en.wikipedia.org/wiki/Mahalanobis_distance
        """
        return (samples - self.mean_profile).T.dot(self.covariance).dot(samples - self.mean_profile)


class Profile(object):

    def __init__(self):
        pass


    def _normal(self, p1, p2):
        """
        returns the normal (-dy, dx)
        See:
            http://stackoverflow.com/a/1243676/1397152
        """
        return np.array([p1[1] - p2[1], p2[0] - p1[0]])
