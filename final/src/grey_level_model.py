"""
Authors: Alejandro Rodriguez, Fernando Collado
See:
    [1] http://www.bmva.org/bmvc/1993/bmvc-93-064.pdf
    [2] http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4462625
    [3] https://pdfs.semanticscholar.org/eb6a/f7f2c40d5f7f86e6acf140fc3040135fd43c.pdf
"""
import numpy as np
from profile import Profile

class GreyLevelModel(object):
    """
    For every landmark point j in the image i of the
    training set, we extract a gray level profile gij , of length
    np pixels, centered around the landmark point. We do
    not use the actual gray level profile but its normalized
    derivative. This gives invariance to the offsets and
    uniform scaling of the gray levels. (From [3])
    """

    def __init__(self, imgs, enhanced_imgs, shapes, index):
        self._mean = None
        self._cov = None
        self._profiles = []
        self._index = index
        self._shapes = shapes
        self._imgs = imgs
        self._enhanced_imgs = enhanced_imgs
        self._compute_grey_level_model()

    def _compute_grey_level_model(self):
        """
        """
        # For each image in the trainign set, get the profile for a given landmark
        for i, img in enumerate(self._imgs):
            profile = Profile(self._index, self._shapes[i], img, self._enhanced_imgs[i])
            self._profiles.append(profile)

        # For each profile, calculate mean and covariance matrix
        mat = []
        for profile in self._profiles:
            mat.append(profile.get_samples())
        mat = np.array(mat)
        self._mean = np.mean(mat, axis=0)
        self._cov = np.cov(mat.T)

    def get_mean(self):
        """
        """
        return self._mean
    
    def get_cov(self):
        """
        """
        return self._cov

    def mahalanobis(self, samples):
        """
        mahalanobis distance returns the quality of the fit
        See:
            https://en.wikipedia.org/wiki/Mahalanobis_distance
        """
        return (samples - self._mean).T.dot(self._cov).dot(samples - self._mean)

