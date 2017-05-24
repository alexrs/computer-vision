"""
Authors: Alejandro Rodriguez, Fernando Collado
"""

# maximum allowed iterations to fit the model
MAX_ITER = 40

class Fit(object):
    """
    TODO
    """

    def __init__(self, initial_fit, img, points):
        self._initial_fit = initial_fit
        self._img = img
        self._points = points
