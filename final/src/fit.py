"""
Authors: Alejandro Rodriguez, Fernando Collado
"""

# maximum allowed iterations to fit the model
MAX_ITER = 40
SEARCH_POINTS = 15 # Number of points to search when fitting the image

class Fit(object):
    """
    TODO
    """

    def __init__(self, initial_fit, img):
        self._initial_fit = initial_fit
        self._img = img
