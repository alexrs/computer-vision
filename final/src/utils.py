"""
Authors: Alejandro Rodriguez, Fernando Collado
"""
import os

class Utils(object):
    """
    The code that does not fit anywhere else
    """

    @staticmethod
    def create_dir(dirname):
        """
        Creates a directory if it does not exists
        """
        if not os.path.exists(dirname):
            os.makedirs(dirname)

