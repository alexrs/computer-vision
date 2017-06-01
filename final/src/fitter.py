"""
Authors: Alejandro Rodriguez, Fernando Collado
"""
import math
import numpy as np
from profile import Profile
from shape import Shape
from procrustes import Procrustes
from plot import Plot
from utils import K, M

MAX_ITER = 40 # maximum allowed iterations to fit the model
MAX_ITER_UPDATE = 20 # maximum allowed iterations to update the parameters

class Fitter(object):
    """
    See:
    Page 19 - http://download.springer.com/static/pdf/658/bok%253A978-3-642-54851-2.pdf?originUrl=http%3A%2F%2Flink.springer.com%2Fbook%2F10.1007%2F978-3-642-54851-2&token2=exp=1495577732~acl=%2Fstatic%2Fpdf%2F658%2Fbok%25253A978-3-642-54851-2.pdf%3ForiginUrl%3Dhttp%253A%252F%252Flink.springer.com%252Fbook%252F10.1007%252F978-3-642-54851-2*~hmac=cbf933680654186fafd57bed1da3ed80fa5ffef7dea1de3f47d61ac90f9fe6f9
    https://pdfs.semanticscholar.org/5b34/59be44b9eb7d8679ba348db4dfabcd5a8522.pdf
    https://pdfs.semanticscholar.org/1da2/f22f46de0726cacfbe894946ea72032e8fbc.pdf
    """

    def __init__(self, current_fit, test_img, test_img_enhanced, gl_models, pc_modes, ground_truth):
        self._current_fit = current_fit
        self._test_img = test_img
        self._test_img_enhanced = test_img_enhanced
        self._gl_models = gl_models
        self._pc_modes = pc_modes
        self._ground_truth = ground_truth

    def fit(self, tol=1e-14):
        """
        See: An Introduction to Active Shape Models, pag 9.
        """
        # Initialise the shape parameters, b, to zero
        b = np.zeros(self._pc_modes.shape[1])
        prev_fit = Shape.from_list_consecutive(np.zeros_like(self._current_fit.data()))

        for i in range(MAX_ITER):
            print "\tFitting image. Iteration: {}".format(i)
            #  Examine a region of the image around each point Xi to find the
            # best nearby match for the point
            Y = self._find(self._current_fit)
            Plot.approximated_shape([Y, self._current_fit], self._test_img, wait=False)


            # 2. Update the parameters
            # 3. Apply constraints to the parameters
            b, translation, scale, theta = self._clip(*self._update(self._current_fit, Y))


            prev_fit = self._current_fit
            t = self._current_fit.collapse() + np.dot(self._pc_modes, b)
            self._current_fit = Shape.from_list_consecutive(t).transform(translation, scale, theta)

            # Plot the current fit
            Plot.approximated_shape([self._ground_truth, self._current_fit], self._test_img, wait=False)

            if np.mean(np.abs(prev_fit.data() - self._current_fit.data())) <= tol:
                break

        return self._current_fit

    def _clip(self, b, translation, scale, theta):
        """
        clip the parameters to allowed values to avoid unallowable shapes
        """
        # clip b to +-3 sqrt(eig)
        b = np.clip(b, -3, 3)

        # clip scale
        scale = np.clip(scale, 0.8, 1.1)

        # clip rotation
        theta = np.clip(theta, -math.pi/6, math.pi/6)

        return b, translation, scale, theta

    def _find(self, current_fit):
        """
        find the best points to move the current shape
        https://books.google.es/books?id=lYxDAAAAQBAJ&pg=PA247&lpg=PA247&dq=Sample+a+profile+M+pixels+either+side+of+the+current+point&source=bl&ots=y9wvHgS7id&sig=uBQ5tVbXvLu5fFLFOEGDSuN1zKk&hl=es&sa=X&ved=0ahUKEwiF6Jal_I3UAhWGhRoKHQVuDbUQ6AEIJDAA#v=onepage&q=Sample%20a%20profile%20M%20pixels%20either%20side%20of%20the%20current%20point&f=false
        """
        best_points = []
        for i, _ in enumerate(current_fit.data()):
            # Sample a profile M pixels either side of the current point
            profile = Profile(i, current_fit, self._test_img, self._test_img_enhanced, M)

            # Test the quality of fit of the corresponding grey-level model
            # at each of the 2(m-k)+1 possible positions along the sample
            # and choose the one which gives the best match
            fmin = np.inf
            best = None
            for j in range(K, K + 2 * (M - K) + 1):
                subprofile = profile.samples()[j-K:j+K+1]
                f = self._gl_models[i].mahalanobis(subprofile)
                if f < fmin:
                    fmin = f
                    best = j

            # Choose the one which gives the best match
            best_points.append(profile.points().astype(int)[best])

        return Shape(np.array(best_points))

    def _update(self, x_model, Y, tol=1e-14):
        """
        update the parameters
        See:
            An introduction to active shape models
        """

        # 1. Initialise the shape parameters b to zero
        b = np.zeros(self._pc_modes.shape[1])
        prev_b = np.zeros_like(b)

        for _ in range(MAX_ITER_UPDATE):

            # 2. Generate the model point positions using x = x_model + Pb
            x = Shape.from_list_consecutive(x_model.collapse() + np.dot(self._pc_modes, b))

            # 3. Find the pose parameters which best align the
            # model points x to the current found points Y
            translation, scale, angle = Procrustes.align_params(x, Y)
            # 4. Project Y into the model co-ordinate frame by inverting the
            # transformation T
            y = Y.inverse_transform(translation, scale, angle)

            # 5. Project y into the tangent plane to x_model by scaling:
            # y' = y / (y*X)
            y_new = Shape.from_list_consecutive(y.collapse() / np.dot(y.collapse(), x_model.collapse().T))

            # 6. Update the model parameters to match to y': b = PT(y' - x_model)
            prev_b = b
            b = np.dot(self._pc_modes.T, (y_new.collapse() - x_model.collapse()))

            # Check for convergence
            if np.mean(np.abs(b-prev_b)) <= tol:
                break

        return b, translation, scale, angle


