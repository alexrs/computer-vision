"""
Authors: Alejandro Rodriguez, Fernando Collado
"""
from enhacement import Enhancement

class Fit(object):
    """

    """

    def __init__(self, initial_fit, img):
        self._initial_fit = initial_fit
        self._img = img


    def __fit_one_level(self, X, img, glms, m, max_iter):
        """Fit the model for one level of the image pyramid.
        """
        # Prepare test image
        enhacend_imgs = Enhancement.sobel(
            Enhancement.clahe(
                Enhancement.equalizeHist(
                    img)))
                    
        print enhacend_imgs

        # 0. Initialise the shape parameters, b, to zero (the mean shape)
        b = np.zeros(self.asm.pc_modes.shape[1])
        X_prev = Landmarks(np.zeros_like(X.points))

        # 4. Repeat until convergence.
        nb_iter = 0
        n_close = 0
        best = np.inf
        best_Y = None
        total_s = 1
        total_theta = 0
        while (n_close < 16 and nb_iter <= max_iter):

            # 1. Examine a region of the image around each point Xi to find the
            # best nearby match for the point
            Y, n_close, quality = self.__findfits(X, img, gimg, glms, m)
            if quality < best:
                best = quality
                best_Y = Y
            Plotter.plot_landmarks_on_image([X, Y], testimg, wait=False,
                                                title="Fitting incisor nr. %d" % (self.incisor_nr,))

            # no good fit found => go back to best one
            if nb_iter == max_iter:
                Y = best_Y

            # 2. Update the parameters (Xt, Yt, s, theta, b) to best fit the
            # new found points X
            b, t, s, theta = self.__update_fit_params(X, Y, testimg)

            # 3. Apply constraints to the parameters, b, to ensure plausible
            # shapes
            # We clip each element b_i of b to b_max*sqrt(l_i) where l_i is the
            # corresponding eigenvalue.
            b = np.clip(b, -3, 3)
            # t = np.clip(t, -5, 5)
            # limit scaling
            s = np.clip(s, 0.95, 1.05)
            if total_s * s > 1.20 or total_s * s < 0.8:
                s = 1
            total_s *= s
            # limit rotation
            theta = np.clip(theta, -math.pi/8, math.pi/8)
            if total_theta + theta > math.pi/4 or total_theta + theta < - math.pi/4:
                theta = 0
            total_theta += theta

            # The positions of the model points in the image, X, are then given
            # by X = TXt,Yt,s,theta(X + Pb)
            X_prev = X
            X = Landmarks(X.as_vector() + np.dot(self.asm.pc_modes, b)).T(t, s, theta)
            Plotter.plot_landmarks_on_image([X_prev, X], testimg, wait=False,
                                                title="Fitting incisor nr. %d" % (self.incisor_nr,))

            nb_iter += 1
            return X

    def __findfits(self, X, img, gimg, glms, m):
        """Examines a region of the given image around each point X_i to find
        best nearby match Y_i.

        Args:
            X (Landmarks): A landmarks model.
            img (np.ndarray): The radiograph from which X was extracted.
            glms (List[GreyLevelModel]): The grey-level models of this incisor.
            m (int): The number of samples used to find a fit.

        Returns:
            Landmarks: A Landmarks object, containing a new fit for each point in X.
            int: the percentage of times that the best found pixel along a search
                    profile is within the center 50% of the profile

        """
        fits = []
        n_close = 0

        profiles = []
        bests = []
        qualities = []
        for ind in range(len(X.points)):
            # 1. Sample a profile m pixels either side of the current point
            profile = Profile(img, gimg, X, ind, m)
            profiles.append(profile)

            # 2. Test the quality of fit of the corresponding grey-level model
            # at each of the 2(m-k)+1 possible positions along the sample
            dmin, best = np.inf, None
            dists = []
            for i in range(self.k, self.k+2*(m-self.k)+1):
                subprofile = profile.samples[i-self.k:i+self.k+1]
                dist = glms[ind].quality_of_fit(subprofile)
                dists.append(dist)
                if dist < dmin:
                    dmin = dist
                    best = i


            # 3. Choose the one which gives the best match
            bests.append(best)
            qualities.append(dmin)
            best_point = [int(c) for c in profile.points[best, :]]

            # 4. Check wheter the best found pixel along the search profile
            # is withing the central 50% of the profile
            is_upper = True if self.incisor_nr < 5 else False
            if (((is_upper and (ind > 9 and ind < 31)) or
                 (not is_upper and (ind < 11 or ind > 29))) and
                    best > 3*m/4 and best < 5*m/4):
                n_close += 1

            # Plotter.plot_fits(gimg, profile, glms[ind], dists, best_point, self.k, m)

        # remove outliers
        bests.extend(bests)
        bests = np.rint(medfilt(np.asarray(bests), 5)).astype(int)
        for best, profile in zip(bests, profiles):
            best_point = [int(c) for c in profile.points[best, :]]
            fits.append(best_point)

        # fit quality
        is_upper = True if self.incisor_nr < 5 else False
        if is_upper:
            quality = np.mean(qualities[10:30])
        else:
            quality = np.mean(qualities[0:10] + qualities[30:40])

        return Landmarks(np.array(fits)), n_close, quality

    def __update_fit_params(self, X, Y, testimg):
        """Finds the best pose (translation, scale and rotation) and shape
        parameters to match a model instance X to a new set of image points Y.

        Args:
            X: a model instance.
            Y: a new set of image points.

        Returns:
            The best pose and shape parameters.

        .. _An Introduction to Active Shape Models:
            Protocol 1 (p9)
        """

        # 1. Initialise the shape parameters, b, to zero (the mean shape).
        b = np.zeros(self.asm.pc_modes.shape[1])
        b_prev = np.ones(self.asm.pc_modes.shape[1])
        i = 0
        while (np.mean(np.abs(b-b_prev)) >= 1e-14):
            i += 1
            # 2. Generate the model point positions using x = X + Pb
            x = Landmarks(X.as_vector() + np.dot(self.asm.pc_modes, b))

            # 3. Find the pose parameters (Xt, Yt, s, theta) which best align the
            # model points x to the current found points Y
            is_upper = True if self.incisor_nr < 5 else False
            t, s, theta = align_params(x.get_crown(is_upper), Y.get_crown(is_upper))

            # 4. Project Y into the model co-ordinate frame by inverting the
            # transformation T
            y = Y.invT(t, s, theta)

            # 5. Project y into the tangent plane to X by scaling:
            # y' = y / (y*X).
            yacc = Landmarks(y.as_vector() / np.dot(y.as_vector(), X.as_vector().T))

            # 6. Update the model parameters to match to y': b = PT(y' - X)
            b_prev = b
            b = np.dot(self.asm.pc_modes.T, (yacc.as_vector()-X.as_vector()))

            # 7. If not converged, return to step 2

        return b, t, s, theta
