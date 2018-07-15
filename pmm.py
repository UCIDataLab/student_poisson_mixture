import numpy as np
from random import random, uniform, seed
from utils import logpoisson_vec_all
import matplotlib.pyplot as plt


class PoissonMixture():

    def __init__(self, K, max_iter=200, eps=0.0001,
                 gamma_prior=True, g_a=1.1, g_b=0.1, rseed=321, verbose=1):
        """

        Parameters
        ----------
        K: int
            Number of mixture components.
        max_iter: int
            Maximum number of EM iterations. The algorithm stops when either
            the number of iteration reaches `max_iter` or the change in log likelihood
            becomes less than `eps`.
        eps: float
            Tolerance with regards to the log likelihood of the model
            to declare convergence. The algorithm stops when either
            the number of iteration reaches `max_iter` or the change in log likelihood
            becomes less than `eps`.
        gamma_prior: bool
            Add gamma prior to the model. Default is True.
        g_a: float
            Hyperparameter for the gamma prior. Used if `gamma_prior` is True.
        g_b: float
            Hyperparameter for the gamma prior. Used if `gamma_prior` is True.
        rseed: int
            Random seed
        verbose: int
            The level of verbosity. Integer value in [0, 2]
        """
        self.K = K
        self.max_iter = max_iter
        self.eps = eps
        self.gamma_prior = gamma_prior
        self.g_a = g_a
        self.g_b = g_b
        self.rseed = rseed
        self.verbose = verbose

    def calculate_loglik(self, Y, lam, pi):
        """
        Calculates total log likelihood of the model.

        Parameters
        ----------
        Y: np.array of size (N, D)
            Number of counts per day for each student
        lam: np.array of size (K, D)
            Component means
        pi: np.array of size (K,)
            Mixture weights

        * N: number of students
          D: number of days in week
          K: number of mixture components

        Returns
        -------
        float
        """
        N = Y.shape[0]
        p_y_z = np.zeros((N, self.K))
        for k in range(self.K):
            p_y_z[:, k] = np.exp(logpoisson_vec_all(lam[k], Y, self.gamma_prior,
                                                    self.g_a, self.g_b))
        p_y = np.dot(p_y_z, pi)
        return np.sum(np.log(p_y))

    def _e_step(self, Y, lam, pi):
        """
        E-step of EM algorithm for Poisson mixture model.
        Calculates membership weights for each student.

        Parameters
        ----------
        Y: np.array of size (N, D)
            Number of counts per day for each student
        lam: np.array of size (K, D)
            Component means
        pi: np.array of size (K,)
            Mixture weights

        * N: number of students
          D: number of days in week
          K: number of mixture components

        Returns
        -------
        np.array of size (N, K)
            calculated membership weights
        """
        N = Y.shape[0]
        p_y_z = np.zeros((N, self.K))
        mw = np.zeros((N, self.K))
        for k in range(self.K):
            p_y_z[:, k] = np.exp(logpoisson_vec_all(lam[k], Y, self.gamma_prior,
                                                    self.g_a, self.g_b))
            mw[:, k] = p_y_z[:, k] * pi[k]
        denom = np.tile(np.sum(mw, axis=1), (self.K, 1)).T
        mw /= denom
        return mw

    def _m_step(self, Y, mw, fix_one_lambda_flat):
        """
        M-step of EM algorithm for Poisson mixture model.
        Does MLE or MAP estimation (depending on whether the model has Gamma prior or not)
        for parameters pi (mixture weights) and lambdas (component means).

        Parameters
        ----------
        Y: np.array of size (N, D)
            Number of counts per day for each student
        mw: np.array of size (N, K)
            Membership weight for each student
        fix_one_lambda_flat: bool
            If True, one of the mixture component will be forced to have a very
            flat lambda vector as its mean.
            Default is set to False.

        * N: number of students
          D: number of days in week
          K: number of mixture components

        Returns
        -------
        tuple[np.array, np.array]

        """
        N = Y.shape[0]
        D = Y.shape[1]

        Nk = np.sum(mw, axis=0)
        pi = Nk / float(N)
        if self.gamma_prior:
            lam = np.dot(mw.T, (Y + self.g_a - 1)) / ((1 + self.g_b) * Nk[:, None])
        else:
            lam = np.dot(mw.T, Y) / (Nk[:, None] + 1e-7)  # add a very small number to avoid division by 0
        if fix_one_lambda_flat:
            lam[0, 1:] = np.sum(lam[0, 1:]) / (D - 1)
        return lam, pi

    def fit(self, Y, fix_one_lambda_flat=False, return_obj=True):
        """
        Fit the Poisson mixture model with data matrix `Y`.

        Parameters
        ----------
        Y: np.array of size (N, D)
            Number of counts per day matrix for each student
            Each row is a student's daily activity count vector.
        fix_one_lambda_flat: bool
            If True, one of the mixture component will be forced to have a very
            flat lambda vector as its mean.
            Default is set to False.
        return_obj: bool
            If True, returns result as an object of class PoisMixResult.
            If False, returns a list of results.

        * N: number of students
          D: number of days in week
          K: number of mixture components

        Returns
        -------
        PoisMixResult
        or tuple[np.array, np.array, np.array, float] that corresponds to [lam, mw, pi, loglik]
        """
        def print_params(lam, pi, loglik):
            print("lam", lam)
            print("pi", pi)
            print("loglik", loglik)
            print("")

        Y = np.array(Y)
        N = Y.shape[0]
        K = self.K

        # mixture weights
        pi = np.ones(K) / float(K)
        # membership weights
        mw = np.zeros((N, K))
        # Initialize log likelihood
        loglik = 0.0

        # yi is a vector
        if (len(Y.shape) == 2) and (Y.shape[1] > 1):

            # Days
            D = Y.shape[1]

            # Initialize lambdas
            lam = np.zeros((K, D))
            seed(self.rseed)
            for k in range(K):
                lam[k] = np.ones((1, D)) * uniform(0.5, 3)

            for i in range(self.max_iter):
                prev_ll = loglik
                loglik = self.calculate_loglik(Y, lam, pi)

                if abs(prev_ll-loglik) < self.eps:
                    break
                if self.verbose > 0:
                    if i % 20 == 0:
                        print("----- number of iterations: " + str(i))
                        print("  loglik: %.3f\n" % loglik)
                if self.verbose > 2:
                    print_params(lam, pi, loglik)

                # Estep
                mw = self._e_step(Y, lam, pi)
                # Mstep
                lam, pi = self._m_step(Y, mw, fix_one_lambda_flat)

            # calculate final log lik
            loglik = self.calculate_loglik(Y, lam, pi)
            if self.verbose > 0:
                print("----- number of iterations: " + str(i))
                print("  loglik: %.3f\n" % loglik)

            if return_obj:
                return PoisMixResult(lam, mw=mw, pi=pi, loglik=loglik,
                                     gamma_prior=self.gamma_prior, g_a=self.g_a, g_b=self.g_b)
            else:
                return lam, mw, pi, loglik

        else:
            print("ERROR: Y should be a 2 dimensional array. Exiting..")
            return


class PoisMixResult():
    """
    A class that stores all the results from Poisson mixture model.

    Variables
    ---------
    assignments: np.array of size N
        List of mixture group indices that each student belongs to
        (when thresholded at membetship weight = 0.5)
    group_counts: dict[int:int]
        Dictionary that shows the number of students assigned to each group index.
        Key: group/component index, Value: Number of students

    * Most variables have the same format as in the class PoissonMixture. (lam, mw, pi, loglik)
    """

    def __init__(self, lam, mw=None, pi=None, loglik=0.0,
                 gamma_prior=False, g_a=2.0, g_b=2.0):
        """
        Parameters
        ----------
        lam: np.array with size (K, D)
            Mixture component means
        mw: np.array with size (N, K)
            membership weight of each student and each mixture component
        pi: np.array with size (K, )
            mixture weight for each component
        loglik: float
            log likelihood of the final model
        gamma_prior: bool
            True if the model has gamma prior
        g_a: float
            Hyperparameter of gamma prior a
        g_b: float
            Hyperparameter of gamma prior b
        """
        self.K = lam.shape[0]
        self.gamma_prior = gamma_prior
        self.g_a = g_a
        self.g_b = g_b

        self.lam = lam
        self.mw = mw
        self.pi = pi
        self.loglik = loglik

        if mw is not None:
            self.N = mw.shape[0]
            self.assignments = np.argmax(self.mw, axis=1)
            from collections import Counter
            self.group_counts = dict(Counter(self.assignments))

    def plot_lambdas(self, xticklabels=None, divide_by_week=False, colors=None,
                     fontsize=15, ticklab_size=15,
                     save_fig=False, file_prefix='./lambda'):
        """
        Plot lambdas (component means) into pdf files.

        Parameters
        ----------
        xticklabels
        divide_by_week
        colors
        save_fig
        file_prefix

        Returns
        -------

        """
        print('Plotting lambdas')
        if colors == None:
            colors = ['blue'] * self.K
        maxlam = int(np.max(self.lam) + 1.0)
        n_lam = self.lam.shape[1]
        for k in range(self.K):
            fig, ax = plt.subplots(figsize=(6, 4))
            if divide_by_week:
                ax.bar(range(n_lam), self.lam[k] / 5.0, align='center', color=colors[k], linewidth=0)
                ax.set_ylabel('AVG NUMBER OF TASKS\nON EACH DAY OF WEEK', fontsize=fontsize)
            else:
                ax.bar(range(n_lam), self.lam[k], align='center', color=colors[k], linewidth=0)
                ax.set_ylabel('TOTAL NUMBER OF TASKS\nON EACH DAY OF WEEK', fontsize=fontsize)
            ax.grid(alpha=0.2)
            ax.set_ylim(0, maxlam)
            ax.set_yticks(range(maxlam + 1))
            ax.set_xlim(-0.5, n_lam - 0.5)
            ax.set_xticks(range(n_lam))
            if xticklabels is None:
                if n_lam == 6:
                    ax.set_xticklabels(['SS', 'M', 'T', 'W', 'T', 'F'], fontsize=ticklab_size)
                elif n_lam == 7:
                    ax.set_xticklabels(['M', 'T', 'W', 'T', 'F', 'S', 'S'], fontsize=ticklab_size)
            else:
                ax.set_xticklabels(xticklabels)

            if save_fig:
                fname = '_'.join([file_prefix, str(self.K), str(k) + '.pdf'])
                fig.savefig(fname)
