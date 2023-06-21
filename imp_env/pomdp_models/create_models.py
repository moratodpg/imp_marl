""" Class for the transition model of an IMP problem. """
import numpy as np
import scipy.stats as stats
import math


class CreateModels:

    def __init__(self, config=None):
        self.T = config["T"]
        self.ncycles = config["ncycles"]
        self.d0_mean = config["d0_mean"]
        self.dcrit = config["dcrit"]
        self.S_mean = config["S_mean"]
        self.S_std = config["S_std"]
        self.lnC_mean = config["lnC_mean"]
        self.lnC_std = config["lnC_std"]
        self.m = config["m"]

    def crack_growth(self, nsamples):
        self.nsamples = int(nsamples)
        d0 = np.random.exponential(self.d0_mean, self.nsamples)
        C = np.random.lognormal(self.lnC_mean, self.lnC_std, self.nsamples)
        S = np.random.normal(self.S_mean, self.S_std, self.nsamples)

        self.dd = np.zeros((self.T + 1, self.nsamples))
        self.dd[0, :] = d0

        for t in range(self.T):
            dt = (((2 - self.m) / 2) * C * (
                        math.pi ** 0.5 * S) ** self.m * self.ncycles + d0 ** (
                              (2 - self.m) / 2)) ** (2 / (2 - self.m))
            dt[(dt < d0) | (dt > self.dcrit)] = self.dcrit + 0.1
            self.dd[t + 1, :] = dt
            d0 = dt
        return

    def transition_model(self, n_bins=30):
        self.n_bins = n_bins
        self.d_interv = 1e-100
        interv_start = math.log(1e-4)
        interv_step = (math.log(self.dcrit) - math.log(1e-4)) / (n_bins - 2)
        interv_end = math.log(self.dcrit)
        interv_transf = np.exp(
            np.arange(interv_start, interv_end + interv_step, interv_step))
        self.d_interv = np.append(self.d_interv, interv_transf)
        self.d_interv = np.append(self.d_interv, 1e20)

        det_rates = self.T + 1
        nsamples = self.dd.shape[-1]

        H, _ = np.histogram(self.dd[0, :], self.d_interv)
        self.b0 = H / nsamples

        self.T0 = np.zeros((det_rates, n_bins, n_bins))
        for i in range(det_rates - 1):
            D = self.dd[i, :]  # Samples a at det. rate i
            D_ = self.dd[i + 1, :]  # Samples a at det. rate i+1
            for j in range(n_bins):
                countd = (D > self.d_interv[j]) & (D < self.d_interv[j + 1])
                Dnext = D_[countd]
                if countd.sum() < 1:
                    self.T0[i, j, j] = 1
                else:
                    H, _ = np.histogram(Dnext, self.d_interv)
                    self.T0[i, j, :] = H / countd.sum()
        self.T0[-1,] = self.T0[-2,]

        self.Tr = np.zeros((det_rates, n_bins, n_bins))
        self.Tr = np.tile(self.b0, (det_rates, n_bins, 1))

        return self.d_interv

    def inspection_model(self, pod_insp=8):

        dobs = np.zeros((self.n_bins, 2))
        d_ref = (self.d_interv[0:-1] + self.d_interv[1:]) / 2
        d_ref[-1] = 21
        pod = stats.expon.cdf(d_ref, 0, pod_insp)
        dobs[:, 1] = pod
        dobs[:, 0] = 1 - dobs[:, 1]
        self.O = dobs
        return self.O