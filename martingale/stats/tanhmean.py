import math
import numpy as np
from martingale.stats.tanhscale import tanh_scale
from martingale.stats.fewmean import FEWMean
from martingale.stats.fewvar import FEWVar



class TanhMean:
    """

               Heuristic Filter w/ parameters

    Uses parameters:

      - fading_factor
      - outlier_fading_factor
      - alpha

    """

    def __init__(self,
                 mean_fading_factor=0.1,
                 var_fading_factor=0.01,
                 outlier_fading_factor=0.3,
                 alpha=0.2):

        # Params
        self.mean_fading_factor = mean_fading_factor
        self.var_fading_factor = var_fading_factor
        self.outlier_fading_factor = outlier_fading_factor
        self.alpha = alpha

        # State
        self._mean = FEWVar(fading_factor=self.mean_fading_factor)
        self._var = FEWVar(fading_factor=self.var_fading_factor)
        self._outlier_ewa = FEWMean(fading_factor=self.outlier_fading_factor)
        for _ in range(10):
            self._outlier_ewa.update(0)
        self.n = 0

    def get_mean(self):
        return self._mean.get_mean()

    def get(self):
        return self.get_mean()

    def get_var(self):
        return self._var.get_var()

    def update(self, x):
        self.n += 1
        x_mean = self._mean.get_mean()
        x_var = self._var.get_var()

        if x_var is None or x_var == 0:
            x_synthetic = x
        else:
            x_std = math.sqrt(x_var)
            z_score = (x - x_mean) / x_std
            outlier_ternary_indicator = int(abs(z_score) > 0.5) * np.sign(z_score)
            self._outlier_ewa.update(outlier_ternary_indicator)
            outlier_mean = self._outlier_ewa.get_mean()
            scale = tanh_scale(outlier_mean * np.sign(z_score), alpha=self.alpha)
            z_ratio = math.tanh(abs(z_score) / scale) / math.tanh(1 / scale)
            x_synthetic = x_mean + z_ratio * (x - x_mean)

            # Heuristic catch-up
            if outlier_mean*np.sign(z_score)>0.25:
                self._mean.update(x_synthetic)
            if outlier_mean*np.sign(z_score)>0.4:
                self._mean.update(x_synthetic)
            if outlier_mean * np.sign(z_score)>0.6:
                self._mean.update(x_synthetic)

        self._mean.update(x_synthetic)
        self._var.update(x_synthetic-x_mean)


    def get_params(self):
        """
        Return current parameters of the estimator.
        """
        return {
            "fading_factor": self.mean_fading_factor,
            "outlier_fading_factor": self.outlier_fading_factor,
            "alpha": self.alpha
        }




if __name__ == '__main__':
    from pprint import pprint
    tm = TanhMean(mean_fading_factor=0.05)
