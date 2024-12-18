import numpy as np
from martingale.processes.browngen import brown_gen

params = {'scale': np.float64(52313.72781558013), 'rho': np.float64(0.6223246580908077), 'jump_size': np.float64(1.02821414252523), 'kappa': np.float64(0.7722222703417326), 'gamma': np.float64(0.06361053797681508), 'tau': np.float64(0.4253168784454925), 'nu': np.float64(0.13874027243797565), 'noise_scale': np.float64(2.049496092375214), 'noise_speed': np.float64(0.09453395448770051)}

if __name__=='__main__':
    brown_gen(n=5000, params=params)