from scipy.stats import beta
import numpy as np
import math


def sample_rho(alpha_param=15, beta_param=5):
    # Sample from Beta(alpha,beta_param)
    X = beta.rvs(alpha_param, beta_param)
    rho = 2 * X - 1
    return rho

def sample_kappa_beta(low=0.001, high=1.0, alpha_param=5, beta_param=2):
    # kappa: mean reversion for vol process v_t
    X = beta.rvs(alpha_param, beta_param)
    return low + (high - low)*X

def sample_gamma_beta(low=0.001, high=0.1, alpha_param=5, beta_param=2):
    # gamma: intensity of jump arrivals, choose something moderate
    X = beta.rvs(alpha_param, beta_param)
    return low + (high - low)*X


def sample_tau_beta(low=0.001, high=0.5, alpha_param=5, beta_param=2):
    # Mean reversion for noise scale
    X = beta.rvs(alpha_param, beta_param)
    return low + (high - low)*X


def sample_nu_beta(low=0.001, high=0.2, alpha_param=5, beta_param=2):
    # serial correlation for epsilon_t in observations
    X = beta.rvs(alpha_param, beta_param)
    return low + (high - low)*X


def sample_noise_scale_beta(low=-2, high=0, alpha_param=15, beta_param=2):
    # relative scale of noise
    X = beta.rvs(alpha_param, beta_param)
    return np.exp(low + (high - low)*X)

def sample_noise_speed_beta(low=-2, high=0, alpha_param=15, beta_param=2):
    # relative speed at which noise scale changes
    X = beta.rvs(alpha_param, beta_param)
    return np.exp(low + (high - low)*X)



def brown_parameters():
    """
    Generate model parameters at the outset:
    - scale: scale parameter with a diffuse lognormal distribution
    - rho: correlation coefficient in [-1,1], centered around 0
    - eta: relative jump size (lognormal)
    - kappa: mean reversion parameter for v_t
    - gamma: intensity of the jump process
    - tau: mean reversion parameter for s_t
    - nu: serial correlation for epsilon_t in observations

    Returns:
        dict of parameters
    """

    scale = np.exp(10*np.random.randn())  # Very diffuse

    rho = sample_rho()

    # jump_size: relative jump size, lognormal
    jump_size = np.exp(0.1*np.random.normal(loc=0, scale=1.0))

    # kappa: mean reversion for vol process v_t
    kappa = sample_kappa_beta()

    # gamma: intensity of jump arrivals, choose something moderate
    gamma = sample_gamma_beta()

    # tau: mean reversion for noise scale
    tau = sample_tau_beta()

    # nu: serial correlation for epsilon_t in observations
    nu = sample_nu_beta()
    
    noise_scale = sample_noise_scale_beta()
    
    noise_speed = sample_noise_speed_beta()

    return {
        'scale': scale,
        'rho': rho,
        'jump_size': jump_size,
        'kappa': kappa,
        'gamma': gamma,
        'tau': tau,
        'nu': nu,
        'noise_scale':noise_scale,
        'noise_speed':noise_speed
    }

def generate_increments(n, dt):
    """
    Generate the increments needed for the simulation:
    - dW_t, dZ_t, dX_t: independent Brownian increments ~ N(0, dt)
    - dN_t: Poisson increments for jump process with rate gamma

    We will generate standard normal increments first and scale by sqrt(dt) later.

    Args:
        n (int): number of time steps
        dt (float): time step size

    Returns:
        dict of increments arrays
    """
    # dW_t, dZ_t, dX_t ~ N(0, dt)
    dW = np.random.randn(n) * np.sqrt(dt)
    dZ = np.random.randn(n) * np.sqrt(dt)
    dX = np.random.randn(n) * np.sqrt(dt)
    return dW, dZ, dX

def generate_epsilon(n, nu):
    """
    Generate the epsilon_t series with serial correlation nu:
    epsilon_t = nu * epsilon_{t-1} + sqrt(1-nu^2)*z_t, z_t ~ N(0,1)

    Args:
        n (int): number of steps
        nu (float): correlation parameter in [-1,1]

    Returns:
        np.array of shape (n,) with epsilon_t values
    """
    eps = np.zeros(n)
    z = np.random.randn(n)  # i.i.d. N(0,1)
    if abs(nu) < 1:
        sigma = np.sqrt(1 - nu**2)
    else:
        # If nu = ±1, this is degenerate. We handle by no noise if nu = ±1.
        sigma = 0.0
    for t in range(1, n):
        eps[t] = nu * eps[ t -1] + sigma * z[t]
    return eps

def generate_poisson_increments(n, dt, gamma):
    """
    Generate Poisson increments dN_t for the jump process with intensity gamma.

    dN_t ~ Poisson(gamma * dt)
    """
    return np.random.poisson(gamma * dt, size=n)

def simulate_processes(n, dt, params):
    """
    Simulate the processes step-by-step and yield one dict at a time.

    Processes:

    - x_t, v_t, s_t states
    - dJ_t = dN_t - gamma*dt (compensated jump measure)
    - dx_t = scale * exp(v_t) * (dW_t + rho dZ_t)
    - dv_t = -kappa * v_t * dt + dZ_t + jump_size(dN_t - gamma dt)
    - ds_t = -tau * dt +  noise_speed * dX_t
    - y_t = x_t + scale * noise_scale * exp(s_t) * epsilon_t

    Initialize x_0, v_0, s_0 = 0.
    """
    scale = params['scale']
    rho = params['rho']
    jump_size = params['jump_size']
    kappa = params['kappa']
    gamma = params['gamma']
    tau = params['tau']
    nu = params['nu']
    noise_scale = params['noise_scale']
    noise_speed = params['noise_speed']
    

    # Generate increments
    dW, dZ, dX = generate_increments(n, dt)
    dN = generate_poisson_increments(n, dt, gamma)
    eps = generate_epsilon(n, nu)

    # Initialize states
    invariance_s_std = (noise_speed/tau)*scale*noise_scale
    invariance_x_std = invariance_s_std/kappa

    # x = np.exp(0.25*np.random.randn())*invariance_x_std
    x = 0.0
    v = 0.0
    s = 0.0 # np.exp(0.25*np.random.randn())*invariance_s_std

    for t in range(n):
        # Compute increments for this step
        dJ = dN[t] - gamma * dt  # compensated jump increment

        # Update v, the vol driving process
        dv = -kappa * v * dt + dZ[t] + jump_size * dJ
        v_new = v + dv

        # Update x
        dx = scale * np.exp(v) * (dW[t] + rho * dZ[t]) * math.sqrt(dt)
        x_new = x + dx

        # Update s ... the noise scale

        ds = -tau * s * dt + noise_speed * math.sqrt(dt)*dX[t]
        s_new = s + ds

        # Observation
        y = x_new + scale * noise_scale * np.exp(s_new) * eps[t]

        # Yield the record
        yield {'x': x_new, 'y': y}

        # Move to next step
        x = x_new
        v = v_new
        s = s_new

def brown_gen(n=1000, dt=0.01, params=None):
    """
    A generator function that yields one dict at a time with keys 'x' and 'y'.

    The process and parameters are described in the docstring above.
    """
    if params is None:
        params = brown_parameters()
    print(params)

    # Simulate and yield records
    for record in simulate_processes(n, dt, params):
        yield record


if __name__=='__main__':
   for obs in brown_gen(n=10, dt=0.1):
      print(obs)
