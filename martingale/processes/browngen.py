from scipy.stats import beta
import numpy as np


def sample_correlation_rho(alpha_param=15, beta_param=5):
    # Sample correlation coefficient from Beta distribution: X ~ Beta(alpha_param, beta_param)
    # Transform X in [0,1] to rho in [-1,1]
    X = beta.rvs(alpha_param, beta_param)
    correlation_rho = 2 * X - 1
    return correlation_rho

def sample_mean_reversion_speed(low=0.001, high=1.0, alpha_param=5, beta_param=2):
    # mean_reversion_speed: kappa parameter for volatility process v_t
    X = beta.rvs(alpha_param, beta_param)
    return low + (high - low) * X

def sample_jump_intensity(low=0.001, high=0.1, alpha_param=5, beta_param=2):
    # jump_intensity: gamma parameter for jump process
    X = beta.rvs(alpha_param, beta_param)
    return low + (high - low) * X

def sample_noise_mean_reversion(low=0.001, high=0.5, alpha_param=5, beta_param=2):
    # noise_mean_reversion: tau parameter for mean reversion in noise scale s_t
    X = beta.rvs(alpha_param, beta_param)
    return low + (high - low) * X

def sample_epsilon_correlation(low=0.001, high=0.2, alpha_param=5, beta_param=2):
    # epsilon_correlation: nu parameter for serial correlation in observation noise
    X = beta.rvs(alpha_param, beta_param)
    return low + (high - low) * X

def sample_relative_noise_scale(low=-2, high=0, alpha_param=15, beta_param=2):
    # relative_noise_scale: scale of the observation noise, log-transformed
    X = beta.rvs(alpha_param, beta_param)
    return np.exp(low + (high - low) * X)

def sample_noise_evolution_rate(low=-3, high=-1, alpha_param=15, beta_param=2):
    # noise_evolution_rate: relative speed at which noise scale evolves
    X = beta.rvs(alpha_param, beta_param)
    return np.exp(low + (high - low) * X)

def brown_parameters():
    """
    Generate model parameters for the simulation:
    - time_step: increment in time
    - volatility_scale: scale factor for volatility
    - correlation_rho: correlation between Brownian increments
    - relative_jump_size: relative jump size (lognormal)
    - mean_reversion_speed: reversion speed (kappa) of volatility process
    - jump_intensity: intensity (gamma) of jump process
    - noise_mean_reversion: mean reversion (tau) for noise scale
    - epsilon_correlation: serial correlation (nu) in observation noise
    - relative_noise_scale: baseline scale of observation noise
    - noise_evolution_rate: rate at which noise scale evolves

    Returns:
        dict of parameters
    """
    time_step = 0.1 * np.exp(np.random.randn())
    volatility_scale = np.exp(10 * np.random.randn())
    correlation_rho = sample_correlation_rho()
    relative_jump_size = np.exp(0.1 * np.random.normal(loc=0, scale=1.0))
    mean_reversion_speed = sample_mean_reversion_speed()
    jump_intensity = sample_jump_intensity()
    noise_mean_reversion = sample_noise_mean_reversion()
    epsilon_correlation = sample_epsilon_correlation()
    relative_noise_scale = sample_relative_noise_scale()
    noise_evolution_rate = sample_noise_evolution_rate()

    return {
        'time_step': time_step,
        'volatility_scale': volatility_scale,
        'correlation_rho': correlation_rho,
        'relative_jump_size': relative_jump_size,
        'mean_reversion_speed': mean_reversion_speed,
        'jump_intensity': jump_intensity,
        'noise_mean_reversion': noise_mean_reversion,
        'epsilon_correlation': epsilon_correlation,
        'relative_noise_scale': relative_noise_scale,
        'noise_evolution_rate': noise_evolution_rate
    }

def generate_increments(n, time_step):
    """
    Generate Brownian increments for simulation:
    - brownian_increment_x: increments driving x_t
    - brownian_increment_v: increments driving v_t
    - brownian_increment_s: increments driving s_t

    Each ~ N(0, time_step).

    Args:
        n (int): number of time steps
        time_step (float): time increment

    Returns:
        tuple of increments (brownian_increment_x, brownian_increment_v, brownian_increment_s)
    """
    brownian_increment_x = np.random.randn(n) * np.sqrt(time_step)
    brownian_increment_v = np.random.randn(n) * np.sqrt(time_step)
    brownian_increment_s = np.random.randn(n) * np.sqrt(time_step)
    return brownian_increment_x, brownian_increment_v, brownian_increment_s

def generate_epsilon(n, epsilon_correlation):
    """
    Generate epsilon_t with serial correlation epsilon_correlation (nu):
    epsilon_t = nu * epsilon_{t-1} + sqrt(1 - nu^2) * z_t, where z_t ~ N(0,1)

    Args:
        n (int): number of steps
        epsilon_correlation (float): correlation parameter in [-1,1]

    Returns:
        np.array of shape (n,) with epsilon_t values
    """
    eps = np.zeros(n)
    z = np.random.randn(n)  # i.i.d. N(0,1)

    if abs(epsilon_correlation) < 1:
        sigma = np.sqrt(1 - epsilon_correlation**2)
    else:
        # If epsilon_correlation = ±1, degenerate case: no noise component
        sigma = 0.0

    for t in range(1, n):
        eps[t] = epsilon_correlation * eps[t - 1] + sigma * z[t]

    return eps

def generate_poisson_increments(n, time_step, jump_intensity):
    """
    Generate Poisson increments for jump process with intensity gamma (jump_intensity):
    dN_t ~ Poisson(jump_intensity * time_step)

    Args:
        n (int): number of time steps
        time_step (float): time increment
        jump_intensity (float): jump process intensity

    Returns:
        np.array: Poisson increments
    """
    return np.random.poisson(jump_intensity * time_step, size=n)

def simulate_processes(n, params):
    """
    Simulate the stochastic processes step-by-step.

    Model:
    - log_price: x_t
    - volatility: v_t
    - noise_state: s_t

    dJ_t = dN_t - gamma * dt (compensated jump measure)
    dx_t = volatility_scale * exp(v_t) * (dW_t + rho * dZ_t)
    dv_t = -kappa * v_t * dt + dZ_t + relative_jump_size * (dN_t - gamma * dt)
    ds_t = -tau * s_t * dt + noise_evolution_rate * dX_t

    Observed price:
    y_t = x_t + volatility_scale * relative_noise_scale * exp(s_t) * epsilon_t

    Args:
        n (int): number of steps
        params (dict): model parameters

    Yields:
        dict with keys {'x', 'y'} at each step
    """
    time_step = params['time_step']
    volatility_scale = params['volatility_scale']
    correlation_rho = params['correlation_rho']
    relative_jump_size = params['relative_jump_size']
    mean_reversion_speed = params['mean_reversion_speed']
    jump_intensity = params['jump_intensity']
    noise_mean_reversion = params['noise_mean_reversion']
    epsilon_correlation = params['epsilon_correlation']
    relative_noise_scale = params['relative_noise_scale']
    noise_evolution_rate = params['noise_evolution_rate']

    # Generate increments
    brownian_increment_x, brownian_increment_v, brownian_increment_s = generate_increments(n, time_step)
    poisson_jump_increment = generate_poisson_increments(n, time_step, jump_intensity)
    eps = generate_epsilon(n, epsilon_correlation)

    # Initialize states
    # States start at zero for simplicity
    log_price = 0.0
    volatility = 0.0
    noise_state = 0.0

    for t in range(n):
        # Compensated jump increment
        dJ = poisson_jump_increment[t] - jump_intensity * time_step

        # Volatility update
        dv = -mean_reversion_speed * volatility * time_step + brownian_increment_v[t] + relative_jump_size * dJ
        volatility_new = volatility + dv

        # Price update
        dx = (volatility_scale * np.exp(volatility) *
              (brownian_increment_x[t] + correlation_rho * brownian_increment_v[t]))
        log_price_new = log_price + dx

        # Noise scale update
        ds = -noise_mean_reversion * noise_state * time_step + noise_evolution_rate * brownian_increment_s[t]
        noise_state_new = noise_state + ds

        # Observed price
        observed_price = log_price_new + volatility_scale * relative_noise_scale * np.exp(noise_state_new) * eps[t]

        # Yield result for this step
        yield {'x': log_price_new, 'y': observed_price}

        # Move to next step
        log_price = log_price_new
        volatility = volatility_new
        noise_state = noise_state_new

def brown_gen(n=1000, params=None):
    """
    A generator function that yields one dict at a time with keys 'x' and 'y'.

    Args:
        n (int): number of steps
        params (dict, optional): If None, parameters are generated anew.

    Yields:
        dict: {'x': ..., 'y': ...} at each time step
    """
    if params is None:
        params = brown_parameters()
    print("Parameters:", params)

    for record in simulate_processes(n, params):
        yield record

if __name__ == '__main__':
    for obs in brown_gen(n=10):
        print(obs)
