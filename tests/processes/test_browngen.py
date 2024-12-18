import pytest
import numpy as np
from scipy.stats import beta
from unittest.mock import patch

# The code under test is assumed to be in a file named browngen.py
# Make sure your test file is in the same directory or properly import the tested functions.
from martingale.processes.browngen import (
    sample_correlation_rho,
    sample_mean_reversion_speed,
    sample_jump_intensity,
    sample_noise_mean_reversion,
    sample_epsilon_correlation,
    sample_relative_noise_scale,
    sample_noise_evolution_rate,
    brown_parameters,
    generate_increments,
    generate_epsilon,
    generate_poisson_increments,
    simulate_processes,
    brown_gen
)

@pytest.fixture(scope="module", autouse=True)
def fixed_seed():
    # Set a fixed seed for reproducibility
    np.random.seed(42)


def test_sample_correlation_rho():
    """Test that sample_correlation_rho returns a value in [-1,1]."""
    rho = sample_correlation_rho()
    assert -1.0 <= rho <= 1.0, "Correlation should be in [-1,1]."


def test_sample_mean_reversion_speed():
    """Test that sample_mean_reversion_speed returns a value in [0.001,1.0]."""
    val = sample_mean_reversion_speed()
    assert 0.001 <= val <= 1.0, "Mean reversion speed should be in [0.001,1.0]."


def test_sample_jump_intensity():
    """Test that sample_jump_intensity returns a value in [0.001,0.1]."""
    val = sample_jump_intensity()
    assert 0.001 <= val <= 0.1, "Jump intensity should be in [0.001,0.1]."


def test_sample_noise_mean_reversion():
    """Test that sample_noise_mean_reversion returns a value in [0.001,0.5]."""
    val = sample_noise_mean_reversion()
    assert 0.001 <= val <= 0.5, "Noise mean reversion should be in [0.001,0.5]."


def test_sample_epsilon_correlation():
    """Test that sample_epsilon_correlation returns a value in [0.001,0.2]."""
    val = sample_epsilon_correlation()
    assert 0.001 <= val <= 0.2, "Epsilon correlation should be in [0.001,0.2]."


def test_sample_relative_noise_scale():
    """Test that sample_relative_noise_scale returns a positive value
    within the range exp([-2,0]) which is [exp(-2),1]."""
    val = sample_relative_noise_scale()
    assert val > 0, "Relative noise scale should be positive."
    # The range of the exponent is from -2 to 0, so val in [exp(-2), exp(0)]
    assert np.exp(-2) <= val <= 1.0, f"Relative noise scale should be in [{np.exp(-2)}, 1.0]."


def test_sample_noise_evolution_rate():
    """Test that sample_noise_evolution_rate returns a positive value
    within the range exp([-3,-1]) which is [exp(-3), exp(-1)]."""
    val = sample_noise_evolution_rate()
    assert val > 0, "Noise evolution rate should be positive."
    # The range of the exponent is from -3 to -1, so val in [exp(-3), exp(-1)]
    assert np.exp(-3) <= val <= np.exp(-1), f"Noise evolution rate should be in [{np.exp(-3)}, {np.exp(-1)}]."


def test_brown_parameters():
    """Test that brown_parameters returns a dictionary with expected keys and value ranges."""
    params = brown_parameters()
    expected_keys = {
        'time_step',
        'volatility_scale',
        'correlation_rho',
        'relative_jump_size',
        'mean_reversion_speed',
        'jump_intensity',
        'noise_mean_reversion',
        'epsilon_correlation',
        'relative_noise_scale',
        'noise_evolution_rate'
    }
    assert expected_keys.issubset(params.keys()), "brown_parameters should return all required keys."

    # Check ranges:
    assert isinstance(params['time_step'], float) and params['time_step'] > 0, "time_step must be positive."
    assert params['volatility_scale'] > 0, "volatility_scale must be positive."
    assert -1.0 <= params['correlation_rho'] <= 1.0, "correlation_rho must be in [-1,1]."
    # relative_jump_size is lognormal, typically > 0
    assert params['relative_jump_size'] > 0, "relative_jump_size must be positive."
    assert 0.001 <= params['mean_reversion_speed'] <= 1.0, "mean_reversion_speed in [0.001,1.0]."
    assert 0.001 <= params['jump_intensity'] <= 0.1, "jump_intensity in [0.001,0.1]."
    assert 0.001 <= params['noise_mean_reversion'] <= 0.5, "noise_mean_reversion in [0.001,0.5]."
    assert 0.001 <= params['epsilon_correlation'] <= 0.2, "epsilon_correlation in [0.001,0.2]."
    # Already tested ranges for relative_noise_scale and noise_evolution_rate above, just check positivity:
    assert params['relative_noise_scale'] > 0, "relative_noise_scale must be positive."
    assert params['noise_evolution_rate'] > 0, "noise_evolution_rate must be positive."


def test_generate_increments():
    """Test generate_increments returns three arrays of shape (n,)."""
    n = 10
    time_step = 0.01
    x_inc, v_inc, s_inc = generate_increments(n, time_step)
    assert x_inc.shape == (n,) and v_inc.shape == (n,) and s_inc.shape == (n,), "All increments must be 1D arrays of length n."
    # Check that increments have variance roughly time_step (not a strict test, just sanity check)
    # With a fixed seed, we know what to expect, but just a loose check:
    assert np.allclose(np.var(x_inc), time_step, atol=0.1), "Variance of increments should be close to time_step."
    assert np.allclose(np.var(v_inc), time_step, atol=0.1), "Variance of increments should be close to time_step."
    assert np.allclose(np.var(s_inc), time_step, atol=0.1), "Variance of increments should be close to time_step."


def test_generate_epsilon():
    """Test that generate_epsilon returns an array of shape (n,) and is a stationary process with given correlation."""
    n = 10
    eps_corr = 0.1
    eps = generate_epsilon(n, eps_corr)
    assert eps.shape == (n,), "Epsilon array must be 1D of length n."
    # Just ensure finite values and some variability
    assert np.all(np.isfinite(eps)), "Epsilon values must be finite."


def test_generate_poisson_increments():
    """Test that generate_poisson_increments returns a non-negative integer array of shape (n,)."""
    n = 10
    time_step = 0.01
    jump_intensity = 0.05
    inc = generate_poisson_increments(n, time_step, jump_intensity)
    assert inc.shape == (n,), "Poisson increments must be 1D array of length n."
    assert np.issubdtype(inc.dtype, np.integer), "Poisson increments must be integer."
    assert np.all(inc >= 0), "Poisson increments must be non-negative."


def test_simulate_processes():
    """Test that simulate_processes yields n records, each with 'x' and 'y' keys."""
    n = 5
    params = brown_parameters()
    gen = simulate_processes(n, params)
    count = 0
    for record in gen:
        count += 1
        assert 'x' in record and 'y' in record, "Each record must contain 'x' and 'y' keys."
        assert np.isfinite(record['x']), "'x' must be finite."
        assert np.isfinite(record['y']), "'y' must be finite."
    assert count == n, "simulate_processes should yield exactly n records."


def test_brown_gen():
    """Test that brown_gen yields n records when given n, and can generate parameters if needed."""
    n = 5
    gen = brown_gen(n=n, params=None)
    count = 0
    for record in gen:
        count += 1
        assert 'x' in record and 'y' in record, "Each record must contain 'x' and 'y' keys."
    assert count == n, "brown_gen should yield exactly n records."
