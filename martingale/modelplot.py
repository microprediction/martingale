import matplotlib.pyplot as plt
import numpy as np


def model_plot(gen, cls):
    # Number of total steps and burn-in period
    n = 200
    burn_in = 10000 - n

    # Advance the generator to settle to ergodic average (burn-in)
    count = 0
    for _ in gen:
        count += 1
        if count >= burn_in:
            break

    # Initialize the model
    model = cls()

    # Storage for values
    x_values = []
    y_values = []
    model_means = []
    model_vars = []

    # Now run the model on the next n steps
    for obs in gen:
        # Update the model with the new observation
        model.update(x=obs['x'])
        model_mean = model.get_mean()
        model_var = model.get_var()

        # Append current results
        x_values.append(obs['x'])
        y_values.append(obs['y'])
        model_means.append(model_mean)
        model_vars.append(model_var)

        if len(x_values) >= n:
            break

    # Convert model_vars to standard deviation
    model_stds = np.sqrt(model_vars)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, label='x', alpha=0.8)
    plt.plot(y_values, linestyle='None', marker='o', label='y', alpha=0.8)
    plt.plot(model_means, label='Model Mean', color='red', linewidth=2)

    # Fill between mean ± std
    timesteps = np.arange(len(model_means))
    plt.fill_between(timesteps,
                     np.array(model_means) - model_stds,
                     np.array(model_means) + model_stds,
                     color='red', alpha=0.2, label='±1 Std Dev')

    plt.title('Simulated Brownian Process with Model Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from martingale.processes.browngen import brown_gen
    from martingale.benchmarks.brownbenchmark import BrownBenchmark

    while True:
        model_plot(brown_gen(n=10000), cls=BrownBenchmark)
