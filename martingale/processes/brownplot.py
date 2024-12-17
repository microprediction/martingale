import matplotlib.pyplot as plt

def noise_plot(gen):
    # Generate data
    n = 100
    dt = 0.1
    x_values = []
    y_values = []

    for obs in brown_gen(n=n, dt=dt):
        x_values.append(obs['x'])
        y_values.append(obs['y'])

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, label='x', alpha=0.8)
    plt.plot(y_values, '.', label='y', alpha=0.8)
    plt.title('Simulated Brownian Process with Noise and Jumps')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    while True:
        from martingale.processes.browngen import brown_gen
        noise_plot(brown_gen)
