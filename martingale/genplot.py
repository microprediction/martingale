import matplotlib.pyplot as plt

def ergodic_plot(gen):
    # Generate data
    n = 200
    x_values = []
    y_values = []

    count = 0
    for _ in gen: # settle to ergodic avg
        count += 1
        if count>10000-n:
            break

    for obs in gen:
        x_values.append(obs['x'])
        y_values.append(obs['y'])

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, label='x', alpha=0.8)
    plt.plot(y_values, linestyle='None',  marker='o', label='y', alpha=0.8)
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
        ergodic_plot(brown_gen(n=10000))
