import numpy as np
import matplotlib.pyplot as plt


def tanh_scale(x, alpha=2.0):
    """

        Intended to relate the outlier fraction to scale used for synthetic observations

    :param x:
    :param alpha:
    :return:
    """
    return 2*alpha ** (x / alpha)


if __name__=='__main__':

    # Test the function at key points
    print("f(-0.2) =", tanh_scale(-0.2))  # Should be about 0.5 for alpha=2
    print("f(0) =", tanh_scale(0.0))      # Should be 1
    print("f(0.2) =", tanh_scale(0.2))    # Should be about 2

    # Plot the function over [-1,1]
    x_values = np.linspace(-1, 1, 200)
    y_values = tanh_scale(x_values, alpha=2.0)

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, label=r"$f(x)=2^{x/0.2}$")
    plt.scatter([-0.2, 0, 0.2], [tanh_scale(-0.2), tanh_scale(0), tanh_scale(0.2)], color='red', zorder=5)
    plt.title("Example Function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()