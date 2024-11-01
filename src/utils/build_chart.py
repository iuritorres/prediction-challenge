from matplotlib import pyplot as plt, use
use('TkAgg')


def build_chart(X, y, y_pred, color_by, label, x_label, y_label, title):
    plt.scatter(X, y, c=color_by, cmap='viridis', label=label)
    plt.plot(X, y_pred, color='red', label='Linear Regression')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend()
    plt.show()
