from utility.conf import plt
from utility.conf import Axes3D
from utility.conf import np


def hist3d(y_label, x_label, z_value, \
           y_label_name="y", x_label_name="x", z_label_name="z", \
           output_filepath = None):

    """
    plot 3D hist.

    notes:
    y -> row -> data.shape[0]
    x -> column -> data->shape[1]

    Input
    -----
    y_label: {numpy array}, shape {n,}. x ticklabels
    x_label: {numpy array}, shape {m,}. y ticklabels
    z_value: {numpy array}, shape {n, m}. z values

    y_label_name: default = "y"
    x_label_name: default = "x"
    z_label_name: default = "z"
    output_filepath: {str}

    Output
    ------
    None

    """

    column_names = x_label.copy()
    row_names = y_label.copy()

    data = z_value.copy()

    fig = plt.figure(figsize=(8, 5))
    ax = Axes3D(fig)

    lx = len(data[0])  # Work out matrix dimensions
    ly = len(data[:, 0])
    xpos = np.arange(0, lx, 1)  # Set up a mesh of positions
    ypos = np.arange(0, ly, 1)
    xpos, ypos = np.meshgrid(xpos + 0.25, ypos + 0.25)

    xpos = xpos.flatten()  # Convert positions to 1D array
    ypos = ypos.flatten()
    zpos = np.zeros(lx * ly)

    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = data.flatten()

    bar_colors = ['red', 'green', 'blue', 'aqua',
                  'burlywood', 'cadetblue', 'chocolate', 'cornflowerblue',
                  'crimson', 'darkcyan', 'darkgoldenrod', 'darkgreen',
                  'purple', 'darkred', 'darkslateblue', 'darkviolet']

    colors = []

    for i in range(len(y_label)):
        for j in range(len(x_label)):
            colors.append(bar_colors[i])

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.6)

    # sh()
    ax.w_xaxis.set_ticklabels(x_label)
    ax.w_yaxis.set_ticklabels(y_label)
    ax.set_xlabel(x_label_name)
    ax.set_ylabel(y_label_name)
    ax.set_zlabel(z_label_name)

    # ax.w_xaxis.set_ticklabels(column_names)
    # ax.w_yaxis.set_ticklabels(row_names)

    # start, stop, step
    ticksx = np.arange(0.5, lx, 1)
    plt.xticks(ticksx, x_label)

    ticksy = np.arange(0.6, ly, 1)
    plt.yticks(ticksy, y_label)

    if output_filepath != None:
        plt.savefig(output_filepath)
        print("save to ", output_filepath)
    else:
        plt.show()
    plt.close()


def plot_array_like(array_1d, xlabel_name="number feature", ylabel_name="accuracy"):
    """
    plot line according to array_1d

    Input
    -----
    array_1d: {numpy array}, shape {n,}
    xlabel_name: {str}
    ylabel_name: {str}

    Output
    ------
    None

    """
    figsize = (8, 5)
    fig = plt.figure(figsize=figsize)

    plt.plot(range(len(array_1d)), array_1d)

    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)

    plt.xlim(0, len(array_1d))

    plt.show()