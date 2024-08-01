from matplotlib import pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 18
BIGGER_SIZE = 26
TITLE_SIZE = 36
FIGSIZE = (16, 8)
GRID = True


def set_pyplot_defaults(
    title_size=TITLE_SIZE,
    small_size=SMALL_SIZE,
    medium_size=MEDIUM_SIZE,
    bigger_size=BIGGER_SIZE,
    grid=GRID,
    figsize=FIGSIZE
):
    plt.rc("font", size=small_size)  # controls default text sizes
    plt.rc("axes", titlesize=title_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=bigger_size)  # fontsize of the x and y labels
    plt.rc("axes", grid=grid)  # is grid on
    plt.rc("xtick", labelsize=medium_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=medium_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=medium_size)  # legend fontsize
    plt.rc("figure", titlesize=title_size)  # fontsize of the figure title
    plt.rc("figure", figsize=figsize)  # size of the figure
