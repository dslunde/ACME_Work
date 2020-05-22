# plotting_intro.py
"""Introductory Labs: Intro to Matplotlib.
Darren Lund
Vol1A (344)
9/13/16
"""

import numpy as np
from matplotlib import pyplot as plt

def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Inputs:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    A = np.random.randn(n,n)
    B = np.mean(A, axis=1)
    return np.var(B)

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    n = range(100,1100,100)
    y = []
    for i in xrange(len(n)) :
        y.append(var_of_means(n[i]))
    plt.plot(n,y)
    plt.show()


def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    x = np.linspace(-2*np.pi,2*np.pi,100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.arctan(x)
    plt.plot(x,y1,'b')
    plt.plot(x,y2,'r')
    plt.plot(x,y3,'g')
    plt.show()


def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Change the range of the y-axis to [-6,6].
    """
    x1 = np.linspace(-2,1,25)
    x1 = x1[:-1]
    x2 = np.linspace(1,6,43)
    x2 = x2[1:]
    y1 = 1 / (x1-1)
    y2 = 1 / (x2-1)
    plt.plot(x1,y1,'m:', lw=10, markersize=50)
    plt.plot(x2,y2,'m:', lw=10, markersize=50)
    plt.ylim(-6,6)
    plt.show()


def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    x = np.linspace(0,2*np.pi,50)
    y1 = np.sin(x)
    y2 = np.sin(2*x)
    y3 = 2*np.sin(x)
    y4 = 2*np.sin(2*x)

    plt.subplot(221)
    plt.axis([0,2*np.pi,-2,2])
    plt.title("y = sin(x)",fontsize=14)
    plt.plot(x,y1,'g-')

    plt.subplot(222)
    plt.axis([0,2*np.pi,-2,2])
    plt.title("y = sin(2x)",fontsize=14)
    plt.plot(x,y2,'r--')

    plt.subplot(223)
    plt.axis([0,2*np.pi,-2,2])
    plt.title("y = 2sin(x)",fontsize=14)
    plt.plot(x,y3,'b--')

    plt.subplot(224)
    plt.axis([0,2*np.pi,-2,2])
    plt.title("y = 2sin(2x)",fontsize=14)
    plt.plot(x,y4,'m:')

    plt.suptitle("The 2s of sine",fontsize=18)
    plt.show()


def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    X = np.load("FARS.npy")

    plt.subplot(121)
    plt.plot(X[:,1],X[:,2],'k,')
    plt.axis("equal")
    plt.xlabel("Latitude of Crash")
    plt.ylabel("Longitude of Crash")

    plt.subplot(122)
    plt.hist(X[:,0], bins=24)
    plt.xlim(0,23)
    plt.xlabel("Hour of the Day")

    plt.suptitle("Fatality Analysis Report",fontsize=18)
    plt.show()


def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """
    x1 = np.linspace(-2*np.pi,0,100)
    x1 = x1[:-1]
    y1 = x1.copy()
    X1,Y1 = np.meshgrid(x1,y1)
    x2 = np.linspace(0,2*np.pi,100)
    x2 = x2[1:]
    y2 = x2.copy()
    X2,Y2 = np.meshgrid(x2,y2)
    Z1 = np.sin(X1) * np.sin(Y1) / (X1*Y1)
    Z2 = np.sin(X2) * np.sin(Y1) / (X2*Y1)
    Z3 = np.sin(X1) * np.sin(Y2) / (X1*Y2)
    Z4 = np.sin(X2) * np.sin(Y2) / (X2*Y2)
    plt.axis([-2*np.pi,2*np.pi,-2*np.pi,2*np.pi])

    plt.subplot(121)
    plt.pcolormesh(X1,Y1,Z1,cmap="viridis")
    plt.pcolormesh(X2,Y1,Z2,cmap="viridis")
    plt.pcolormesh(X1,Y2,Z3,cmap="viridis")
    plt.pcolormesh(X2,Y2,Z4,cmap="viridis")
    plt.colorbar()

    plt.subplot(122)
    plt.contour(X1,Y1,Z1,15,cmap="viridis")
    plt.contour(X2,Y1,Z2,15,cmap="viridis")
    plt.contour(X1,Y2,Z3,15,cmap="viridis")
    plt.contour(X2,Y2,Z4,15,cmap="viridis")
    plt.colorbar()

    plt.show()
