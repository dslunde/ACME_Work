# linear_transformations.py
"""Volume 1A: Linear Transformations.
Darren Lund
Vol 1 A
9/20/16
"""

import time
import numpy as np
from random import random
from matplotlib import pyplot as plt

# Check Problem 1 by plotting functions
def check1(A) :
    plt.axis([-1,1,-1,1])
    plt.subplot(231)
    plt.plot(A[0,:],A[1,:],'k,')
    plt.title("Original")

    a,b = random(),random()

    B = stretch(A,a,b)

    plt.subplot(232)
    plt.plot(B[0,:],B[1,:],'k,')
    plt.title("Stretch:")

    a,b = random(),random()

    B = shear(A,a,b)

    plt.subplot(233)
    plt.plot(B[0,:],B[1,:],'k,')
    plt.title("Shear:")

    a,b = random(),random()

    B = reflect(A,a,b)

    plt.subplot(234)
    plt.plot(B[0,:],B[1,:],'k,')
    plt.title("Reflect:")

    theta = 4*np.pi*random() - 2*np.pi

    B = rotate(A,theta)

    plt.subplot(235)
    plt.plot(B[0,:],B[1,:],'k,')
    plt.title("Rotate:")

    plt.show()

# Problem 1
def stretch(A, a, b):
    """Scale the points in 'A' by 'a' in the x direction and 'b' in the
    y direction.

    Inputs:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    T = np.array([[a,0],[0,b]])
    return np.dot(T,A)
    #raise NotImplementedError("Problem 1 Incomplete")

def shear(A, a, b):
    """Slant the points in 'A' by 'a' in the x direction and 'b' in the
    y direction.

    Inputs:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    T = np.array([[1,a],[b,1]])
    return np.dot(T,A)
    #raise NotImplementedError("Problem 1 Incomplete")

def reflect(A, a, b):
    """Reflect the points in 'A' about the line that passes through the origin
    and the point (a,b).

    Inputs:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """
    T = (np.array([[a**2-b**2,2*a*b],[2*a*b,b**2-a**2]])) / float(a**2+b**2)
    return np.dot(T,A)
    #raise NotImplementedError("Problem 1 Incomplete")

def rotate(A, theta):
    """Rotate the points in 'A' about the origin by 'theta' radians.

    Inputs:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """
    T = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return np.dot(T,A)
    #raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def solar_system(T, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (10,0) and the initial
    position of the moon is (11,0).

    Parameters:
        T (int): The final time.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    t = np.linspace(0,T,150)
    p_earthx = [10]
    p_earthy = [0]
    p_moonx = [11]
    p_moony = [0]
    for i in xrange(1,len(t)):
        p_earth_new = rotate(np.vstack((p_earthx[0],p_earthy[0])),t[i]*omega_e)
        p_earthx.append(float(p_earth_new[0]))
        p_earthy.append(float(p_earth_new[1]))
        p_moon_new = rotate(np.vstack(((p_moonx[0]-p_earthx[0]),
            (p_moony[0]-p_earthy[0]))),t[i]*omega_m)
        p_moonx.append(float(p_moon_new[0]+p_earthx[i]))
        p_moony.append(float(p_moon_new[1]+p_earthy[i]))

    plt.plot(p_earthx,p_earthy,'b',label="Earth")
    plt.plot(p_moonx,p_moony,'g',label="Moon")
    plt.legend(loc="upper left")
    plt.show()
    #raise NotImplementedError("Problem 2 Incomplete")


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in xrange(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in xrange(n)] for i in xrange(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]


# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    n = np.linspace(1,250,8)
    v_time = []
    m_time = []
    start = 0
    end = 0
    for i in n :
        x = random_vector(int(i))
        A,B = random_matrix(int(i)), random_matrix(int(i))
        start = time.time()
        matrix_vector_product(A,x)
        end = time.time()
        v_time.append(end-start)
        start = time.time()
        matrix_matrix_product(A,B)
        end = time.time()
        m_time.append(end-start)

    plt.subplot(121)
    plt.axis([0,250,0,.025])
    plt.plot(n,v_time,'b.-')
    plt.xlabel("n")
    plt.ylabel("Seconds")
    plt.title("Matrix-Vector-Multiplicatoin")

    plt.subplot(122)
    plt.axis([0,250,0,5])
    plt.plot(n,m_time,'g.-')
    plt.xlabel("n")
    plt.ylabel("Seconds")
    plt.title("Matrix-Matrix-Multiplication")

    plt.show()
    #raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    n = np.linspace(1,2**8,8)
    v_time = []
    m_time = []
    npv_time = []
    npm_time = []
    start = 0
    end = 0
    for i in n :
        x = random_vector(int(i))
        A,B = random_matrix(int(i)), random_matrix(int(i))
        start = time.time()
        matrix_vector_product(A,x)
        end = time.time()
        v_time.append(end-start)
        start = time.time()
        matrix_matrix_product(A,B)
        end = time.time()
        m_time.append(end-start)
        start = time.time()
        np.dot(A,x)
        end = time.time()
        npv_time.append(end-start)
        start = time.time()
        np.dot(A,B)
        end = time.time()
        npm_time.append(end-start)

    plt.subplot(121)
    plt.plot(n,v_time,'b.-', label="Norm V")
    plt.plot(n,m_time,'g.-', label="Norm M")
    plt.plot(n,npv_time,'r.-', label="NP V")
    plt.plot(n,npm_time,'k.-', label="NP M")
    plt.xlabel("n")
    plt.ylabel("Seconds")
    plt.axis([0,250,0,5])
    plt.legend(loc="upper left")

    plt.subplot(122)
    plt.loglog(n,v_time,'b.-', basex=2, basey=2, label="Norm V")
    plt.loglog(n,m_time,'g.-', basex=2, basey=2, label="Norm M")
    plt.loglog(n,npv_time,'r.-', basex=2, basey=2, label="NP V")
    plt.loglog(n,npm_time,'k.-', basex=2, basey=2, label="NP M")
    plt.xlabel("n")
    plt.ylabel("Seconds")
    plt.axis([0,250,0,5])
    plt.legend(loc="upper left")

    plt.show()
    #raise NotImplementedError("Problem 4 Incomplete")

if __name__ == "__main__":
    #check1(np.load("horse.npy"))
    #T = float(2) * np.pi
    #solar_system(T,1,13)
    prob4()
