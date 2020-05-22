# numpy_intro.py
"""Introductory Labs: Intro to NumPy.
Darren Lund
Volume 1
9/3/16
"""

import numpy as np

def prob1():
    """Define the matrices A and B as arrays. Return the matrix product AB."""
    A = np.array([[3,-1,4],[1,5,-9]])
    B = np.array([[2,6,-5,3],[5,-8,9,7],[9,-3,-2,-3]])
    return np.dot(A,B)

def prob2():
    """Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A."""
    A = np.array([[3,1,4],[1,5,9],[-5,3,1]])
    return (-np.dot(np.dot(A,A),A)+9*np.dot(A,A)-15*A)

def prob3():
    """Define the matrices A and B as arrays. Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    A = np.triu(np.ones((7,7)))
    B = np.tril(-np.ones((7,7))) + np.triu(np.full_like(A,5)) - 5*np.eye(7)
    C = np.dot(np.dot(A,B),A)
    return C.astype(np.int64)

def prob4(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    A1 = np.copy(A)
    A1[A1 < 0] = 0
    return A1

def prob5():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the identity matrix of appropriate size and each 0 is a matrix
    of all zeros, also of appropriate sizes.
    """
    A = np.array([[0,2,4],[1,3,5]],dtype=np.int64)
    C = -2*np.eye(3,dtype=np.int64)
    B = np.tril(np.full_like(C,3,dtype=np.int64))
    R1 = np.hstack((np.full((3,3),0),A.T,np.eye(3)))
    R2 = np.hstack((A,np.full((2,2),0),np.full_like(A,0)))
    R3 = np.hstack((B,np.full_like(A.T,0),C))
    return np.vstack((R1,R2,R3))

def prob6(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    A = A.astype(np.float64)
    B = A.sum(axis=1)
    B = B.reshape((-1,1))
    return np.divide(A,B)

def prob7():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    grid = np.load("grid.npy")
    A = np.max(grid[:,:-3]*grid[:,1:-2]*grid[:,2:-1]*grid[:,3:])
    B = np.max(grid[:-3,:]*grid[1:-2,:]*grid[2:-1,:]*grid[3:,:])
    C = np.max(grid[:-3,:-3]*grid[1:-2,1:-2]*grid[2:-1,2:-1]*grid[3:,3:])
    D = np.max(grid[3:,:-3]*grid[2:-1,1:-2]*grid[1:-2,2:-1]*grid[:-3,3:])
    return max(A,B,C,D)

if __name__ == "__main__" :
    print prob1()
    #print prob2()
    #print prob3()
    #print prob4(prob1())
    #print prob5()
    #A = np.array([[1,1,0,1],[0,1,0,1],[1,1,1,1],[1,0,0,0],[2,1,4,2]])
    #print prob6(A)
    #print prob7()
    # D
