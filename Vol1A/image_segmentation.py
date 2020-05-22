# image_segmentation.py
"""Volume 1A: Image Segmentation.
Darren Lund
Mathematical Analysis Lab
11/14/16
"""

from matplotlib import pyplot as plt
from scipy import linalg as la
from scipy.sparse import linalg as las
from scipy import sparse
import numpy as np
import math

def mach_fix(a) :
    if np.isclose([int(a)],[a]) :
        a = int(a)
    elif a > 0 :
        if np.isclose([int(a)+1],[a]) :
            a = int(a) + 1
    else :
        if np.isclose([int(a)-1],[a]) :
            a = int(a) - 1
    return a

# Problem 1: Implement this function.
def laplacian(A):
    '''
    Compute the Laplacian matrix of the adjacency matrix A.
    Inputs:
        A (array): adjacency matrix for undirected weighted graph,
             shape (n,n)
    Returns:
        L (array): Laplacian matrix of A

    '''
    #raise NotImplementedError("Problem 1 Incomplete")
    D = np.zeros_like(A)
    for i in xrange(A.shape[0]) :
        D[i][i] = sum(A[:,i])
    return D-A

# Problem 2: Implement this function.
def n_components(A,tol=1e-8):
    '''
    Compute the number of connected components in a graph
    and its algebraic connectivity, given its adjacency matrix.
    Inputs:
        A -- adjacency matrix for undirected weighted graph,
             shape (n,n)
        tol -- tolerance value
    Returns:
        n_components -- the number of connected components
        lambda -- the algebraic connectivity
    '''
    #raise NotImplementedError("Problem 2 Incomplete")
    L = laplacian(A)
    x = np.real(la.eigvals(L))
    multiplicity = 0
    for i in xrange(len(x)) :
        if abs(x[i]) < tol :
            x[i] = 0
            multiplicity += 1
    nonzero_x = [x[i] for i in xrange(len(x)) if x[i] != 0]
    if multiplicity > 1 :
        alg_mult = 0
    else :
        alg_mult = min(nonzero_x)
        alg_mult = mach_fix(alg_mult)
    return multiplicity,alg_mult


# Problem 3: Implement this function.
def adjacency(filename="dream.png", radius = 5.0, sigma_I = .15, sigma_d = 1.7):
    '''
    Compute the weighted adjacency matrix for
    the image given the radius. Do all computations with sparse matrices.
    Also, return an array giving the main diagonal of the degree matrix.

    Inputs:
        filename (string): filename of the image for which the adjacency matrix will be calculated
        radius (float): maximum distance where the weight isn't 0
        sigma_I (float): some constant to help define the weight
        sigma_d (float): some constant to help define the weight
    Returns:
        W (sparse array(csc)): the weighted adjacency matrix of img_brightness,
            in sparse form.
        D (array): 1D array representing the main diagonal of the degree matrix.
    '''
    #raise NotImplementedError("Problem 3 Incomplete")
    def W_val(i,j,d,sI,sd) :
        W[i,j] = np.exp(-(abs(flat[i]-flat[j])/sI**2)-d/sd**2)

    inplace = np.vectorize(W_val)

    P,pic = getImage(filename)
    m,n = pic.shape
    flat = pic.flatten()
    W = sparse.lil_matrix((m*n,m*n))
    for i in xrange(m*n) :
        r,d = getNeighbors(i,radius,m,n)
        inplace(i,r,d,sigma_I,sigma_d)
        #W[i,r] = np.exp(-abs(flat[i]-flat[r])/sigma_I**2-d/sigma_d**2)
    W1 = W.tocsc()
    D = np.array(W1.sum(axis=1).reshape(m*n))[0]
    return W1,D



# Problem 4: Implement this function.
def segment(filename="dream.png"):
    '''
    Compute and return the two segments of the image as described in the text.
    Compute L, the laplacian matrix. Then compute D^(-1/2)LD^(-1/2),and find
    the eigenvector corresponding to the second smallest eigenvalue.
    Use this eigenvector to calculate a mask that will be usedto extract
    the segments of the image.
    Inputs:
        filename (string): filename of the image to be segmented
    Returns:
        seg1 (array): an array the same size as img_brightness, but with 0's
                for each pixel not included in the positive
                segment (which corresponds to the positive
                entries of the computed eigenvector)
        seg2 (array): an array the same size as img_brightness, but with 0's
                for each pixel not included in the negative
                segment.
    '''
    #raise NotImplementedError("Problem 4 Incomplete")
    P,pic = getImage(filename)
    m,n = pic.shape
    W,D = adjacency(filename)
    D1 = D**(-1./2)
    """
    for i in xrange(len(D)) :
        D1[i] = D[i]**(-1./2)
    """
    D1 = sparse.spdiags(D1,0,m*n,m*n)
    D = sparse.spdiags(D,0,m*n,m*n)
    L = D-W
    A = D1.dot(L.dot(D1))
    eigv = las.eigs(A,2,which='SM')[1][:,1]
    mask = (eigv.reshape(m,n) > 0)
    return mask * pic, ~mask * pic

# Helper function used to convert the image into the correct format.
def getImage(filename='dream.png'):
    '''
    Reads an image and converts the image to a 2-D array of brightness
    values.

    Inputs:
        filename (str): filename of the image to be transformed.
    Returns:
        img_color (array): the image in array form
        img_brightness (array): the image array converted to an array of
            brightness values.
    '''
    img_color = plt.imread(filename)
    img_brightness = (img_color[:,:,0]+img_color[:,:,1]+img_color[:,:,2])/3.0
    return img_color,img_brightness

# Helper function for computing the adjacency matrix of an image
def getNeighbors(index, radius, height, width):
    '''
    Calculate the indices and distances of pixels within radius
    of the pixel at index, where the pixels are in a (height, width) shaped
    array. The returned indices are with respect to the flattened version of the
    array. This is a helper function for adjacency.

    Inputs:
        index (int): denotes the index in the flattened array of the pixel we are
                looking at
        radius (float): radius of the circular region centered at pixel (row, col)
        height, width (int,int): the height and width of the original image, in pixels
    Returns:
        indices (int): a flat array of indices of pixels that are within distance r
                   of the pixel at (row, col)
        distances (int): a flat array giving the respective distances from these
                     pixels to the center pixel.
    '''
    # Find appropriate row, column in unflattened image for flattened index
    row, col = index/width, index%width
    # Cast radius to an int (so we can use arange)
    r = int(radius)
    # Make a square grid of side length 2*r centered at index
    # (This is the sup-norm)
    x = np.arange(max(col - r, 0), min(col + r+1, width))
    y = np.arange(max(row - r, 0), min(row + r+1, height))
    X, Y = np.meshgrid(x, y)
    # Narrows down the desired indices using Euclidean norm
    # (i.e. cutting off corners of square to make circle)
    R = np.sqrt(((X-np.float(col))**2+(Y-np.float(row))**2))
    mask = (R<radius)
    # Return the indices of flattened array and corresponding distances
    return (X[mask] + Y[mask]*width, R[mask])

# Helper function used to display the images.
def displayPosNeg(img_color,pos,neg):
    '''
    Displays the original image along with the positive and negative
    segments of the image.

    Inputs:
        img_color (array): Original image
        pos (array): Positive segment of the original image
        neg (array): Negative segment of the original image
    Returns:
        Plots the original image along with the positive and negative
            segmentations.
    '''
    plt.subplot(131)
    plt.imshow(neg)
    plt.subplot(132)
    plt.imshow(pos)
    plt.subplot(133)
    plt.imshow(img_color)
    plt.show()
