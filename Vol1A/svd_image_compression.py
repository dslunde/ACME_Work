# svd_image_compression.py
"""Volume 1A: SVD and Image Compression.
Darren Lund
Math 345
12/2/16
"""

from scipy import linalg as la
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def truncated_svd(A,k=None):
    """Computes the truncated SVD of A. If k is None or equals the number
        of nonzero singular values, it is the compact SVD.
    Parameters:
        A: the matrix
        k: the number of singular values to use
    Returns:
        U - the matrix U in the SVD
        s - the diagonals of Sigma in the SVD
        Vh - the matrix V^H in the SVD
    """
    #raise NotImplementedError("truncated_svd incomplete")
    e_vals,e_vecs = la.eig(np.dot(A.conj().T,A))
    s_evals = list(np.argsort(e_vals))
    e_vals.sort()
    e_vals = e_vals[::-1]
    e_vals = [x for x in e_vals if not np.isclose(x,0)]
    e_vals = e_vals[:k]
    if len(e_vals) == 0 :
        raise ValueError("Matrix only has 0 for eigenvalues.")
    else :
        for i in xrange(len(e_vals)) :
            e_vals[i] = e_vals[i]**(.5)
        z = s_evals.index(max(s_evals))
        V = e_vecs[:,z]
        U = np.dot(A,V)/float(e_vals[0])
        s_evals[z] = 0
        for i in xrange(1,len(e_vals)) :
            z = s_evals.index(max(s_evals))
            V = np.vstack((V,e_vecs[:,z]))
            s_evals[z] = 0
            U = np.vstack((U,np.dot(A,V[i,:])/float(e_vals[i])))
        return U.T,e_vals,V.conj()


# Problem 2
def visualize_svd():
    """Plot each transformation associated with the SVD of A."""
    #raise NotImplementedError("visualize_svd incomplete")
    A = np.array([[3,1],[1,3]])
    U,s,Vh = truncated_svd(A)
    S = np.diag(s)

    theta = np.linspace(0,2*np.pi,100)
    xc = np.cos(theta)
    yc = np.sin(theta)

    xe1 = np.linspace(0,1,100)
    ye1 = [0]*100

    xe2 = [0]*100
    ye2 = np.linspace(0,1,100)

    plt.subplot(221)
    plt.axis('equal')
    plt.plot(xc,yc,'k')
    plt.plot(xe1,ye1,'g')
    plt.plot(xe2,ye2,'g')

    V = Vh

    xca = [sum(x) for x in zip([x*V[0,0] for x in xc],[x*V[0,1] for x in yc])]
    yca = [sum(x) for x in zip([x*V[1,0] for x in xc],[x*V[1,1] for x in yc])]
    xe1a = [sum(x) for x in zip([x*V[0,0] for x in xe1],[y*V[0,1] for y in ye1])]
    ye1a = [sum(x) for x in zip([x*V[1,0] for x in xe1],[y*V[1,1] for y in ye1])]
    xe2a = [sum(x) for x in zip([x*V[0,0] for x in xe2],[y*V[0,1] for y in ye2])]
    ye2a = [sum(x) for x in zip([x*V[1,0] for x in xe2],[y*V[1,1] for y in ye2])]

    plt.subplot(222)
    plt.axis('equal')
    plt.plot(xca,yca,'k')
    plt.plot(xe1a,ye1a,'g')
    plt.plot(xe2a,ye2a,'g')

    V = np.dot(S,V)

    xca = [sum(x) for x in zip([x*V[0,0] for x in xc],[x*V[0,1] for x in yc])]
    yca = [sum(x) for x in zip([x*V[1,0] for x in xc],[x*V[1,1] for x in yc])]
    xe1a = [sum(x) for x in zip([x*V[0,0] for x in xe1],[y*V[0,1] for y in ye1])]
    ye1a = [sum(x) for x in zip([x*V[1,0] for x in xe1],[y*V[1,1] for y in ye1])]
    xe2a = [sum(x) for x in zip([x*V[0,0] for x in xe2],[y*V[0,1] for y in ye2])]
    ye2a = [sum(x) for x in zip([x*V[1,0] for x in xe2],[y*V[1,1] for y in ye2])]

    plt.subplot(223)
    plt.axis('equal')
    plt.plot(xca,yca,'k')
    plt.plot(xe1a,ye1a,'g')
    plt.plot(xe2a,ye2a,'g')

    V = np.dot(U,V)

    xca = [sum(x) for x in zip([x*V[0,0] for x in xc],[x*V[0,1] for x in yc])]
    yca = [sum(x) for x in zip([x*V[1,0] for x in xc],[x*V[1,1] for x in yc])]
    xe1a = [sum(x) for x in zip([x*V[0,0] for x in xe1],[y*V[0,1] for y in ye1])]
    ye1a = [sum(x) for x in zip([x*V[1,0] for x in xe1],[y*V[1,1] for y in ye1])]
    xe2a = [sum(x) for x in zip([x*V[0,0] for x in xe2],[y*V[0,1] for y in ye2])]
    ye2a = [sum(x) for x in zip([x*V[1,0] for x in xe2],[y*V[1,1] for y in ye2])]

    plt.subplot(224)
    plt.axis('equal')
    plt.plot(xca,yca,'k')
    plt.plot(xe1a,ye1a,'g')
    plt.plot(xe2a,ye2a,'g')

    plt.show()

# Problem 3
def svd_approx(A, k):
    """Returns best rank k approximation to A with respect to the induced 2-norm.

    Inputs:
    A - np.ndarray of size mxn
    k - rank

    Return:
    Ahat - the best rank k approximation
    """
    #raise NotImplementedError("svd_approx incomplete")
    U,s,Vh = la.svd(A)
    S = np.diag(s[:k])
    return U[:,:k].dot(S).dot(Vh[:k,:])

# Problem 4
def lowest_rank_approx(A,e):
    """Returns the lowest rank approximation of A with error less than e
    with respect to the induced 2-norm.

    Inputs:
    A - np.ndarray of size mxn
    e - error

    Return:
    Ahat - the lowest rank approximation of A with error less than e.
    """
    #raise NotImplementedError("lowest_rank_approx incomplete")
    U,s,Vh = la.svd(A)
    counter = 0
    while counter < len(s) and s[counter] > e:
        counter += 1
    S = np.diag(s[:counter])
    return U[:,:counter].dot(S).dot(Vh[:counter,:])

# Problem 5
def compress_image(filename,k):
    """Plot the original image found at 'filename' and the rank k approximation
    of the image found at 'filename.'

    filename - jpg image file path
    k - rank
    """
    #raise NotImplementedError("compress_image incomplete")
    X = plt.imread(filename)
    Y = np.zeros_like(X)
    if len(X.shape) == 3 :
        R = X[:,:,0]
        B = X[:,:,1]
        G = X[:,:,2]
        R = svd_approx(R,k)
        B = svd_approx(B,k)
        G = svd_approx(G,k)
        #X[:,:,0] = X[:,:,0]/255.
        #X[:,:,1] = X[:,:,1]/255.
        #X[:,:,2] = X[:,:,2]/255.
        Y[:,:,0] = R#/255.
        Y[:,:,1] = B#/255.
        Y[:,:,2] = G#/255.
    elif len(X.shape) == 2 :
        X = lowest_rank_approx(X,k)

    plt.subplot(121)
    plt.imshow(X)
    plt.subplot(122)
    plt.imshow(Y)

    plt.show()
