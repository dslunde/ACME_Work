# drazin.py
"""Volume 1: The Drazin Inverse.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
import scipy.sparse.csgraph as cg
import pandas as pd


# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.allclose(la.det(A),0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        bool: True of Ad is the Drazin inverse of A, False otherwise.
    """
#raise NotImplementedError("Problem 1 Incomplete")
    drazin = False
    if np.allclose(np.dot(A,Ad),np.dot(Ad,A)) :
        if np.allclose(Ad,np.dot(Ad,np.dot(A,Ad))) :
            if np.allclose(np.dot(np.linalg.matrix_power(A,k+1),Ad),np.linalg.matrix_power(A,k)) :
                drazin = True
    return drazin


# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        Ad ((n,n) ndarray): The Drazin inverse of A.
    """
#raise NotImplementedError("Problem 2 Incomplete")
    n = A.shape[0]
    Q1,S,k = la.schur(A,sort = lambda x : abs(x) > tol)
    Q2,T,k1 = la.schur(A,sort = lambda x : abs(x) <= tol)
    U = np.hstack((S[:,:k],T[:,:(n-k)]))
    Uinv = la.inv(U)
    V = np.dot(Uinv,np.dot(A,U))
    Z = np.zeros((n,n))
    if k != 0 :
        Z[:k,:k] = la.inv(V[:k,:k])
    return np.dot(U,np.dot(Z,Uinv))

# Problem 3
def effective_res(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ER ((n,n) ndarray): A matrix of which the ijth entry is the effective
        resistance from node i to node j.
    """
#raise NotImplementedError("Problem 3 Incomplete")
    lap = cg.laplacian(A)
    n = A.shape[0]
    R = np.zeros((n,n))
    for i in xrange(n) :
        L = np.copy(lap)
        L[:,i] = np.eye(1,n,i)
        D = drazin_inverse(L)
        R[:,i] = np.diag(D)
    np.fill_diagonal(R,0)
    return R


# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.
        
        Parameters:
            filename (str): The name of a file containing graph data.
        """
    #raise NotImplementedError("Problem 4 Incomplete")
        df = pd.read_csv(filename,header=None)
        data = df.values
        peeps = {}
        count = 0
        for i in xrange(data.shape[0]) :
            if data[i,0] not in peeps.keys() :
                peeps[data[i,0]] = [data[i,1]]
            else :
                peeps[data[i,0]].append(data[i,1])
            if data[i,1] not in peeps.keys() :
                peeps[data[i,1]] = [data[i,0]]
            else :
                peeps[data[i,1]].append(data[i,0])
        n = len(peeps.keys())
        self.people = peeps.keys()
        self.people.sort()
        self.A = np.zeros((n,n))
        for i in xrange(n) :
            friends = peeps[self.people[i]]
            for j in xrange(len(friends)) :
                col = self.people.index(friends[j])
                if self.A[i,col] != 1 :
                    self.A[i,col] = 1
                    self.A[col,i] = 1
        self.ER = effective_res(self.A)
        mask = self.A == 0
        self.adj_er = self.ER * mask
                        


    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.
        
        Parameters:
            node (str): The name of a node in the network.
        
        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.
        
        Raises:
            ValueError: If node is not in the graph.
        """
    #raise NotImplementedError("Problem 5 Incomplete")
        er_max = np.max(self.adj_er)
        mask = self.adj_er == 0
        unzeroed_er = self.adj_er + (er_max+1)*mask
        if node :
            if node not in self.people :
                raise ValueError('Person not found.')
            pos = self.people.index(node)
            pers = np.argmin(unzeroed_er,axis=0)[pos]
            return self.people[pers]
        else :
            pos = np.argmin(unzeroed_er)
            row = pos / self.A.shape[0]
            col = pos % self.A.shape[0]
            return self.people[row],self.people[col]

    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """
#raise NotImplementedError("Problem 5 Incomplete")
        if node1 not in self.people or node2 not in self.people :
            raise ValueError('One of those individuals is not in the network.')
        row = self.people.index(node1)
        col = self.people.index(node2)
        if self.A[row,col] != 1 :
            self.A[row,col],self.A[col,row] = 1,1
            self.ER = effective_res(self.A)
            mask = self.A == 0
            self.adj_er = self.ER * mask


