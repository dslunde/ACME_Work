# pagerank.py
"""Volume 1: The Page Rank Algorithm.
    THOR
    How to be a God Part 1
    Today
    """

from scipy import linalg as la
from scipy.sparse import dok_matrix as dk
import numpy as np

# Problem 1
def to_matrix(filename, n):
    """Return the nxn adjacency matrix described by datafile.
        
        Parameters:
        datafile (str): The name of a .txt file describing a directed graph.
        Lines describing edges should have the form '<from node>\t<to node>\n'.
        The file may also include comments.
        n (int): The number of nodes in the graph described by datafile
        
        Returns:
        A SciPy sparse dok_matrix.
        """
#raise NotImplementedError("Problem 1 Incomplete")
    A = dk((n,n),dtype = np.float32)
    with open(filename,'r') as file :
        for line in file :
            try:
                edge = line.strip().split()
                A[int(edge[0]),int(edge[1])] = 1
            except ValueError:
                pass
    return A

# Problem 2
def calculateK(A,N):
    """Compute the matrix K as described in the lab.
        
        Parameters:
        A (ndarray): adjacency matrix of an array
        N (int): the datasize of the array
        
        Returns:
        K (ndarray)
        """
#raise NotImplementedError("Problem 2 Incomplete")
    if type(A) != np.ndarray :
        A = A.toarray()
    A = A.astype(np.float32)
    B = np.copy(A)
    zero_row = np.zeros((1,N))
    for i in xrange(N) :
        if np.allclose(B[i,:],zero_row) :
            B[i,:] = np.ones((1,N))
    D = np.copy(zero_row)
    for i in xrange(N) :
        D += B[:,i]
    D = np.diag(1./D[0])
    K = np.dot(D,B).T
    return K

# Problem 3
def iter_solve(adj, N=None, d=.85, tol=1E-5):
    """Return the page ranks of the network described by 'adj'.
        Iterate through the PageRank algorithm until the error is less than 'tol'.
        
        Parameters:
        adj (ndarray): The adjacency matrix of a directed graph.
        N (int): Restrict the computation to the first 'N' nodes of the graph.
        If N is None (default), use the entire matrix.
        d (float): The damping factor, a float between 0 and 1.
        tol (float): Stop iterating when the change in approximations to the
        solution is less than 'tol'.
        
        Returns:
        The approximation to the steady state.
        """
#raise NotImplementedError("Problem 3 Incomplete")
    m,n = adj.shape
    if N == None :
        N = n
    p = np.random.rand(N).astype(np.float32)
    p = p/np.sum(p)

    adj = adj[:N,:N]
    K = calculateK(adj,N)

    iter = 0
    while iter < 10000 :
        pnext = d*np.dot(K,p)+float(1-d)/N*np.ones(N)
        if la.norm(pnext-p) < tol :
            return pnext
        iter += 1
        if pnext.shape != p.shape :
            raise ValueError("p changed sizes.")
        p = pnext
    raise ValueError("Did not converge.")

# Problem 4
def eig_solve(adj, N=None, d=.85):
    """Return the page ranks of the network described by 'adj'. Use SciPy's
        eigenvalue solver to calculate the steady state of the PageRank algorithm
        
        Parameters:
        adj (ndarray): The adjacency matrix of a directed graph.
        N (int): Restrict the computation to the first 'N' nodes of the graph.
        If N is None (default), use the entire matrix.
        d (float): The damping factor, a float between 0 and 1.
        tol (float): Stop iterating when the change in approximations to the
        solution is less than 'tol'.
        
        Returns:
        The approximation to the steady state.
        """
#raise NotImplementedError("Problem 4 Incomplete")
    m,n = adj.shape
    if N == None :
        N = n
    p = np.random.rand(N).astype(np.float32)
    p = p/np.sum(p)
    
    adj = adj[:N,:N]
    K = calculateK(adj,N)

    B = d*K+float(1-d)/N*np.ones(N)
    eig_vals,eig_vect = la.eig(B)

    return eig_vect[:,0]/np.sum(eig_vect[:,0])

# Problem 5
def team_rank(filename='ncaa2013.csv'):
    """Use iter_solve() to predict the rankings of the teams in the given
        dataset of games. The dataset should have two columns, representing
        winning and losing teams. Each row represents a game, with the winner on
        the left, loser on the right. Parse this data to create the adjacency
        matrix, and feed this into the solver to predict the team ranks.
        
        Parameters:
        filename (str): The name of the data file.
        Returns:
        ranks (list): The ranks of the teams from best to worst.
        teams (list): The names of the teams, also from best to worst.
        """
#raise NotImplementedError("Problem 5 Incomplete")
    team_dict = {}
    games = []
    with open(filename,'r') as file :
        file.readline()
        for line in file :
            teams = line.strip().split(',')
            for team in teams :
                if team not in team_dict.keys() :
                    team_dict[team] = len(team_dict.keys())
            games.append(teams)

    n = len(team_dict.keys())
    A = np.zeros((n,n))
    for game in games :
        A[team_dict[game[1]],team_dict[game[0]]] = 1
    rankings = iter_solve(A,d=0.7)
    sort = np.argsort(rankings)
    ranks = sorted(rankings)[::-1]
    team_order = [team_dict.keys()[team_dict.values().index(sort[i])] for i in xrange(n)][::-1]
    return ranks,team_order


