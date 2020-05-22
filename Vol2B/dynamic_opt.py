# dynamic_opt.py
"""Volume 2: Dynamic Optimization.
<Name>
<Class>
<Date>
"""

import numpy as np
from matplotlib import pyplot as plt


# Problem 1
def graph_policy(policy, b, u):
    """Plot the utility gained over time.
    Return the total utility gained with the policy given.

    Parameters:
        policy (ndarray): Policy vector.
        b (float): Discount factor. 0 < beta < 1.
        u (function): Utility function.

    Returns:
        total_utility (float): Total utility gained from the policy given.
    """
#raise NotImplementedError("Problem 1 Incomplete")
    if np.sum(policy) != 1 :
        raise ValueError("Policy must sum to 1.")
    n = policy.shape[0]
    x = np.linspace(1,n,n)
    utils = [b**i*u(policy[i]) for i in xrange(n)]
    y = np.array([sum(utils[:i+1]) for i in xrange(n)])
    plt.plot(x,y)
    plt.xlabel("Time")
    plt.ylabel("Total Utility")
    plt.title("Graphing the optimal Policy")
    plt.show()
    return y[n-1]

# Problem 2
def consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): Consumption matrix.
    """
#raise NotImplementedError("Problem 2 Incomplete")
    w = [i/float(N) for i in xrange(N+1)]
    C = np.zeros((N+1,N+1))
    for i in xrange(1,N+1) :
        for j in xrange(i) :
            C[i,j] = u(w[i-j])
    return C



# Problems 3-5
def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
#raise NotImplementedError("Problems 3-5 Incomplete")
    w = [i/float(N) for i in xrange(N+1)]
    A = np.zeros((N+1,T+1))
    P = np.zeros((N+1,T+1))
    P[:,T] = w
    C = consumption(N,u)
    for i in xrange(N+1) :
        A[i,T] = u(w[i])
    for i in xrange(T,0,-1) :
        R = np.diag(A[:,i])
        for j in xrange(R.shape[0]) :
            R[j:,j] = R[j,j]
        np.fill_diagonal(R,0)
        temp = C + B*R
        A[:,i-1] = np.max(temp,axis=1)
        for k in xrange(P.shape[0]) :
            P[k,i-1] = w[k] - w[np.argmax(temp,axis=1)[k]]
    return A,P


# Problem 6
def find_policy(T, N, B, u=lambda x: np.sqrt(x)):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        maximum_utility (float): The total utility gained from the
            optimal policy.
        optimal_policy ((N,) nd array): The matrix describing the optimal
            percentage to consume at each time.
    """
#raise NotImplementedError("Problem 6 Incomplete")
    A,P = eat_cake(T,N,B,u)
    n = P.shape[1]
    policy = []
    et = 0
    for i in xrange(n) :
        row = n*(1-et)
        if not np.allclose(row,int(row)) :
            row += 0.9
        row = int(row)
        policy.append(P[row,i])
        et += P[row,i]
    policy = np.array(policy)
    graph_policy(policy,B,u)
    max_util = sum([B**i*u(policy[i]) for i in xrange(policy.shape[0])])
    return max_util,policy

