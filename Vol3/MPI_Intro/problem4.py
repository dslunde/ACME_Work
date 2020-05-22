# Problem 4
"""The n-dimensional open unit ball is the set U_n = {x in R^n : ||x|| < 1}.
Estimate the volume of U_n by making N draws on each available process except
for the root process. Have the root process print the volume estimate.

Command line arguments:
    n (int): the dimension of the unit ball.
    N (int): the number of random draws to make on each process but the root.

Usage:
    # Estimate the volume of U_2 (the unit circle) with 2000 draws per process.
    $ mpiexec -n 4 python problem4.py 2 2000
    Volume of 2-D unit ball: 3.13266666667      # Results will vary slightly.
"""
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from sys import argv
import numpy as np
from scipy.linalg import norm

if __name__ == "__main__" :
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    r = COMM.Get_size()
    n = int(argv[1])
    N = int(argv[2])
    if RANK != 0 :
        pulls = np.random.rand(n,N)
        norms = norm(pulls,axis=0)
        M = np.array(np.count_nonzero([norms[i] < 1 for i in range(len(norms))]))
        COMM.Send(M,dest=0)
    else :
        count = 0
        total = 0
        m = np.zeros(1,dtype=int)
        while count < r-1 :
            COMM.Recv(m,source=ANY_SOURCE)
            total += m[0]
            count += 1
        print(2**n*total/((r-1)*N))