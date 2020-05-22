# Problem 3
"""In each process, generate a random number, then send that number to the
process with the next highest rank (the last process should send to the root).
Print what each process starts with and what each process receives.

Usage:
    $ mpiexec -n 2 python problem3.py
    Process 1 started with [ 0.79711384]        # Values and order will vary.
    Process 1 received [ 0.54029085]
    Process 0 started with [ 0.54029085]
    Process 0 received [ 0.79711384]

    $ mpiexec -n 3 python problem3.py
    Process 2 started with [ 0.99893055]
    Process 0 started with [ 0.6304739]
    Process 1 started with [ 0.28834079]
    Process 1 received [ 0.6304739]
    Process 2 received [ 0.28834079]
    Process 0 received [ 0.99893055]
"""
from mpi4py import MPI
import numpy as np

if __name__ == "__main__" :
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    comms = COMM.Get_size()
    n = np.array(np.random.rand())
    m = np.zeros(1)
    if (RANK+1) == comms :
        COMM.Send(n,dest=0)
        COMM.Recv(m,source=(RANK-1))
    elif RANK == 0 :
        COMM.Send(n,dest=(RANK+1))
        COMM.Recv(m,source=(comms-1))
    else :
        COMM.Send(n,dest=(RANK+1))
        COMM.Recv(m,source=(RANK-1))
    print('Process {} started with {}'.format(RANK,n))
    print('Process {} finished with {}'.format(RANK,m))