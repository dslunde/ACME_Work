# Problem 2
"""Pass a random NumPy array of shape (n,) from the root process to process 1,
where n is a command-line argument. Print the array and process number from
each process.

Usage:
    # This script must be run with 2 processes.
    $ mpiexec -n 2 python problem2.py 4
    Process 1: Before checking mailbox: vec=[ 0.  0.  0.  0.]
    Process 0: Sent: vec=[ 0.03162613  0.38340242  0.27480538  0.56390755]
    Process 1: Recieved: vec=[ 0.03162613  0.38340242  0.27480538  0.56390755]
"""
from mpi4py import MPI
from sys import argv
import numpy as np

if __name__ == "__main__" :
    n = int(argv[1])
    RANK = MPI.COMM_WORLD.Get_rank()
    if RANK == 0 :
        a = np.array(np.random.rand(n))
        print('Sending {} from processor {} to 1'.format(a,RANK))
        MPI.COMM_WORLD.Send(a,dest=1)
        print('Message sent!')
    elif RANK == 1 :
        a = np.zeros(n)
        print('Receiving message from processor 0:')
        MPI.COMM_WORLD.Recv(a,source=0)
        print('Received the following message: {}'.format(a))