# Problem 1
"""Print 'Hello from process n' from processes with an even rank and
print 'Goodbye from process n' from processes with an odd rank (where
n is the rank).

Usage:
    $ mpiexec -n 4 python problem1.py
    Goodbye from process 3                  # Order of outputs will vary.
    Hello from process 0
    Goodbye from process 1
    Hello from process 2

    # python problem1.py
    Hello from process 0
"""
from mpi4py import MPI

if __name__ == "__main__" :
    RANK = MPI.COMM_WORLD.Get_rank()
    if RANK % 2 == 0 :
        print("Hello from processor {}".format(RANK))
    else :
        print("Goodbye from processor {}".format(RANK))