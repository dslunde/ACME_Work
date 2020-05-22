# shell2.py
"""Volume 3: Unix Shell 2.
Darren Lund
Batman
Today
"""
import os
import subprocess
from glob import glob

# Problem 5
def grep(target_string, file_pattern):
    """Find all files in the current directory or its subdirectories that
    match the file pattern, then determine which ones contain the target
    string.

    Parameters:
        target_string (str): A string to search for in the files whose names
            match the file_pattern.
        file_pattern (str): Specifies which files to search.
    """
    #raise NotImplementedError("Problem 5 Incomplete")
    master_list = [] 
    for direct,subdirect,file in os.walk('.') :
        files = glob(str(direct)+'/'+file_pattern)
        for file in files :
            with open(file,'r') as f :
                if target_string in f.read() :
                    master_list.append(file)
    return master_list

# Problem 6
def largest_files(n):
    """Return a list of the n largest files in the current directory or its
    subdirectories (from largest to smallest).
    """
    #raise NotImplementedError("Problem 6 Incomplete")
    sizes = []
    names = []
    for dir , subdir, files in os.walk('.') :
        for f in files :
            file_name = str(f)
            size = os.path.getsize(os.path.join(dir,f))
            names.append(file_name)
            sizes.append(size)
    largest = []
    for i in range(n) :
        ind = sizes.index(max(sizes))
        largest.append(names[ind])
        sizes[ind] = 0
    return largest 
