# kdtrees.py
"""Volume 2A: Data Structures 3 (K-d Trees).
Darren Lund
321
10/13/16
"""

from sklearn import neighbors
from scipy.spatial import KDTree
from trees import BST
from trees import BSTNode
import numpy as np
import math

# Problem 1
def metric(x, y):
    """Return the euclidean distance between the 1-D arrays 'x' and 'y'.

    Raises:
        ValueError: if 'x' and 'y' have different lengths.

    Example:
        >>> metric([1,2],[2,2])
        1.0
        >>> metric([1,2,1],[2,2])
        ValueError: Incompatible dimensions.
    """
    #raise NotImplementedError("Problem 1 Incomplete")
    if len(x) != len(y) :
        raise ValueError("The points must be in the same space (sizes are different).")
    else :
        d = [(x[i]-y[i])**2 for i in xrange(len(x))]
        return math.sqrt(sum(d))


# Problem 2
def exhaustive_search(data_set, target):
    """Solve the nearest neighbor search problem exhaustively.
    Check the distances between 'target' and each point in 'data_set'.
    Use the Euclidean metric to calculate distances.

    Inputs:
        data_set ((m,k) ndarray): An array of m k-dimensional points.
        target ((k,) ndarray): A k-dimensional point to compare to 'dataset'.

    Returns:
        ((k,) ndarray) the member of 'data_set' that is nearest to 'target'.
        (float) The distance from the nearest neighbor to 'target'.
    """
    #raise NotImplementedError("Problem 2 Incomplete")
    min_dist = float("inf")
    row = float("inf")
    for i in xrange(data_set.shape[0]) :
        dist = metric(data_set[i,:],target)
        if dist < min_dist :
            min_dist = dist
            row = i
    return data_set[row,:] , min_dist


# Problem 3: Write a KDTNode class.
class KDTNode(BSTNode) :
    """A k-dimensional binary search tree node object.
    Used to store specific data points in k dimensions.
    Has two children, a left one where every data element is less than it
    in the ith dimension at level i (root is level 0) and a right which is bigger.
    Also has an axis element to say what dimension it is (i in this case).
    """
    def __init__(self,data) :
        if type(data).__module__ != np.__name__ :
            raise TypeError("Data must be a k-dimensional array.")
        else :
            self.axis = 0
            BSTNode.__init__(self,data)

    def __str__(self) :
        return str(self.value)


# Problem 4: Finish implementing this class by overriding
#            the __init__(), insert(), and remove() methods.
class KDT(BST):
    """A k-dimensional binary search tree object.
    Used to solve the nearest neighbor problem efficiently.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other
            nodes in the tree, the root houses data as a NumPy array.
        k (int): the dimension of the tree (the 'k' of the k-d tree).
    """
    def __init__(self,data_set) :
        BST.__init__(self)
        for i in xrange(data_set.shape[0]) :
            self.insert(data_set[i,:])

    def find(self, data):
        """Return the node containing 'data'. If there is no such node
        in the tree, or if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            'data' is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.axis] < current.value[current.axis]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursion on the root of the tree.
        return _step(self.root)

    def insert(self, data):
        """Insert a new node containing 'data' at the appropriate location.
        Return the new node. This method should be similar to BST.insert().
        """
        #raise NotImplementedError("Problem 4 Incomplete")
        def inrecurse(current,data) :
            if np.allclose(data, current.value) :
                raise ValueError("Element is already in the tree.")
            elif data[current.axis] < current.value[current.axis] :
                if current.left == None :
                    current.left = KDTNode(data)
                    current.left.axis = (current.axis+1) % len(data)
                    current.left.prev = current
                    return current.left.value
                else :
                    inrecurse(current.left,data)
            else :
                if current.right == None :
                    current.right = KDTNode(data)
                    current.right.axis = (current.axis+1) % len(data)
                    current.right.prev = current
                    return current.right.value
                else :
                    inrecurse(current.right,data)

        if self.root == None :
            self.root = KDTNode(data)
            return self.root.value
        else :
            inrecurse(self.root,data)

    def remove(*arg) :
        raise NotImplementedError("Remove disabled for KD tree.")


# Problem 5
def nearest_neighbor(data_set, target):
    """Use your KDT class to solve the nearest neighbor problem.

    Inputs:
        data_set ((m,k) ndarray): An array of m k-dimensional points.
        target ((k,) ndarray): A k-dimensional point to compare to 'dataset'.

    Returns:
        The point in the tree that is nearest to 'target' ((k,) ndarray).
        The distance from the nearest neighbor to 'target' (float).
    """
    def KDTsearch(current, neighbor, distance):
        """The actual nearest neighbor search algorithm.

        Inputs:
            current (KDTNode): the node to examine.
            neighbor (KDTNode): the current nearest neighbor.
            distance (float): the current minimum distance.

        Returns:
            neighbor (KDTNode): The new nearest neighbor in the tree.
            distance (float): the new minimum distance.
        """
        #print "Call"
        if current == None :
            return neighbor,distance
        index = current.axis
        #print index,distance
        if metric(current.value,target) < distance :
            neighbor = current
            distance = metric(current.value,target)
        if current.value[index] > target[index] :
            neighbor,distance = KDTsearch(current.left,neighbor,distance)
            if target[index] + distance >= current.value[index] :
                neighbor,distance = KDTsearch(current.right,neighbor,distance)
        else :
            neighbor,distance = KDTsearch(current.right,neighbor,distance)
            if target[index] - dist <= current.value[index] :
                neighbor,distance = KDTsearch(current.left,neighbor,distance)
        return neighbor,distance

    kd = KDT(data_set)
    if kd.root == None :
        return ValueError("No data.")
    else :
        dist = metric(kd.root.value,target)
        if kd.root.value[0] > target[0] :
            current = kd.root.left
        else :
            current = kd.root.right
        n,d = KDTsearch(current,kd.root,dist)
    return n,d
    #raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def postal_problem():
    """Use the neighbors module in sklearn to classify the Postal data set
    provided in 'PostalData.npz'. Classify the testpoints with 'n_neighbors'
    as 1, 4, or 10, and with 'weights' as 'uniform' or 'distance'. For each
    trial print a report indicating how the classifier performs in terms of
    percentage of correct classifications. Which combination gives the most
    correct classifications?

    Your function should print a report similar to the following:
    n_neighbors = 1, weights = 'distance':  0.903
    n_neighbors = 1, weights =  'uniform':  0.903       (...and so on.)
    """
    #raise NotImplementedError("Problem 6 Incomplete")
    def classification(nclass,wtype) :
        return neighbors.KNeighborsClassifier(n_neighbors=nclass,weights=wtype,p=2)

    labels,points,testlabels,testpoints = np.load('PostalData.npz').items()
    labels = labels[1]
    points = points[1]
    testlabels = testlabels[1]
    testpoints = testpoints[1]
    x = [1,4,10]
    y = ['uniform','distance']

    for i in x :
        for j in y :
            nbrs = classification(i,j)
            nbrs.fit(points,labels)
            prediction = nbrs.predict(testpoints)
            total = 0
            correct = 0
            for p in xrange(len(prediction)) :
                if prediction[p] == testlabels[p] :
                    correct += 1
                total += 1
            a = float(correct)/total
            print "n_neighbors=" + str(i) + ", weight=" + j + ": " + str(a)

if __name__ == "__main__" :
    #a = metric([3,-2,1],[5,2,3])
    #print a
    """
    data = np.random.random((100,5))
    target = np.random.random(5)
    tree = KDTree(data)

    min_dist,index = tree.query(target)
    n,d = nearest_neighbor(data,target)
    print min_dist, " vs " , d
    print tree.data[index] , " vs " , n
    """
    postal_problem()
