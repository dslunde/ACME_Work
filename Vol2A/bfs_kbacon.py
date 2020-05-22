# bfs_kbacon.py
"""Volume 2A: Breadth-First Search (Kevin Bacon).
Darren Lund
Math 321
10/31/16
"""

from collections import deque
from matplotlib import pyplot as plt
import networkx as nx

# Problems 1-4: Implement the following class
class Graph(object):
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a list of the
    corresponding node's neighbors.

    Attributes:
        dictionary: the adjacency list of the graph.
    """

    def __init__(self, adjacency):
        """Store the adjacency dictionary as a class attribute."""
        self.dictionary = adjacency

    # Problem 1
    def __str__(self):
        """String representation: a sorted view of the adjacency dictionary.

        Example:
            >>> test = {'A':['B'], 'B':['A', 'C',], 'C':['B']}
            >>> print(Graph(test))
            A: B
            B: A; C
            C: B
        """
        #raise NotImplementedError("Problem 1 Incomplete")
        graph = ""
        for k in sorted(self.dictionary.keys()) :
             graph += k + ": " + "; ".join(self.dictionary[k]) + "\n"
        return graph[:-1]

    # Problem 2
    def traverse(self, start):
        """Begin at 'start' and perform a breadth-first search until all
        nodes in the graph have been visited. Return a list of values,
        in the order that they were visited.

        Inputs:
            start: the node to start the search at.

        Returns:
            the list of visited nodes (in order of visitation).

        Raises:
            ValueError: if 'start' is not in the adjacency dictionary.

        Example:
            >>> test = {'A':['B'], 'B':['A', 'C',], 'C':['B']}
            >>> Graph(test).traverse('B')
            ['B', 'A', 'C']
        """
        #raise NotImplementedError("Problem 2 Incomplete")
        if start not in self.dictionary.keys() :
            raise ValueError("Invalid starting node.")
        elif len(self.dictionary[start]) == 0 :
            return [start]
        else :
            to_visit = deque()
            marked = set()
            marked.add(start)
            order = [start]
            for n in self.dictionary[start] :
                to_visit.append(n)
                marked.add(n)
            current = start
            while to_visit :
                current = to_visit.popleft()
                for n in self.dictionary[current] :
                    if n not in marked :
                        to_visit.append(n)
                        marked.add(n)
                order.append(current)
            return order

    # Problem 3 (Optional)
    def DFS(self, start):
        """Begin at 'start' and perform a depth-first search until all
        nodes in the graph have been visited. Return a list of values,
        in the order that they were visited. If 'start' is not in the
        adjacency dictionary, raise a ValueError.

        Inputs:
            start: the node to start the search at.

        Returns:
            the list of visited nodes (in order of visitation)
        """
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 4
    def shortest_path(self, start, target):
        """Begin at the node containing 'start' and perform a breadth-first
        search until the node containing 'target' is found. Return a list
        containg the shortest path from 'start' to 'target'. If either of
        the inputs are not in the adjacency graph, raise a ValueError.

        Inputs:
            start: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from start to target,
                including the endpoints.

        Example:
            >>> test = {'A':['B', 'F'], 'B':['A', 'C'], 'C':['B', 'D'],
            ...         'D':['C', 'E'], 'E':['D', 'F'], 'F':['A', 'E', 'G'],
            ...         'G':['A', 'F']}
            >>> Graph(test).shortest_path('A', 'G')
            ['A', 'F', 'G']
        """
        #raise NotImplementedError("Problem 4 Incomplete")
        if start not in self.dictionary.keys() or target not in self.dictionary.keys():
            raise ValueError("Invalid start or target node.")
        elif len(self.dictionary[start]) == 0 :
            if start == target :
                return [start]
            else :
                raise RuntimeError("Path nonexistent.")
        else :
            to_visit = deque()
            marked = set()
            marked.add(start)
            path_dict = {}
            for n in self.dictionary[start] :
                to_visit.append(n)
                marked.add(n)
                path_dict[n] = start
            current = start
            while current != target and to_visit:
                current = to_visit.popleft()
                for n in self.dictionary[current] :
                    if n not in marked :
                        to_visit.append(n)
                        marked.add(n)
                        if n not in path_dict.keys() :
                            path_dict[n] = current
            if current != target :
                print path_dict
                raise RuntimeError("Path nonexistent.")
            else :
                path = [target]
                while current != start:
                    path.append(path_dict[current])
                    current = path_dict[current]
                return path[::-1]


# Problem 5: Write the following function
def convert_to_networkx(diction):
    """Convert 'dictionary' to a networkX object and return it."""
    #raise NotImplementedError("Problem 5 Incomplete")
    graph = Graph(diction)
    nx_graph = nx.Graph()
    for n in graph.dictionary.keys() :
        for m in graph.dictionary[n] :
            nx_graph.add_edge(n,m)
    return nx_graph


# Helper function for problem 6
def parse(filename="movieData.txt"):
    """Generate an adjacency dictionary where each key is
    a movie and each value is a list of actors in the movie.
    """

    # open the file, read it in, and split the text by '\n'
    with open(filename, 'r') as movieFile:
        moviesList = movieFile.read().split('\n')
    graph = dict()

    # for each movie in the file,
    for movie in moviesList:
        # get movie name and list of actors
        names = movie.split('/')
        title = names[0]
        graph[title] = []
        # add the actors to the dictionary
        for actor in names[1:]:
            graph[title].append(actor)

    return graph


# Problems 6-8: Implement the following class
class BaconSolver(object):
    """Class for solving the Kevin Bacon problem."""

    # Problem 6
    def __init__(self, filename="movieData.txt"):
        """Initialize the networkX graph and with data from the specified
        file. Store the graph as a class attribute. Also store the collection
        of actors in the file as an attribute.
        """
        #raise NotImplementedError("Problem 6 Incomplete")
        if not isinstance(filename,str):
            raise TypeError("Invalid file name.")
        else :
            movie_dict = parse(filename)
            self.actors = set()
            for movie in movie_dict.keys() :
                for actor in movie_dict[movie] :
                    self.actors.add(actor)
            self.movie_graph = convert_to_networkx(movie_dict)

    # Problem 6
    def path_to_bacon(self, start, target="Bacon, Kevin"):
        """Find the shortest path from 'start' to 'target'."""
        #raise NotImplementedError("Problem 6 Incomplete")
        if start not in self.actors or target not in self.actors:
            raise ValueError("Unkown actor for either start or target.")
        else :
            return nx.shortest_path(self.movie_graph,start,target)

    # Problem 7
    def bacon_number(self, start, target="Bacon, Kevin"):
        """Return the Bacon number of 'start'."""
        #raise NotImplementedError("Problem 7 Incomplete")
        try :
            path = self.path_to_bacon(start,target)
            return len(path)/2
        except ValueError :
            raise ValueError("Starting or target actor unkown.")

    # Problem 7
    def average_bacon(self, target="Bacon, Kevin"):
        """Calculate the average Bacon number in the data set.
        Note that actors are not guaranteed to be connected to the target.

        Inputs:
            target (str): the node to search the graph for
        """
        #raise NotImplementedError("Problem 7 Incomplete")
        bacon_numbers = []
        not_connected = []
        for actor in self.actors :
            if nx.has_path(self.movie_graph,actor,target) :
                bacon_numbers.append(self.bacon_number(actor))
            else :
                not_connected.append(actor)
        if len(bacon_numbers) != 0 :
            return sum(bacon_numbers) / float(len(bacon_numbers)) , len(not_connected)
        else :
            return 0, len(not_connected)
# =========================== END OF FILE =============================== #
