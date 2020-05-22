# markov_chains.py
"""Volume II: Markov Chains.
Darren Lund
Math 321
10/31/16
"""

import random
import numpy as np


# Problem 1
def random_markov(n):
    """Create and return a transition matrix for a random Markov chain with
    'n' states. This should be stored as an nxn NumPy array.
    """
    #raise NotImplementedError("Problem 1 Incomplete")
    a = np.random.random(n**2)
    a = a.reshape((n,n))
    for i in xrange(n) :
        a[:,i] = a[:,i]/sum(a[:,i])
    return a

# Problem 2
def forecast(n):
    """Forecast tomorrow's weather given that today is hot."""
    transition = np.array([[0.7, 0.6], [0.3, 0.4]])
    cast = [0]
    for i in xrange(n) :
        # Sample from a binomial distribution to choose a new state.
        cast.append(np.random.binomial(1, transition[cast[i], 1-cast[i]]))
    return cast[1:]


# Problem 3
def four_state_forecast(days):
    """Run a simulation for the weather over the specified number of days,
    with mild as the starting state, using the four-state Markov chain.
    Return a list containing the day-by-day results, not including the
    starting day.

    Examples:
        >>> four_state_forecast(3)
        [0, 1, 3]
        >>> four_state_forecast(5)
        [2, 1, 2, 1, 1]
    """
    #raise NotImplementedError("Problem 3 Incomplete")
    transition = np.array([[0.5,0.3,0.1,0],[0.3,0.3,0.3,0.3],[0.2,0.3,0.4,0.5],
                    [0,0.1,0.2,0.2]])
    cast = [1]
    for i in xrange(days) :
        a = np.random.multinomial(1,transition[:,cast[i]])
        cast.append(np.where(a==1)[0][0])
    return cast[1:]


# Problem 4
def steady_state(A, tol=1e-12, N=40):
    """Compute the steady state of the transition matrix A.

    Inputs:
        A ((n,n) ndarray): A column-stochastic transition matrix.
        tol (float): The convergence tolerance.
        N (int): The maximum number of iterations to compute.

    Raises:
        ValueError: if the iteration does not converge within N steps.

    Returns:
        x ((n,) ndarray): The steady state distribution vector of A.
    """
    #raise NotImplementedError("Problem 4 Incomplete")
    xnew = np.zeros_like(A[:,0])
    xnew[0] = 1
    xold = np.zeros_like(xnew)
    counter = 0
    while not np.allclose(xold,xnew) and counter <= N :
        counter += 1
        xold = xnew
        xnew = np.dot(A,xold)
    if counter <= N :
        return xnew
    else :
        raise ValueError("Nonconvergent transition.")


# Problems 5 and 6
class SentenceGenerator(object):
    """Markov chain creator for simulating bad English.

    Attributes:
        (what attributes do you need to keep track of?)

    Example:
        >>> yoda = SentenceGenerator("Yoda.txt")
        >>> print yoda.babble()
        The dark side of loss is a path as one with you.
    """

    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        #raise NotImplementedError("Problem 5 Incomplete")
        words = ['$tart']
        with open(filename,'r') as training_set :
            lines = training_set.read().split('\n')
            for line in lines :
                sentence = line.split(' ')
                for word in sentence :
                    if word not in words :
                        words.append(word)
        words.append('$top')

        self.words = words
        n = len(words)
        A = np.zeros((n,n))

        i = 0
        j = 0

        with open(filename,'r') as count :
            lines = count.read().split('\n')
            for line in lines :
                j = 0
                sentence = line.split(' ')
                for word in sentence :
                    i = words.index(word)
                    A[i,j] += 1
                    j = i
                A[n-1,j] += 1
            A[n-1,n-1] = 1

        for i in xrange(n) :
            A[:,i] = A[:,i]/float(sum(A[:,i]))

        self.transition = A


    def babble(self):
        """Begin at the start sate and use the strategy from
        four_state_forecast() to transition through the Markov chain.
        Keep track of the path through the chain and the corresponding words.
        When the stop state is reached, stop transitioning and terminate the
        sentence. Return the resulting sentence as a single string.
        """
        #raise NotImplementedError("Problem 6 Incomplete")
        sentence = []
        n = self.transition.shape[0]
        current = 0
        while current != n-1 :
            new = np.random.multinomial(1,self.transition[:,current])
            word_position = np.where(new==1)[0][0]
            sentence.append(self.words[word_position])
            current = word_position
        sentence = sentence[:-1]
        return " ".join(sentence)

if __name__ == "__main__" :
    #print random_markov(5)
    #print forecast(10)
    #print four_state_forecast(5)
    """
    t1 = np.array([[0.7, 0.6], [0.3, 0.4]])
    t2 = np.array([[0.5,0.3,0.1,0],[0.3,0.3,0.3,0.3],[0.2,0.3,0.4,0.5],
                    [0,0.1,0.2,0.2]])
    s1 = steady_state(t1)
    s2 = steady_state(t2)
    n = 50000
    f1 = forecast(n)
    f2 = four_state_forecast(n)
    a1 = np.array([(n-sum(f1)),sum(f1)]) / float(n)
    f2h = [x for x in f2 if x==0]
    f2m = [x for x in f2 if x==1]
    f2c = [x for x in f2 if x==2]
    f2f = [x for x in f2 if x==3]
    a2 = np.array([len(f2h),len(f2m),len(f2c),len(f2f)]) / float(n)
    print s1, " and ", s2
    print a1, " and ", a2
    """
    A = SentenceGenerator('tswift1989.txt')
    for i in xrange(20) :
        print A.babble()
