# simplex.py
"""Volume 2B: Simplex.
    <Name>
    <Class>
    <Date>
    
    Problems 1-6 give instructions on how to build the SimplexSolver class.
    The grader will test your class by solving various linear optimization
    problems and will only call the constructor and the solve() methods directly.
    Write good docstrings for each of your class methods and comment your code.
    
    prob7() will also be tested directly.
    """

import numpy as np

# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem
        
        maximize        c^Tx
        subject to      Ax <= b
        x >= 0
        via the Simplex algorithm.
        """
    
    def __init__(self, c, A, b):
        """
            
            Parameters:
            c (1xn ndarray): The coefficients of the linear objective function.
            A (mxn ndarray): The constraint coefficients matrix.
            b (1xm ndarray): The constraint vector.
            
            Raises:
            ValueError: if the given system is infeasible at the origin.
            """
    #return NotImplementedError("SimplexSolver Not Started")
        if not (abs(b)==b).all() :
            raise ValueError("Origin infeasible")
        self.n = c.shape[0]
        self.m = b.shape[0]
        self.L = [x for x in xrange(self.n,self.n+self.m)]
        for x in xrange(self.n) :
            self.L.append(x)
        self.createTableu(c,A,b)
        self.pivot = [0,0]
        self.findPivot()

    def createTableu(self,c,A,b) :
        Ahat = np.hstack((A,np.eye(self.m)))
        chat = np.hstack((-c,np.zeros(self.m)))
        T1 = np.hstack((0.,chat.T.astype(float),1.))
        T2 = np.hstack((b.reshape(-1,1),Ahat.astype(float),np.zeros_like(b.reshape(-1,1))))
        self.T = np.vstack((T1.astype(float),T2.astype(float)))
    
    def findPivot(self) :
        self.foundPivot = False
        for i in xrange(1,self.T.shape[1]) :
            if self.T[0][i] < 0 :
                self.pivot[1] = i
                self.foundPivot = True
                break;
        if self.foundPivot :
            self.findExit()
        else :
            self.pivot = [0,0]

    
    def findExit(self) :
        #print self.pivot
        ratios = [float('inf')]
        for i in xrange(1,self.T.shape[0]) :
            if self.T[i][self.pivot[1]] > 0 :
                ratios.append(self.T[i][0]/self.T[i][self.pivot[1]])
            else :
                ratios.append(float('inf'))
                #print ratios
        i = ratios.index(min(ratios))
        ratios = [ratios[x]==ratios[i] for x in xrange(len(ratios))]
        ratios_mod = ratios
        ratios_mod[i] = False
        if True in ratios_mod :
            self.BlandsRule(ratios,i)
        else :
            self.pivot[0] = i
        #print self.pivot

    def BlandsRule(self,rats,x) :
        min = self.L[x]
        for i in xrange(x+1,len(rats)) :
            if rats[i] == True :
                if self.L[i] < self.L[min] :
                    rats[min] = False
                    min = i
                else :
                    rats[i] = False
        self.pivot[0] = rats.index(True) + 1

    def Pivot(self) :
    #print "Pivot begins.\n"
    #Check for unboundedness
        row = self.pivot[0]
        col = self.pivot[1]
        lrow = row-1
        lcol = (col+self.n)%(self.n+self.m)
        if lcol == 0 :
            lcol = self.n+self.m
        self.L[lrow],self.L[lcol] = self.L[lcol],self.L[lrow]
        self.T[row][:] = self.T[row,:]/self.T[row][col]
        for i in xrange(self.T.shape[0]) :
            if i != row :
                self.T[i][:] = self.T[i][:] - self.T[i][col]*self.T[row][:]


    def solve(self):
        """Solve the linear optimization problem.
            
            Returns:
            (float) The maximum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
            """
#raise NotImplementedError("SimplexSolver Incomplete")
        iters = 0
        while self.foundPivot and iters < 5000:
            self.Pivot()
            self.findPivot()
            iters += 1
        basic = {}
        nonbasic = {}
        for i in xrange(self.n+self.m) :
            if i < self.m :
                basic[self.L[i]] = self.T[i+1][0]
            else :
                nonbasic[self.L[i]] = 0
        return (self.T[0][0],basic,nonbasic)


# Problem 7
def prob7(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.
        
        Parameters:
        filename (str): the path to the data file.
        
        Returns:
        The minimizer of the problem (as an array).
        """
#raise NotImplementedError("Problem 7 Incomplete")
    dict = np.load(filename)
    A = np.vstack((dict['A'],np.eye(dict['A'].shape[1])))
    b = np.hstack((dict['m'],dict['d']))
    ss = SimplexSolver(dict['p'],A,b)
    x = ss.solve()
    sol = np.zeros((dict['A'].shape[1],1))
    for a in xrange(dict['A'].shape[1]) :
        if a in x[1].keys() :
            sol[a] = x[1][a]
    return sol
