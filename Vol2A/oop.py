# oop.py
"""Introductory Labs: Object Oriented Programming.
<Name>
<Class>
<Date>
"""

import math

class Backpack(object):
    """A Backpack object class. Has a name and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
        color (str): color of the backpack
        max_size (int): maximum number of items you can fit in the
            backpack
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size=5):
        """Set the name and initialize an empty contents list.

        Inputs:
            name (str): the name of the backpack's owner.
            color (str): the color of the backpack
            max_size (int) (optional): maximum capacity (default=5)

        Returns:
            A Backpack object wth no contents.
        """
        self.name = name
        self.contents = []
        self.color = color
        self.max_size = max_size

    def put(self, item):
        """Add 'item' to the backpack's list of contents."""
        if len(self.contents) < self.max_size :
            self.contents.append(item)
        else :
            print "No Room!"

    def take(self, item):
        """Remove 'item' from the backpack's list of contents."""
        self.contents.remove(item)

    def dump(self):
        self.contents = []

    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)

    def __eq__(self,other):
        """Compare two backpacks.  If 'self' has the same name,
        color, and number of objects as 'other', return True.
        Otherwise, return False.
        """
        equal = self.name == other.name and self.color == other.color
        return equal and len(self.contents) == len(other.contents)

    def __str__(self):
        """Returns a string representation of a backpack of the form:
        Owner:      <name>
        Color:      <color>
        Size:       <number of items in contents>
        Max Size:   <max_size>
        Contents:   [<item1>, <item2>, ... ]
        """
        first = "Owner:\t\t" + self.name + "\n"
        second = "Color:\t\t" + self.color + "\n"
        third = "Size:\t\t" + str(len(self.contents)) + "\n"
        fourth = "Max Size:\t" + str(self.max_size) + "\n"
        fifth = "Contents:\t" + str(self.contents)
        return first + second + third + fourth + fifth

    def __ne__(self,other):
        """See '__eq__' for the requirements of two backpack to be equal.
        """
        return not self == other


# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit
            in the knapsack.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color, max_size=3):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default
        instead of 5.

        Inputs:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit
                in the knapsack. Defaults to 3.

        Returns:
            A Knapsack object with no contents.
        """
        Backpack.__init__(self, name, color, max_size)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    """
    A Jet Pack is a Backpack of default max_size = 2, and a fuel element
    (default of 10)

    Attributes:
        name (str): the name of the owner
        color (str): color of the jetpack
        max_size (int): maximum number of items we can store in the jetpack
        contents (list): items stored in the jetpack
        fuel (int): how much fuel is in the jetpack
    """
    def __init__(self, name, color, max_size=2, fuel=10):
        """Use the Backpack initializer with max_size of 2.
        Set new fuel attribute.

        Inputs:
            name (str): the name of the owner
            color (str): color of the jetpack
            max_size (int): maximum number of items we can store in the jetpack

        Returns:
            A Jetpack object with no contents.
        """
        Backpack.__init__(self,name,color,max_size)
        self.fuel = fuel

    def fly(self, burn):
        """A method to fly the jetpack.  Fuel is decremented
        by the amount burned.  If they try to burn more fuel than
        they have, no fuel is burned.

        Inputs:
            burn (int): the amount of fuel burned in flight.
        """
        if burn > self.fuel :
            print "Not enough fuel!"
        else :
            self.fuel -= burn

    def dump(self):
        """Uses the Backpack dump method, then resets fuel to 0
        """
        Backpack.dump(self)
        self.fuel = 0

# Problem 4: Write a 'ComplexNumber' class.
class ComplexNumber(object):
    """A class designed to replicate complex numbers.

    Attributes:
        real (int): the real part of the complex number
        imag (int): the imaginary part of the complex number
    """
    def __init__(self,real=0,imag=0):
        """Initialize the complex number.  If no inputs were provided,
        the complex number initialized is 0.

        Returns:
            A ComplexNumber object
        """
        self.real = real
        self.imag = imag

    def conjugate(self):
        """Creats a new ComplexNumber object that is the conjugate of
        the object provide.  I.E. if 'self' = a + bi, then it returns
        the new object a - bi.

        Returns:
            A ComplexNumber equal to the conjugate of self.
        """
        return ComplexNumber(self.real,-self.imag)

    # Magic Functions -----------------------------------------------

    def __abs__(self):
        """Gets the length of a ComplexNumber object.
        If 'self' = a + bi, it should return sqrt(a^{2}+b^{2})

        Returns:
            An integer the length of the complex number
        """
        return math.sqrt(self.real**2+self.imag**2)

    def __lt__(self,other):
        """Determines if one complex number is less than another.
        Uses __abs__(self) to dtermine length of complex numbers.

        Returns:
            A boolean, True if 'self' < 'other', False otherwise.
        """
        return self.__abs__() < other.__abs__()

    def __gt__(self,other):
        """Determines if one complex number is greater than another.
        Uses __abs__(self) to dtermine length of complex numbers.

        Returns:
            Boolean; True if 'self' > 'other', False otherwise.
        """
        return self.__abs__() > other.__abs__()

    def __eq__(self,other):
        """Determines if two complex numbers are equal.
        This is only true if they have the same real and imaginary parts.

        Returns:
            Boolean; True if the two numbers are identical, False otherwise.
        """
        return (self.real == other.real) and (self.imag == other.imag)

    def __ne__(self,other):
        """Determines if two complex numbers are not equal.
        Calls the __eq__(self,other) function and returns the opposite.

        Returns:
            Boolean; True if the two numbers are different, False otherwise.
        """
        return not self == other

    def __add__(self,other):
        """Adds two complex numbers by adding their real and imaginary
        components.

        Returns:
            A new ComplexNumber object equal to the sum of the two provided.
        """
        return ComplexNumber(self.real+other.real,self.imag+other.imag)

    def __sub__(self,other):
        """Subtracts two complex numbers by subtracting their real and imaginary
        components.

        Returns:
            A new ComplexNumber object equal to the difference of the two provided.
        """
        return ComplexNumber(self.real-other.real,self.imag-other.imag)

    def __mul__(self,other):
        """Multiplies two complex numbers by FOILing their components.

        Returns:
            A new ComplexNumber object equal to the product of the two provided.
        """
        real = self.real*other.real-self.imag*other.imag
        imag = self.real*other.imag+self.imag*other.real
        return ComplexNumber(real,imag)

    def __div__(self,other):
        """Divides two complex numbers by multipyling by the conjugate
        of the denominator to the numerator and denominator and dividing the
        components as needed.

        Returns:
            A new ComplexNumber object.
        """
        divisor = other*other.conjugate()
        denominator = float(divisor.real) #Since divisor should have no imaginary part
        numerator = self*other.conjugate()
        return ComplexNumber(numerator.real/denominator,numerator.imag/denominator)

    def __str__(self):
        """Turns a CompleNumber object into its written representation
        a+bi

        Returns:
            A ComplexNumber object as a string.
        """
        if self.imag > 0:
            string = str(self.real) + "+" + str(self.imag) + "i"
        elif self.imag < 0:
            string = str(self.real) + "-" + str(-self.imag) + "i"
        else:
            string = str(self.real)
        return string
