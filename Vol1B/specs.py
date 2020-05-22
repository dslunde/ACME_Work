# specs.py
"""Volume IB: Testing.
Darren Lund
01/10/17
"""
import math

# Problem 1 Write unit tests for addition().
# Be sure to install pytest-cov in order to see your code coverage change.
def addition(a,b):
    return a + b

def smallest_factor(n):
    """Finds the smallest prime factor of a number.
    Assume n is a positive integer.
    """
    if n == 1:
        return 1
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return i
    return n


# Problem 2 Write unit tests for operator().
def operator(a, b, oper):
    if type(oper) != str:
        raise ValueError("Oper should be a string")
    if len(oper) != 1:
        raise ValueError("Oper should be one character")
    if oper == "+":
        return a+b
    if oper == "/":
        if b == 0:
            raise ValueError("You can't divide by zero!")
        return a/float(b)
    if oper == "-":
        return a-b
    if oper == "*":
        return a*b
    else:
        raise ValueError("Oper can only be: '+', '/', '-', or '*'")

# Problem 3 Write unit test for this class.
class ComplexNumber(object):
    def __init__(self, real=0, imag=0):
        self.real = real
        self.imag = imag

    def conjugate(self):
        return ComplexNumber(self.real, -self.imag)

    def norm(self):
        return math.sqrt(self.real**2 + self.imag**2)

    def __add__(self, other):
        real = self.real + other.real
        imag = self.imag + other.imag
        return ComplexNumber(real, imag)

    def __sub__(self, other):
        real = self.real - other.real
        imag = self.imag - other.imag
        return ComplexNumber(real, imag)

    def __mul__(self, other):
        real = self.real*other.real - self.imag*other.imag
        imag = self.imag*other.real + other.imag*self.real
        return ComplexNumber(real, imag)

    def __div__(self, other):
        if other.real == 0 and other.imag == 0:
            raise ValueError("Cannot divide by zero")
        bottom = (other.conjugate()*other*1.).real
        top = self*other.conjugate()
        return ComplexNumber(top.real / bottom, top.imag / bottom)

    def __eq__(self, other):
        return self.imag == other.imag and self.real == other.real

    def __str__(self):
        return "{}{}{}i".format(self.real, '+' if self.imag >= 0 else '-',
                                                                abs(self.imag))

# Problem 5: Write code for the Set game here
def Set(filename) :
    valid = ['0','1','2']
    with open(filename,'r') as hand :
        cards = hand.read().split('\n')

    if len(cards) != 12 :
        raise ValueError("Wrong number of cards.")

    for c in cards :
        if len(c) != 4:
            raise ValueError("Wrong number of digits.")
        else :
            for digit in c :
                if digit not in valid :
                    raise ValueError("Invalid card value.")

    cards.sort()
    for i in xrange(11) :
        if cards[i] == cards[i+1] :
            raise ValueError("Duplicate card found.")

    sets = []
    for i in xrange(12) :
        for j in xrange(i+1,12) :
            for k in xrange(j+1,12) :
                if test(int(cards[i])+int(cards[j])+int(cards[k])) :
                    sets.append([cards[i],cards[j],cards[k]])
    return len(sets)

def test(value) :
    for d in str(value) :
        if int(d) % 3 != 0 :
            return False
    return True
