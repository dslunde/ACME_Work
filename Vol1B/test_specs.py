# test_specs.py
"""Volume 1B: Testing.
Darren Lund
Math 323
01/10/17
"""

import math
import specs
import pytest

# Problem 1: Test the addition and fibonacci functions from specs.py
def test_addition():
    assert specs.addition(2,3) == 5, "Positive addition failed."
    assert specs.addition(999999999,1) == 1000000000, "Large addition failed."
    assert specs.addition(23,0) == 23, "Addition by zero failed."
    assert specs.addition(-5,-3) == -8, "Negative addition failed."
    assert specs.addition(4,-3) == 1

def test_smallest_factor():
    assert specs.smallest_factor(25) == 5
    assert specs.smallest_factor(1) == 1
    assert specs.smallest_factor(17) == 17
    assert specs.smallest_factor(34) == 2

# Problem 2: Test the operator function from specs.py
def test_operator():
    assert specs.operator(4,0,'+') == 4, "Non-negative addition failed."
    assert specs.operator(-3,-2,'+') == -5, "Strictly negative addition failed."
    assert specs.operator(-2,5,'+') == 3
    assert specs.operator(4,0,'-') == 4, "Non-negative subtratction failed."
    assert specs.operator(-3,-2,'-') == -1, "Strictly negative subtraction failed."
    assert specs.operator(-2,5,'-') == -7
    assert specs.operator(5,-2,'-') == 7
    assert specs.operator(4,0,'*') == 0, "Multiplication by zero failed."
    assert specs.operator(-3,-2,'*') == 6, "Negative multiplication failed."
    assert specs.operator(-2,5,'*') == -10
    assert specs.operator(-6,-2,'/') == 3, "Negative division failed."
    assert specs.operator(2,5,'/') == 0.4, "Float division failed."
    assert specs.operator(5,1,'/') == 5, "Division by 1 failed."
    assert specs.operator(2,0.5,'/') == 4, "Division by fraction failed."

    with pytest.raises(Exception) as error :
        specs.operator(4,0,1)
    assert error.typename == "ValueError"
    assert error.value.args[0] == "Oper should be a string"

    with pytest.raises(Exception) as error :
        specs.operator(4,0,'hi')
    assert error.typename == "ValueError"
    assert error.value.args[0] == "Oper should be one character"

    with pytest.raises(Exception) as error :
        specs.operator(4,0,'P')
    assert error.typename == "ValueError"
    assert error.value.args[0] == "Oper can only be: '+', '/', '-', or '*'"

    with pytest.raises(Exception) as error :
        specs.operator(4,0,'/')
    assert error.typename == "ValueError"
    assert error.value.args[0] == "You can't divide by zero!"

# Problem 3: Finish testing the complex number class
@pytest.fixture
def set_up_complex_nums():
    number_1 = specs.ComplexNumber(1, 2)
    number_2 = specs.ComplexNumber(5, 5)
    number_3 = specs.ComplexNumber(2, 9)
    return number_1, number_2, number_3

def test_complex_addition(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1 + number_2 == specs.ComplexNumber(6, 7)
    assert number_1 + number_3 == specs.ComplexNumber(3, 11)
    assert number_2 + number_3 == specs.ComplexNumber(7, 14)
    assert number_3 + number_3 == specs.ComplexNumber(4, 18)

def test_complex_subtraction(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1 - number_2 == specs.ComplexNumber(-4, -3)
    assert number_1 - number_3 == specs.ComplexNumber(-1, -7)
    assert number_2 - number_3 == specs.ComplexNumber(3, -4)
    assert number_3 - number_3 == specs.ComplexNumber(0, 0)

def test_complex_multiplication(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1 * number_2 == specs.ComplexNumber(-5, 15)
    assert number_1 * number_3 == specs.ComplexNumber(-16, 13)
    assert number_2 * number_3 == specs.ComplexNumber(-35, 55)
    assert number_3 * number_3 == specs.ComplexNumber(-77, 36)

def test_complex_division(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1 / number_2 == specs.ComplexNumber(.3, .1)
    assert number_1 / number_3 == specs.ComplexNumber(4./17, -1./17)
    assert number_2 / number_3 == specs.ComplexNumber(11./17, -7./17)
    assert number_3 / number_3 == specs.ComplexNumber(1, 0)

    with pytest.raises(Exception) as error :
        specs.ComplexNumber(1,2) / specs.ComplexNumber(0,0)
    assert error.typename == "ValueError"
    assert error.value.args[0] == "Cannot divide by zero"

def test_complex_equals(set_up_complex_nums) :
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1 == specs.ComplexNumber(1,2)
    assert number_2 == specs.ComplexNumber(5,5)
    assert number_3 == specs.ComplexNumber(2,9)

def test_complex_norm(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1.norm() == math.sqrt(5)
    assert number_2.norm() == math.sqrt(50)
    assert number_3.norm() == math.sqrt(85)

def test_complex_conjugate(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1.conjugate() == specs.ComplexNumber(1,-2)
    assert number_2.conjugate() == specs.ComplexNumber(5,-5)
    assert number_3.conjugate() == specs.ComplexNumber(2,-9)

def test_complex_string(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert str(number_1) == "1+2i"
    assert str(number_2) == "5+5i"
    assert str(number_3) == "2+9i"
    assert str(specs.ComplexNumber(1,-5)) == "1-5i"

# Problem 4: Write test cases for the Set game.
def test_set_hands() :
    bad_hands = ['hand1.txt','hand2.txt','hand3.txt','hand4.txt','hand5.txt']
    error_messages = ['Wrong number of cards.','Wrong number of cards.',
                        'Wrong number of digits.','Invalid card value.',
                        'Duplicate card found.']
    for x in xrange(len(bad_hands)) :
        with pytest.raises(Exception) as error :
            specs.Set('hands/'+bad_hands[x])
        assert error.typename == "ValueError", "Hand "+str(x+1)+" wasn't a value error."
        assert error.value.args[0] == error_messages[x], "Hand "+str(x+1)+" had the wrong error message."

        assert specs.Set('hands/hand6.txt') == 3, "Incorrect number of sets."
