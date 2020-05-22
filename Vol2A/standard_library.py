# standard_library.py
"""
Introductory Labs: The Standard Library.
Darren Lund
AlgDes&Opt Lab
9/1/16
"""

import calculator as calc
import sys
import random as r

# Problem 1
def prob1(l) :
    """
    Accept a list 'l' of numbers as input and return a new list with the
    minimum, maximum, and average of the contents of 'l'.
    """
    return [min(l) , max(l) , float(sum(l)) / len(l) ]

# Problem 2
def prob2() :
    """
    Programmatically determine which Python objects are mutable and which
    are immutable.  Test numbers, strings, lists, tuples, and dictionaries
    Print your results to the terminal.
    """
    stuff = [ 1 , "hi" , ['a','b','c'] , ('a',2) , {1 : 'x' , 2 : 'b'} ]
    mutable = []
    for i in range(len(stuff)) :
        type1 = stuff[i]
        type2 = type1
        if i == 0 :
            type2 += 1
        elif i == 1 :
            type2 += 'a'
        elif i == 2 :
            type2.append(1)
        elif i == 3 :
            type2 += (1,)
        elif i == 4 :
            type2[1] = 'a'
        mutable.append(type1 == type2)
    #print "Mutable:"
    for i in range(len(stuff)) :
        if mutable[i] == True :
            string = "Mutable"
        else :
            string = "Immutable"
        print type(stuff[i]) , string

# Problem 3
def prob3(x,y) :
    """
    Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any methods other than those that are imported from your
    'calculator' module.

    Parameters:
        a (float): the length of one of the sides of the triangle.
        b (float): the length of the other nonhypotenuse side of the triangle.

    Returns:
        The length of the triangle's hypotenuse.
    """
    return calc.root2(calc.add(calc.prod(x,x),calc.prod(y,y)))

# Problem 4: Implement shut the box

# Main game code
def shut_the_box() :
	if len(sys.argv) < 2 :
		name = raw_input("Player Name: ")
	else :
		name = sys.argv[1]
	left = range(1,10)
	dice = range(1,7)
	game_over = False
	while not game_over and len(left) > 0 :
		if sum(left) > 6 :
			roll = calc.add(r.choice(dice),r.choice(dice))
		else :
			roll = r.choice(dice)
		print "\nNumbers left:" , left
		print "Roll:" , roll
		if not doable(roll,left) :
			game_over = True
			print "Game over!"
		else :
			elim = ""
			good = False
			while not good :
				elim = raw_input("Numbers to eliminate: ")
				if not check_in(elim,left,roll) :
					print "Invalid input"
				else :
					good = True
			for i in range(len(elim)) :
				if elim[i] != ' ' :
					left.remove(int(elim[i]))
	print "\nScore for player %s: %d points" % (name,sum(left))
	if len(left) == 0 :
		print "Congratulations! You shut the box!"

# Check if roll is playable (recursive)
def doable(roll,left) :
	if len(left) == 0 :
		able = False
	elif roll in left :
		able = True
	elif roll < left[0] or roll > sum(left) :
		able = False
	else :
		index = len(left)-1
		while left[index] > roll :
			index -= 1
		smaller_than_roll = left[:index+1]
		for i in range(len(smaller_than_roll)) :
			without_i = smaller_than_roll[i+1:]
			if roll - smaller_than_roll[i] in without_i :
				able = True
				break
			else :
				able = doable(roll-smaller_than_roll[i],without_i)
				if able == True :
					break
	return able

# Check is input is acceptable
def check_in(elim,left,roll) :
	good = True
	sum = 0
	if len(elim) % 2 == 0 :
		good = False
	else :
		for i in range(len(elim)) :
			if i % 2 == 1 and elim[i] != ' ' :
				good = False
				break
			elif i % 2 == 0 and (not is_int(elim[i]) or int(elim[i]) not in left) :
				good = False
				break
			elif i % 2 == 0 :
				sum += int(elim[i])
	if sum != roll :
		good = False
	return good

# Helper function to check if something's an int
def is_int(x) :
	try :
		int(x)
		good = True
	except ValueError :
		good = False
	return good
# End code for "Shut the Box"

if __name__ == "__main__" :
    #print(prob1([1,2,3,4,5,6,7,8,9]))
    #prob2()
    shut_the_box()
