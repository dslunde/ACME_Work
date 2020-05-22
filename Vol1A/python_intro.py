# Lab1.py
"""
The first Lab Assignment
"""

"""
PROBLEM 7
"""
# Calculates sum of the Alternating Harmonic series using n terms
def alt_harmonic(n) :
	n = int(n)
	sn = sum([pow(-1,i+1) / float(i) for i in range(1,n+1)])
	return sn

"""
PROBLEM 6
"""
# Finds and returns largest palindromic number that is a product of two three digit numbers
def palindrome() :
	x = 999
	y = 999
	z = x*y
	palins = []
	while len(palins) <= 100000 and x > 99 and y > 99:
		if check(z) :
			palins.append(z)
		if x == y :
			x = x-1
			y = 999
		else :
			y = y-1
		z = x*y
	#print len(palins)
	return max(palins)

#Check if number is palindrome
def check(num) :
	num = str(num)
	palindrome = False
	if num == backward(num) :
		palindrome = True
	return palindrome

"""
PROBLEM 5
"""
# Pig Latin Function
def pig_latin(word) :
	word = str(word)
	vowels = { "a" , "e" , "i" , "o" , "u" }
	lword = word.lower()
	if lword[0] in vowels :
		new_word = word + "hay"
	else :
		new_word = word[1:] + word[0] + "ay"
	return new_word

"""
PROBLEM 4
"""
# Creates a list of strings and performs some operations on it.
def list_ops() :
	list = [ "bear" , "ant" , "dog" , "cat" ]
	list.append("eagle")
	list[2] = "fox"
	list.remove(list[1])
	list.sort()
	list.reverse()
	return list

"""
PROBLEM 3
"""
# Reverses the order of a string
def backward(string) :
	string = str(string)
	new_str = string[::-1]
	return new_str

# Returns the first half of a string
def first_half(string) :
	string = str(string)
	end = len(string) / 2
	new_str = string[:end]
	return new_str

"""
PROBLEM 2
"""
# Returns the volume of a sphere of radius 'r'
def sphere_volume(r) :
	v = 3.14159 * 4 / 3 * pow(r,3)
	return v

"""
PROBLEM 1
"""
if __name__ == "__main__" :
	print("Hello, world!")
	volume = sphere_volume(2)
	print volume
	str1 = first_half("hello")
	str2 = backward("hello")
	print(str1)
	print(str2)
	my_list = list_ops()
	print (my_list[0:])
	print pig_latin("Hello") , pig_latin("anybody") , pig_latin("hear") , pig_latin("ABBA")
	p = palindrome()
	print(p)
