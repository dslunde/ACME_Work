# exceptions.py
"""Introductory Labs: Exceptions and File I/O.
Darren Lund
Math 321
9/15/16
"""

from random import choice

# Problem 1
def arithmagic():
    step_1 = raw_input("Enter a 3-digit number where the first and last "
                                            "digits differ by 2 or more: ")
    if len(step_1) != 3 :
        raise ValueError("That input is not a 3 digit number!")
    elif step_1.isdigit() == False :
        raise TypeError("Input is not an integer!")
    elif abs(int(step_1[0]) - int(step_1[2])) < 2 :
        raise ValueError("Difference between first and last digits is too small!")
    step_2 = raw_input("Enter the reverse of the first number, obtained "
                                            "by reading it backwards: ")
    if step_1[::-1] != step_2 :
        raise ValueError("That's not the reverse of your first number!")
    step_3 = raw_input("Enter the positive difference of these numbers: ")
    if step_3[0] == '-' :
        raise ValueError("Entry is negative. It must be positive.")
    elif step_3.isdigit() == False :
        raise TypeError("Input is not an integer!")
    elif int(step_3) != abs(int(step_1) - int(step_2)) :
        raise ValueError("Your math is wrong.")
    step_4 = raw_input("Enter the reverse of the previous result: ")
    if step_3[::-1] != step_4 :
        raise ValueError("That's not the reverse of the result you got.")
    print str(step_3) + " + " + str(step_4) + " = 1089 (ta-da!)"


# Problem 2
def random_walk(max_iters=1e12):
    try :
        walk = 0
        direction = [-1, 1]
        q = 0
        for i in xrange(int(max_iters)):
            q = i
            walk += choice(direction)
    except KeyboardInterrupt:
        print "\nProcess interrupted at iteration" , q
    else :
        print "Process complete."
        return walk


# Problems 3 and 4: Write a 'ContentFilter' class.
class ContentFilter(object):
    """
    A class for file objects.  Allows reading of files and writing to
        new files.

    Attributes:
        name (str) = The name of the file
        contents (str) = The contents of the file as a single string.
    """
    def __init__(self,name):
        if not isinstance(name,str):
            raise TypeError("Invalid file name.")
        with open(name,'r') as new_file:
            contents = new_file.read()
        self.name = name
        self.contents = contents

    def uniform(self,new_file,mode='w',case="upper"):
        """
        Writes or appends to a new document this file where all case-based
            characters are upper-cased, or lower-cased.

        Inputs :
            new_file (str) : Writes to or appends on to this file.  If file
                doesn't exist, a new file will be created with this name.
            mode (str) : Either 'w' for write, or 'a' for append.
            case (str) : Either 'upper' for all upper-cased, or 'lower' for
                all lower-cased.
        """
        if case != "upper" and case != "lower":
            raise ValueError("Case must be either 'upper' (default) or 'lower'.")
        elif mode != 'w' and mode != 'a':
            raise ValueError("Modes available are write ('w') (default) and append ('a')")

        with open(new_file,mode) as out_file:
            if case == "lower":
                out_file.write(self.contents.lower())
            else :
                out_file.write(self.contents.upper())

    def reverse(self,new_file,mode='w',unit="line"):
        """
        Writes or appends to a new document this file where either the lines
            have been reversed, or the words in each line have been reversed.

        Inputs :
            new_file (str) : Writes to or appends on to this file.  If file
                doesn't exist, a new file will be created with this name.
            mode (str) : Either 'w' for write, or 'a' for append.
            unit (str) : Either 'lline' for reversing line order, or 'word'
                for reversing word order in each line.
        """
        if unit != "line" and unit != "word":
            raise ValueError("You can only reverse by 'line' or 'word'.")
        elif mode != 'w' and mode != 'a':
            raise ValueError("Modes available are write ('w') (default) and append ('a')")

        A = self.contents.split('\n')
        if unit == "word":
            for i in xrange(len(A)):
                B = A[i].split(' ')
                B = B[::-1]
                A[i] = ' '.join(B)
        else :
            A = A[::-1]

        with open(new_file,mode) as out_file:
            out_file.write('\n'.join(A))

    def transpose(self,new_file,mode='w'):
        """
        Writes or appends to a new document this file where the first word
            of each line makes up the new first line, the second word of each
            line the new second line, and so on and so forth.

        Inputs :
            new_file (str) : Writes to or appends on to this file.  If file
                doesn't exist, a new file will be created with this name.
            mode (str) : Either 'w' for write, or 'a' for append.
        """
        if mode != 'w' and mode != 'a':
            raise ValueError("Modes available are write ('w') (default) and append ('a')")
        #if len(self.contents) != 0 :
        A = self.contents.split('\n')
        if A[len(A)-1] == [''] :
            A = A[:-1]
        for i in xrange(len(A)):
            B = A[i].split(' ')
            A[i] = B

        m = len(A)
        n = len(A[0])
        #print m , n
        list1 = A[0]
        C = []

        for i in xrange(n):
            C.append([list1[i]])

        #print C
        for i in xrange(1,m):
            list1 = A[i]
            #print list1
            for j in xrange(len(C)):
                C[j].append(list1[j])

        for i in xrange(len(C)):
            C[i] = ' '.join(C[i])

        with open(new_file,mode) as out_file:
            out_file.write('\n'.join(C))

    # ----- Magic Methods ------------------------------------

    def __str__(self) :
        """Returns a string representation of a ContentFilter of the form:
        Source File:            <name>
        Total Characters:       <number of characters in file>
        Alphabetic Characters:  <number of alphabetic characters>
        Numerical Characters:   <number of numerical characters>
        Whitespace Characters:  <number of new lines, tabs, and spaces>
        Number of Lines:        <number of lines>
        """
        num_characters = len(self.contents)
        alpha = 0
        num = 0
        white_space = 0
        nline = 1
        for i in xrange(num_characters):
            a = self.contents[i]
            if a == '\n':
                nline += 1
                white_space += 1
            elif a.isspace() :
                white_space += 1
            elif a.isalpha() :
                alpha += 1
            elif a.isdigit() :
                num += 1

        result1 = "Source file:\t\t" + self.name + "\n"
        result2 = "Total Characters:\t" + str(num_characters) + "\n"
        result3 = "Alphabetic Characters:\t" + str(alpha) + "\n"
        result4 = "Numerical Characters:\t" + str(num) + "\n"
        result5 = "Whitespace Characters:\t" + str(white_space) + "\n"
        result6 = "Number of Lines:\t" + str(nline)
        return result1 + result2 + result3 + result4 + result5 + result6

if __name__ == "__main__" :
    arithmagic()
    print random_walk(1)
