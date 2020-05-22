# linked_lists.py
"""Volume II Lab 4: Data Structures 1 (Linked Lists)
Darren Lund
Math 321
9/22/16
"""


# Problem 1
class Node(object):
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store 'data' in the 'value' attribute."""
        if not isinstance(data,(int,str,float,long)) :
            raise TypeError("This list is typist. It only accepts: int, str, float, and long.")
        else :
            self.value = data


class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the 'Node' class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store 'data' in the 'value' attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None
        self.prev = None


class LinkedList(object):
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the 'head' and 'tail' attributes by setting
        them to 'None', since the list is empty initially.
        Initialize size to 0 for the same reasons.
        """
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, data):
        """Append a new node containing 'data' to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node

        # Increase size of list
        self.size += 1

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing 'data'.
        If no such node exists, raise a ValueError.

        Examples:
            >>> l = LinkedList()
            >>> for i in [1,3,5,7,9]:
            ...     l.append(i)
            ...
            >>> node = l.find(5)
            >>> node.value
            5
            >>> l.find(10)
            ValueError: <message>
        """
        current = self.head
        while current != None and current.value != data :
            current = current.next
        if current == None :
            raise ValueError("Element not in list.")
        else :
            return current
        #raise NotImplementedError("Problem 2 Incomplete")

    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in [1,3,5]:
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        return self.size
        #raise NotImplementedError("Problem 3 Incomplete")

    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()   |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:   |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)    |   ...     l2.append(i)
            ...                     |   ...
            >>> print(l1)           |   >>> print(l2)
            [1, 3, 5]               |   ['a', 'b', 'c']
        """
        current = self.head
        result = "["
        while current != None :
            if isinstance(current.value,str) :
                result += "'" + current.value + "'"
            else :
                result += str(current.value)
            if current.next != None:
                result += ", "
            current = current.next
        result += "]"
        return result
        #raise NotImplementedError("Problem 3 Incomplete")

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing 'data'. Return nothing.

        Raises:
            ValueError: if the list is empty, or does not contain 'data'.

        Examples:
            >>> print(l1)       |   >>> print(l2)
            [1, 3, 5, 7, 9]     |   [2, 4, 6, 8]
            >>> l1.remove(5)    |   >>> l2.remove(10)
            >>> l1.remove(1)    |   ValueError: <message>
            >>> l1.remove(9)    |   >>> l3 = LinkedList()
            >>> print(l1)       |   >>> l3.remove(10)
            [3, 7]              |   ValueError: <message>
        """
        if self.size == 0 :
            raise ValueError("List is empty!")
        elif self.size == 1 :
            self.head = None
            self.size = 0
        else :
            current = self.head
            if current.value == data :
                self.head = current.next
                self.size -= 1
            elif self.tail.value == data :
                self.tail = self.tail.prev
                self.tail.next = None
                self.size -= 1
            else :
                while current != None :
                    if current.next != None and current.next.value == data :
                        current.next = current.next.next
                        current.next.next.prev = current
                        self.size -= 1
                        break;
                    else :
                        current = current.next
                if current == None :
                    raise ValueError("Element not in list!")
        #raise NotImplementedError("Problem 4 Incomplete")

    # Problem 5
    def insert(self, data, place):
        """Insert a node containing 'data' immediately before the first node
        in the list containing 'place'. Return nothing.

        Raises:
            ValueError: if the list is empty, or does not contain 'place'.

        Examples:
            >>> print(l1)           |   >>> print(l1)
            [1, 3, 7]               |   [1, 3, 5, 7, 7]
            >>> l1.insert(7,7)      |   >>> l1.insert(3, 2)
            >>> print(l1)           |   ValueError: <message>
            [1, 3, 7, 7]            |
            >>> l1.insert(5,7)      |   >>> l2 = LinkedList()
            >>> print(l1)           |   >>> l2.insert(10,10)
            [1, 3, 5, 7, 7]         |   ValueError: <message>
        """
        if self.size == 0 :
            raise ValueError("List is empty.  Just add it, nitwit.")
        else :
            current = self.head
            if current.value == place :
                self.head = LinkedListNode(data)
                self.head.next = current
                current.prev = self.head
                self.size += 1
            elif self.tail.value == place :
                new_node = LinkedListNode(data)
                new_node.next = self.tail
                new_node.prev = self.tail.prev
                new_node.prev.next = new_node
                new_node.next.prev = new_node
                self.size += 1
            else :
                while current != None and current.next != None:
                    if current.next.value == place :
                        new_node = LinkedListNode(data)
                        new_node.next = current.next
                        new_node.prev = current
                        new_node.prev.next = new_node
                        new_node.next.prev = new_node
                        self.size += 1
                        break;
                    else :
                        current = current.next
                if current.next == self.tail :
                    raise ValueError("The node you want to place before "
                                        "doesn't exist")
        #raise NotImplementedError("Problem 5 Incomplete")

# Problem 6: Write a Deque class.
class Deque(LinkedList) :
    """A node based class that implements a Deque, and inherits from the
    previously defined LinkedList class.  Functions are added or disabled
    from the LinkedList class to insure that it behaves like a deque.
    """
    def __init__(self) :
        """Initializes the deque.  Simply calls the __init__ function for
        LinkedList.
        """
        LinkedList.__init__(self)

    def remove(*args,**kwargs) :
        """Disables the LinkedList remove function.
        """
        raise NotImplementedError("Use 'pop()' or 'popleft()' for removal.")

    def insert(*args,**kwargs) :
        """Disables the LinkedList insert function.
        """
        raise NotImplementedError("Use 'append()' and 'appendleft()' to "
                                    "add elements.")

    def pop(self) :
        """Removes the right most (tail) node.
        """
        element = self.tail.value
        LinkedList.remove(self,element)
        return element

    def popleft(self) :
        """Removes the left most (head) node.
        """
        element = self.head.value
        LinkedList.remove(self,element)
        return element

    def appendleft(self,data) :
        """Appends data to the left side (head) of the deque.
        """
        LinkedList.insert(self,data,self.head.value)

# Problem 7
def prob7(infile, outfile):
    """Reverse the file 'infile' by line and write the results to 'outfile'."""
    with open(infile,'r') as new_file:
        contents = new_file.read()

    A = contents.split('\n')
    deq = Deque()

    for i in A :
        deq.append(i)

    with open(outfile,'w') as out_file:
        for i in xrange(len(A)-1,-1,-1) :
            out_file.write(deq.pop())
            if i != -1 :
                out_file.write('\n')
    #raise NotImplementedError("Problem 7 Incomplete")

if __name__ == "__main__" :
    #print "Hello"
    prob7("english.txt","bad_english.txt")
