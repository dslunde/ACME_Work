# regular_expressions.py
"""Volume 3: Regular Expressions.
Darren Lund
Becoming Batman
Now
"""

import re

# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #raise NotImplementedError("Problem 1 Incomplete")
    return re.compile("python")

def test_prob1() :
    pos = [ 'python is great' , 'i <3 python' , 'anacondapythonstuff' , 'pypypythonthonthon']
    pattern = prob1()
    assert all(pattern.search(p) for p in pos)

# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #raise NotImplementedError("Problem 2 Incomplete")
    return re.compile(r"\^\{@\}\(\?\)\[%\]\{\.\}\(\*\)\[_\]\{&\}\$")

# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #raise NotImplementedError("Problem 3 Incomplete")
    return re.compile(r"^(Book|Mattress|Grocery) (store|supplier)$")

# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #raise NotImplementedError("Problem 4 Incomplete")
    return re.compile(r"^([a-zA-Z]|_)(\w|_)*$")

# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """
    #raise NotImplementedError("Problem 5 Incomplete")
    pat1 = re.compile(r"^((?:(?:\s)*)(?:if|elif|else|for|while|try|except|finally|with|def|class).*)$" , re.MULTILINE)
    #print(pat1.findall(code))
    return pat1.sub(r"\1:",code)

# Problem 6
def prob6(filename="fake_contacts.txt"):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """
    #raise NotImplementedError("Problem 6 Incomplete")
    black_book = {}
    regex_name = "[A-Z][a-z]+ (?:[A-Z]. )?[A-Z][a-z]+"
    regex_phone = "((?:1-)?(?:(?:\(\d{3}\))|(?:\d{3}))(?:\/|-| )?\d{3}(?:\/|-| )?\d{4})"
    regex_birth = "\d\d?(?:\/|-)\d\d?(?:\/|-)(?:\d{4}|\d{2})"
    regex_email = "((?:\S*)@(?:(?:[a-zA-Z]|\.)*)*\.(?:com|org|edu|net))"
    pat_name = re.compile(regex_name)
    pat_phone = re.compile(regex_phone)
    pat_birth = re.compile(regex_birth)
    pat_email = re.compile(regex_email)
    with open(filename,'r') as f :
        contacts = f.read().split('\n')
    for contact in contacts :
        if len(contact) != 0 :
            name = pat_name.findall(contact)
            phone = pat_phone.findall(contact)
            if len(phone) == 0 :
                phone = [None]
            birth = pat_birth.findall(contact)
            if len(birth) == 0 :
                birth = [None]
            email = pat_email.findall(contact)
            if len(email) == 0 :
                email = [None]
            black_book[name[0]] = {"birthday":birth[0] , "email":email[0] , "phone":phone[0]}
    return black_book
        