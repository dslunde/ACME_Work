"""Volume 3: Introduction to BeautifulSoup.
Darren Lund
BATMAN 235
Tomorrow
"""

from matplotlib import pyplot as plt
from bs4 import BeautifulSoup
import re
import numpy as np


# Example HTML string from the lab.
pig_html = """
<html><head><title>Three Little Pigs</title></head>
<body>
<p class="title"><b>The Three Little Pigs</b></p>
<p class="story">Once upon a time, there were three little pigs named
<a href="http://example.com/larry" class="pig" id="link1">Larry,</a>
<a href="http://example.com/mo" class="pig" id="link2">Mo</a>, and
<a href="http://example.com/curly" class="pig" id="link3">Curly.</a>
<p>The three pigs had an odd fascination with experimental construction.</p>
<p>...</p>
</body></html>
"""


# Problem 1
def prob1():
    """Examine the source code of http://www.example.com. Determine the names
    of the tags in the code and the value of the 'type' attribute associated
    with the 'style' tag.

    Returns:
        (set): A set of strings, each of which is the name of a tag.
        (str): The value of the 'type' attribute in the 'style' tag.
    """
    #raise NotImplementedError("Problem 1 Incomplete")
    tags = set(['html','head','title','meta','style','body','div','h1','p','a'])
    val = 'text/css'
    return tags,val


# Problem 2
def prob2(code):
    """Return a list of the names of the tags in the given HTML code."""
    #raise NotImplementedError("Problem 1 Incomplete")
    soup = BeautifulSoup(code)
    return [tag.name for tag in soup.find_all(True)]


# Problem 3
def prob3(filename="example.html"):
    """Read the specified file and load it into BeautifulSoup. Find the only
    <a> tag with a hyperlink and return its text.
    """
    #raise NotImplementedError("Problem 3 Incomplete")
    with open(filename,'r') as f :
        code = f.read()
    soup = BeautifulSoup(code,"html.parser")
    p_tags = soup.find_all('a')
    #print([child for tag in p_tags for child in tag.children])
    sols = [tag for tag in p_tags for child in tag.children if not isinstance(child,str) and'href' in child.attrs]
    if len(sols) > 0 :
        ans = sols[0].string
    else :
        ans = ''
    return ans


# Problem 4
def prob4(filename="san_diego_weather.html"):
    """Read the specified file and load it into BeautifulSoup. Return a list
    of the following tags:

    1. The tag containing the date 'Thursday, January 1, 2015'.
    2. The tags which contain the links 'Previous Day' and 'Next Day'.
    3. The tag which contains the number associated with the Actual Max
        Temperature.

    Returns:
        (list) A list of bs4.element.Tag objects (NOT text).
    """
    #raise NotImplementedError("Problem 4 Incomplete")
    with open(filename,'r') as f :
        code = f.read()
    soup = BeautifulSoup(code,"html.parser")
    x = soup.find_all(string='Thursday, January 1, 2015')
    x = [x[i].parent for i in range(len(x))][0]
    print(x)
    y1 = soup.find_all('div',attrs={'class':'previous-link'})[0].find_all('a')[0]
    y2 = soup.find_all('div',attrs={'class':'next-link'})[0].find_all('a')[0]
    y3 = soup.find_all('td',attrs={'class':'indent'},string='Max Temperature')[0].parent.find_all('span',attrs={'class':'wx-value'})[0]
    return [x,y1,y2,y3]


# Problem 5
def prob5(filename="large_banks_index.html"):
    """Read the specified file and load it into BeautifulSoup. Return a list
    of the tags containing the links to bank data from September 30, 2003 to
    December 31, 2014, where the dates are in reverse chronological order.

    Returns:
        (list): A list of bs4.element.Tag objects (NOT text).
    """
    #raise NotImplementedError("Problem 5 Incomplete")
    pattern = re.compile(r"(^(?:September|October|December) \d+, 2003$)|(^\S* \d+, 20(?:(?:0[4-9])|(?:1[0-4])))")
    with open(filename,'r') as f :
        code = f.read()
    soup = BeautifulSoup(code,"html.parser")
    dates = soup.find_all(string=pattern)
    return [date.parent for date in dates]

# Problem 6
def prob6(filename="large_banks_data.html"):
    """Read the specified file and load it into BeautifulSoup. Create a single
    figure with two subplots:

    1. A sorted bar chart of the seven banks with the most domestic branches.
    2. A sorted bar chart of the seven banks with the most foreign branches.

    In the case of a tie, sort the banks alphabetically by name.
    """
    #raise NotImplementedError("Problem 6 Incomplete")
    with open(filename,'r') as f :
        code = f.read()
    soup = BeautifulSoup(code,"html.parser")
    banks = soup.find_all('table')[0].find_all('table')[1].find_all('tr')
    BANKS = {}
    for bank in banks[1:] :
        info = bank.find_all('td')
        name = info[0].string
        domest = info[-4].string.replace(',', '')
        foreign = info[-3].string.replace(',', '')
        if domest == '.' :
            domest = 0
        if foreign == '.' :
            foreign = 0
        BANKS[name] = {"Domestic":int(domest),"Foreign":int(foreign)}
    magic_number = 7
    X = np.arange(magic_number)
    most_domest = sorted(BANKS, key=lambda x : BANKS[x]['Domestic'],reverse=True)
    most_foreign = sorted(BANKS, key=lambda x : BANKS[x]['Foreign'],reverse=True)
    most_domest = most_domest[:magic_number]
    most_domest_num = [int(BANKS[most_domest[i]]['Domestic']) for i in range(magic_number)]
    most_foreign = most_foreign[:magic_number]
    most_foreign_num = [int(BANKS[most_foreign[i]]['Foreign']) for i in range(magic_number)]
    plt.subplot(211)
    plt.barh(X,most_domest_num,align='center')
    plt.title("Domestic")
    plt.yticks(X,most_domest,rotation=10,fontsize=8)
    plt.subplot(212)
    plt.barh(X,most_foreign_num,align='center')
    plt.yticks(X,most_foreign,rotation=10,fontsize=8)
    plt.title("Foreign")
    plt.tight_layout()
    plt.show()