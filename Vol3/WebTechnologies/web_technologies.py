# solutions.py
"""Volume 3: Web Technologies. Solutions File."""

import numpy as np
import json
import socket
import re
import operator
from matplotlib import pyplot as plt
import os
import requests as req
from scipy.spatial import KDTree

# Problem 1
def prob1(filename="nyc_traffic.json"):
    """Load the data from the specified JSON file. Look at the first few
    entries of the dataset and decide how to gather information about the
    cause(s) of each accident. Make a readable, sorted bar chart showing the
    total number of times that each of the 7 most common reasons for accidents
    are listed in the data set.
    """
    #raise NotImplementedError("Problem 1 Incomplete")
    with open(filename,'r') as f :
        traffic = json.load(f)
    entry = iter(traffic)
    n = len(traffic)
    reasons = {};
    patt = re.compile(r"contributing_factor_vehicle_\d+")
    for i in range(n) :
        stuff = next(entry)
        veh_reasons = patt.findall('\n'.join(stuff.keys()))
        for vehicle in veh_reasons :
            reason = stuff[vehicle]
            if reason in reasons.keys() :
                reasons[reason] = reasons[reason] + 1
            else :
                reasons[reason] = 1
    sorted_reasons = sorted(reasons.items(), key=operator.itemgetter(1),reverse = True)
    sorted_reasons = sorted_reasons[0:7]
    m = len(sorted_reasons)
    R = [sorted_reasons[i][0] for i in range(m)]
    X = np.arange(m)
    Y = [sorted_reasons[i][1] for i in range(m)]
    plt.bar(X,Y, align='center')
    plt.xticks(X,R, rotation=15,fontsize=6)
    plt.ylabel('Instances')
    plt.show()

class TicTacToe:
    def __init__(self):
        """Initialize an empty board. The O's go first."""
        self.board = [[' ']*3 for _ in range(3)]
        self.turn, self.winner = "O", None

    def move(self, i, j):
        """Mark an O or X in the (i,j)th box and check for a winner."""
        if self.winner is not None:
            raise ValueError("the game is over!")
        elif self.board[i][j] != ' ':
            raise ValueError("space ({},{}) already taken".format(i,j))
        self.board[i][j] = self.turn

        # Determine if the game is over.
        b = self.board
        if any(sum(s == self.turn for s in r)==3 for r in b):
            self.winner = self.turn     # 3 in a row.
        elif any(sum(r[i] == self.turn for r in b)==3 for i in range(3)):
            self.winner = self.turn     # 3 in a column.
        elif b[0][0] == b[1][1] == b[2][2] == self.turn:
            self.winner = self.turn     # 3 in a diagonal.
        elif b[0][2] == b[1][1] == b[2][0] == self.turn:
            self.winner = self.turn     # 3 in a diagonal.
        else:
            self.turn = "O" if self.turn == "X" else "X"

    def empty_spaces(self):
        """Return the list of coordinates for the empty boxes."""
        return [(i,j) for i in range(3) for j in range(3)
                                        if self.board[i][j] == ' ' ]
    def __str__(self):
        return "\n---------\n".join(" | ".join(r) for r in self.board)


# Problem 2
class TicTacToeEncoder(json.JSONEncoder):
    """A custom JSON Encoder for TicTacToe objects."""
    #raise NotImplementedError("Problem 2 Incomplete")
    def default(self,obj) :
        if not isinstance(obj,TicTacToe) :
            raise TypeError("Expected a TicTacToe object")
        winner = [str(obj.winner)]
        return {"dtype": "TicTacToe", "board":obj.board, "turn":list(obj.turn),"winner":winner}

# Problem 2
def tic_tac_toe_decoder(obj):
    """A custom JSON decoder for TicTacToe objects."""
    #raise NotImplementedError("Problem 2 Incomplete")
    if obj["dtype"] != "TicTacToe" :
        raise TypeError("Expected a TicTacToe object")
    new_ttt = TicTacToe()
    new_ttt.board = obj["board"]
    new_ttt.turn = obj["turn"][0]
    if obj["winner"][0] == "None" :
        new_ttt.winner = None
    else :
        new_ttt.winner = obj["winner"][0]
    return new_ttt


def mirror_server(server_address=("0.0.0.0", 33333)):
    """A server for reflecting strings back to clients in reverse order."""
    print("Starting mirror server on {}".format(server_address))

    # Specify the socket type, which determines how clients will connect.
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(server_address)    # Assign this socket to an address.
    server_sock.listen(1)               # Start listening for clients.

    while True:
        # Wait for a client to connect to the server.
        print("\nWaiting for a connection...")
        connection, client_address = server_sock.accept()

        try:
            # Receive data from the client.
            print("Connection accepted from {}.".format(client_address))
            in_data = connection.recv(1024).decode()    # Receive data.
            print("Received '{}' from client".format(in_data))

            # Process the received data and send something back to the client.
            out_data = in_data[::-1]
            print("Sending '{}' back to the client".format(out_data))
            connection.sendall(out_data.encode())       # Send data.

        # Make sure the connection is closed securely.
        finally:
            connection.close()
            print("Closing connection from {}".format(client_address))

def mirror_client(server_address=("0.0.0.0", 33333)):
    """A client program for mirror_server()."""
    print("Attempting to connect to server at {}...".format(server_address))

    # Set up the socket to be the same type as the server.
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(server_address)    # Attempt to connect to the server.
    print("Connected!")

    # Send some data from the client user to the server.
    out_data = input("Type a message to send: ")
    client_sock.sendall(out_data.encode())              # Send data.

    # Wait to receive a response back from the server.
    in_data = client_sock.recv(1024).decode()           # Receive data.
    print("Received '{}' from the server".format(in_data))

    # Close the client socket.
    client_sock.close()


# Problem 3
def tic_tac_toe_server(server_address=("0.0.0.0", 44444)):
    """A server for playing tic-tac-toe with random moves."""
    #raise NotImplementedError("Problem 3 Incomplete")
    print("Starting TicTacToe server on {}".format(server_address))
    
    # Specify the socket type, which determines how clients will connect.
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #if server_address[1] > 1023 :
        
    server_sock.bind(server_address)    # Assign this socket to an address.
    server_sock.listen(1)               # Start listening for clients.

    while True:
        # Wait for a client to connect to the server.
        print("\nWaiting for a connection...")
        connection, client_address = server_sock.accept()

        try:
            # Receive data from the client.
            print("Connection accepted from {}.".format(client_address))
            ended = False
            while not ended :
                in_data = connection.recv(1024).decode()    # Receive data.
                print("Received '{}' from client".format(in_data))

                # Process the received data and send something back to the client.
                out_data = json.loads(in_data, object_hook=tic_tac_toe_decoder)
                if out_data.winner == "O" :
                    out_data = "WIN"
                    ended = True
                elif len(out_data.empty_spaces()) == 0 :
                    out_data = "DRAW"
                    ended = True
                else :
                    options = out_data.empty_spaces()
                    r = np.random.randint(0,len(options))
                    move = options[r]
                    i,j = int(move[0]),int(move[1])
                    out_data.move(i,j)
                    if out_data.winner == "X" :
                        also = "LOSE"
                        ended = True
                        print("Sending '{}' back to the client".format(also))
                        connection.sendall(also.encode())
                    out_data = json.dumps(out_data,cls=TicTacToeEncoder)
                print("Sending '{}' back to the client".format(out_data))
                connection.sendall(out_data.encode())       # Send data.

        # Make sure the connection is closed securely.
        finally:
            connection.close()
            print("Closing connection from {}".format(client_address))

# Problem 4
def tic_tac_toe_client(server_address=("0.0.0.0", 44444)):
    posits = [0 , 1, 2]
    """A client program for tic_tac_toe_server()."""
    #raise NotImplementedError("Problem 4 Incomplete")
    def move(out_data) :
        moved = False
        while not moved :
            print(out_data)
            #print("Valid moves are : " + str(out_data.empty_spaces()))
            print("Enter move as 'i j' where i is the row (0-2) and j the column (0-2)")
            print("Please make a move : ")
            try :
                i,j = input().strip().split(' ')
                i,j = int(i), int(j)
                if i not in posits or j not in posits :
                    raise ValueError("Invalid Move")
                out_data.move(i,j)
                moved = True
            except :
                print("Invalid move.")
        return out_data
    
    print("Attempting to connect to server at {}...".format(server_address))

    # Set up the socket to be the same type as the server.
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(server_address)    # Attempt to connect to the server.
    print("Connected!")

    # Send some data from the client user to the server.
    in_data = TicTacToe()
    stuff = True
    while stuff :
        in_data = move(in_data)
        in_data = json.dumps(in_data,cls=TicTacToeEncoder)
        client_sock.sendall(in_data.encode())              # Send data.

        # Wait to receive a response back from the server.
        in_data = client_sock.recv(1024).decode() # Receive data.
        print("Received '{}' from the server".format(in_data))
        try :
            in_data = json.loads(in_data,object_hook=tic_tac_toe_decoder)
        except :
            if in_data == "LOSE" :
                in_data = client_sock.recv(1024).decode() # Receive board.
                in_data = json.loads(in_data,object_hook=tic_tac_toe_decoder)
                print(in_data)
                print("You suck at Tic-Tac-Toe.  I wasn't even trying.")
                stuff = False
            elif in_data == "WIN" :
                print("Wow...you are so good at Tic-Tac-Toe. Congratu-freakin-lations.")
                stuff = False
            elif in_data == "" :
                print("You're so bad even my computer doesn't wan't to play with you.")
                stuff = False

    # Close the client socket.
    client_sock.close()


# Problem 5
def download_nyc_data():
    """Make requests to download data from the following API endpoints.

    Recycling bin locations: https://data.cityofnewyork.us/api/views/sxx4-xhzg/rows.json?accessType=DOWNLOAD

    Residential addresses: https://data.cityofnewyork.us/api/views/7823-25a9/rows.json?accessType=DOWNLOAD

    Save the recycling bin data as nyc_recycling.json and the residential
    address data as nyc_addresses.json.
    """
    #raise NotImplementedError("Problem 5 Incomplete")
    rec_bin_locs = "https://data.cityofnewyork.us/api/views/sxx4-xhzg/rows.json?accessType=DOWNLOAD"
    res_add_locs = "https://data.cityofnewyork.us/api/views/7823-25a9/rows.json?accessType=DOWNLOAD"
    with open('nyc_recycling.json','w') as outfile :
        json.dump(req.get(rec_bin_locs).json(),outfile)
    with open('nyc_addresses.json','w') as outfile :
        json.dump(req.get(res_add_locs).json(),outfile)


# Problem 6
def prob6(recycling="nyc_recycling.json", addresses="nyc_addresses.json"):
    """Load the specifiec data files. Use a k-d tree to determine the distances
    from each address to the nearest recycling bin, and plot a histogram of
    the results.

    DO NOT call download_nyc_data() in this function.
    """
    #raise NotImplementedError("Problem 6 Incomplete")
    with open(recycling,'r') as rec :
        bins = json.load(rec)
    data = bins["data"]
    n = len(data)
    locs = np.zeros((n,2))
    for i in range(n) :
        locs[i] = data[i][-2:]
        if locs[i,0] != locs[i,0] or locs[i,1] != locs[i,1] :
            locs[i] = [0,0]
    locs[:,0] , locs[:,1] = locs[:,1] , locs[:,0]
    
    tree = KDTree(locs)
    with open(addresses,'r') as add :
        homes = json.load(add)
    adds = homes["data"]
    m = len(adds)
    abodes = np.zeros((m,2))
    for i in range(m) :
        P, p,q = adds[i][8].strip().split(' ')
        abodes[i,0], abodes[i,1] = float(p[1:]), float(q[:-1])
    distances = []
    for i in range(m) :
        min_distance, index = tree.query(abodes[i])
        distances.append(min_distance)
    distances.sort()
    n,bins,dist = plt.hist(distances,100)
    plt.show()