import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def not_in_alphabet(vector: np.ndarray, vectors: list[np.ndarray])-> bool:
    """
    Checks if a vector is in the list of vectors or not
    """
    for v in vectors:
        if np.array_equal(v, vector):
            return False
    return True

def nearest_neighbor(vector: np.ndarray, vectors: list[np.ndarray])-> np.ndarray:
    """
    Finds the closest neighbor by calculating norm2 between a vector and all other vectors
    """
    scores = [np.linalg.norm(vector - v) for v in vectors]
    return vectors[scores.index(min(scores))]

def rnn(model: dict[str, np.ndarray], vector: np.ndarray, activation = 'tanh')-> float:
    """
    Computes the prediction of an rnn model for a given input vector
    E: activation(Ux+Wh)
    D: activation(Vh)
    """
    if activation == 'sigmoid':
        hidden_layer = sigmoid(model["w"] @ model["h"] + model["u"] @ vector + model["b"])
        return sigmoid(model["v"] @ hidden_layer + model["c"])
    else:
        hidden_layer = np.tanh(model["w"] @ model["h"] + model["u"] @ vector + model["b"])
        return np.tanh(model["v"] @ hidden_layer + model["c"])

def drnn_dx_sigmoid(srn: dict[str, np.ndarray], x: np.ndarray)-> np.ndarray:
    """
    Return the derivative of sigmoid rnn prediction with respect to x
    """
    return (rnn(srn, x, 'sigmoid') * (1 - rnn(srn, x, 'sigmoid')) *
            sigmoid(srn["w"] @ srn["h"] + srn["u"] @ x + srn["b"]) *
            (1 - sigmoid(srn["w"] @ srn["h"] + srn["u"] @ x + srn["b"])) * srn["v"] @ srn["u"])

def drnn_dx_tanh(srn: dict[str, np.ndarray], x: np.ndarray)-> np.ndarray:
    """
    Return the derivative of tanh rnn prediction with respect to x
    """
    # return (1 - np.power(rnn(srn, x), 2)) @ np.power(np.tanh(srn["w"] @ srn["h"] + srn["u"] @ x), 2)
    b = np.power((1 / np.cosh(srn['h'] @ srn['w'] + srn['u'] @ x + srn["b"])), 2)
    c = np.power((1 / np.cosh(srn['v'] * np.tanh(srn['h'] @ srn['w'] + srn['u'] @ x + srn["b"]))), 2)
    return srn['v'] * (b*c) @ srn['u']

def loss_update_sigmoid(srno: dict[str, np.ndarray], srna: dict[str, np.ndarray], x: np.ndarray):
    """
    returns the loss function derivative update with respect to x for srn with sigmoid activation
    """
    return -2 * ((rnn(srna, x, 'sigmoid') - rnn(srno, x, 'sigmoid')) *
                (drnn_dx_sigmoid(srno, x) - drnn_dx_sigmoid(srna, x)))

def loss_update_tanh(srno: dict[str, np.ndarray], srna: dict[str, np.ndarray], x: np.ndarray):
    """
    returns the loss function derivative update with respect to x for srn with tanh activation
    """
    return 0.5 * (rnn(srno, x) + rnn(srna, x))  * (drnn_dx_tanh(srno, x) + drnn_dx_tanh(srna, x))

def generate_xi(srno: dict[str, np.ndarray], srna: dict[str, np.ndarray], input_size: int, activation = 'tanh')-> np.ndarray:
    """
    Takes two SRNs and returns a vector of size n of the two SRNs
    srno: Simple Recurrent Net of the Original rnn
    srna: Simple Recurrent Net of the Automata
    input_size: the size of the input vectors
    activation: the type of the activation functions; sigmoid or tanh
    """
    if activation == 'tanh':
        eta = 0.01
        x = np.zeros(input_size)
        for i in range(100):
            cost_function = 1/4 * np.power(rnn(srno, x) + rnn(srna, x), 2)
            x = np.subtract(x, eta * loss_update_tanh(srno, srna, x))
            if np.linalg.norm(cost_function) < 0.0001:
                break
        return x
    else:
        eta = 0.01
        x = np.zeros(input_size)
        for i in range(100):
            cost_function = 1 - np.power(rnn(srno, x, 'sigmoid') - rnn(srna, x, 'sigmoid'), 2)
            x = x - eta * loss_update_sigmoid(srno, srna, x)
            if np.linalg.norm(cost_function) < 0.0001:
                break
        return x

def cex_found(cex, srno, srna, activation = 'tanh'):
    """
    Checks if CEX is already found
    srno: Simple Recurrent Net of the Original rnn
    srna: Simple Recurrent Net of the Automata
    activation: the type of the activation functions; sigmoid or tanh
    """
    if activation == 'tanh':
        for vector in cex:
            #compares the two predictions of srno and srna on a vector xi
            if rnn(srno, vector) == rnn(srno, vector):
                srno["h"] = np.tanh(np.matmul(srno["w"], srno["h"]) + np.matmul(srno["u"], vector))
                srna["h"] = np.tanh(np.matmul(srna["w"], srna["h"]) + np.matmul(srna["u"], vector))
            else:
                return True
        return False
    else:
        for vector in cex:
            #compares the two predictions of srno and srna on a vector xi
            if rnn(srno, vector, 'sigmoid') == rnn(srno, vector, 'sigmoid'):
                srno["h"] = sigmoid(np.matmul(srno["w"], srno["h"]) + np.matmul(srno["u"], vector))
                srna["h"] = sigmoid(np.matmul(srna["w"], srna["h"]) + np.matmul(srna["u"], vector))
            else:
                return True
        return False

def generate_cex(srno: dict[str, np.ndarray], srna: dict[str, np.ndarray], alphabet: list[np.ndarray], seq_size: int, input_size: int, activation = 'tanh')-> list[np.ndarray]:
    """
    Generates a CEX such that srno(CEX) !=0 srna(CEX)
    srno: Simple Recurrent Net of the Original rnn
    srna: Simple Recurrent Net of the Automata
    alphabet: the alphabet set of the language
    seq_size: the maximum length for the CEX
    input_size: the size of the input vectors
    activation: the type of the activation functions; sigmoid or tanh
    """
    if activation == 'tanh' or activation == 'sigmoid':
        pass
    else:
        raise ValueError(f"Change {activation} to one of the supported activation functions: sigmoid, tanh.")
    cex = []
    for i in range(seq_size):
        #generates an input vector xi for a sequence X=x1..xn such that srno(X)!=srna(X)
        xi = generate_xi(srno, srna, input_size, activation)
        #checks if the generated vector xi belongs to the alphabet otherwise replaces it with its nearest neighbor
        if not_in_alphabet(xi, alphabet):
            xi = nearest_neighbor(xi, alphabet)
        cex.append(xi)
        #checks if the cex sequence is already found to break before reaching maximum length n
        if cex_found(cex, srno, srna, activation):
            break
        # update the hidden state values for the recurrent nets
        srno["h"] = sigmoid(np.matmul(srno["w"], srno["h"]) + np.matmul(srno["u"], xi))
        srna["h"] = sigmoid(np.matmul(srna["w"], srna["h"]) + np.matmul(srna["u"], xi))
    return cex