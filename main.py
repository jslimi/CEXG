import numpy as np

from cexg import *
from dfa2rnn import dfa2srn

#defining transition matrix
T_m = np.array([[0, 1, 1, 0],[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])
#defining decoder matrix
D_m = np.array([0, 0, 0, 1])
#transforming dfa to srn
srn = dfa2srn(T_m, D_m)

# defining the alphabet
alphabet_ = [np.array([1.0, 0.0]), np.array([0.0, 1.0])] # [a, b]
letter = np.zeros(4)
for i in range(4):
    if i%2==0:
        letter[i] = 1.
print(rnn(srn, letter, 'sigmoid'))
exit()

# accepts only b
srn1 = {"u": np.array([[177.4457,   0.0000],
    [  0.0000, 177.4457],
    [177.4457,   0.0000],
    [  0.0000, 177.4457]]),
        "w": np.array([[-266.1685, -266.1685,  -88.7228,  -88.7228],
    [ -88.7228,  -88.7228, -266.1685, -266.1685],
    [ -88.7228,  -88.7228, -266.1685, -266.1685],
    [-266.1685, -266.1685,  -88.7228,  -88.7228]]),
        "v": np.array([88.7228, 88.7228,  -88.7228,  -88.7228]),
        "h": np.array([1., 0., 0., 0.]),
        "b": np.array([0., 0., 0., 0.]),
        "c": np.array([0., 0., 0., 0.])}

# accepts a and b (anything)
srn2 = {"u": np.array([[177.4457,   0.0000],
    [  0.0000, 177.4457],
    [177.4457,   0.0000],
    [  0.0000, 177.4457]]),
        "w": np.array([[-266.1685, -266.1685,  -88.7228,  -88.7228],
    [ -88.7228,  -88.7228, -266.1685, -266.1685],
    [ -88.7228,  -88.7228, -266.1685, -266.1685],
    [-266.1685, -266.1685,  -88.7228,  -88.7228]]),
        "v": np.array([-88.7228, -88.7228, -88.7228, -88.7228]),
        "h" :np.array([1., 0., 0., 0.]),
        "b": np.array([0., 0., 0., 0.]),
        "c": np.array([0., 0., 0., 0.])}

# generating cex
counter_example = generate_cex(srn1, srn2, alphabet_, 10, 2)
result = ""
for cex_ in counter_example:
    if np.array_equal(cex_, np.array([1.0, 0.0])):
        result+="a"
    elif np.array_equal(cex_, np.array([0.0, 1.0])):
        result+="b"
    else:
        print("This letter not in alphabet: ",  cex_)
        break

print("Generated counterexample: ", result)