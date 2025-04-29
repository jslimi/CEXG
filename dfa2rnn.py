"""
Modified code from: https://github.com/23Vladymir57/TMLR_Code
"""
import numpy as np

def dfa2srn(trans_mat, decod_mat):
    """np.array, np.array -> list(torch.tensors)
    Given a transitions function and a decoder defined by matrices this functions outputs a saturated Simple recurrent network capable of
    simulating the DFA.
    """
    S, Q = trans_mat.shape
    J = 128 * np.log(2)
    ### Initalizing the parameters
    # The encoder
    U = np.zeros((S * Q, S))
    W = 3 * np.ones((S * Q, S * Q))
    # The decoder
    V = np.zeros(S * Q)
    #biases
    b = np.zeros(S * Q)
    c = np.zeros(S * Q)

    ### Constructing the parameters
    # The encoder
    for j in range(Q):
        for k in range(S):
            U[k + j * S, k] = 2

    for s in range(Q):
        for k in range(S):
            elem = int(trans_mat[k, s])
            W[k + elem * S, s * S:(s + 1) * S] = 1

    # The decoder
    for k in range(Q):
        if decod_mat[k] == 1:
            V[k * S:(k + 1) * S] = decod_mat[k]
        else:
            V[k * S:(k + 1) * S] = -1

    ### Packing all the tensors
    target = [J*U, J*(-W), J*V, b, c]

    return target

# T_m = np.array([[1,0],[0,1]])
# D_m = np.array([0,0])
#
# print(dfa2srn(T_m, D_m))