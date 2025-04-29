from utils import *
from aalpy.base import SUL

class RnnBinarySUL(SUL):
    """
    SUL used to learn DFA from RNN Binary Classifiers.
    """

    def __init__(self, nn):
        super().__init__()
        self.word = []
        self.rnn = nn

    def pre(self):
        self.word = []
        self.rnn.state = self.rnn.rnn.initial_state()

    # def post(self):
    #     self.rnn.renew()

    def step(self, letter):
        self.word.append(letter)
        if letter is None:
            return self.rnn.step(None)
        return self.rnn.step(letter)


tomita = 2
filename = f"rnn/tom{tomita}.th"
model_state = torch.load(filename)
for name, param in model_state.items():
    print(f"Parameter Name: {name}, Requires Grad: {param.requires_grad}")
    print(param)