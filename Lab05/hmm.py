import numpy as np
from hmmlearn import hmm

model = hmm.CategoricalHMM(n_components=3)
model.startprob_ = np.array([1 / 3, 1 / 3, 1 / 3])
model.transmat_ = np.array(
    [
        [0.0, 0.5, 0.5],
        [0.5, 0.25, 0.25],
        [0.5, 0.25, 0.25],
    ]
)
model.emissionprob_ = np.array(
    [
        [0.10, 0.20, 0.40, 0.30],
        [0.15, 0.25, 0.50, 0.10],
        [0.20, 0.30, 0.40, 0.10],
    ]
)


# sample = np.array([[2], [1], [0]])
# print("log-likelihood of sequence [S, B, FB]:", model.score(sample))
# print("most likely hidden state sequence:", model.decode(sample)[1])

# FB = 0, B = 1, S = 2, NS = 3
# FB, FB, S, B, B, S, B, B, NS, B, B, S.

sample = np.array([[0], [0], [2], [1], [1], [2], [1], [1], [3], [1], [1], [2]])

ind_to_hidden_state_lable = {0: "dificil", 1: "mediu", 2: "usor"}

most_likely_sequence = model.decode(sample)[1]
most_likely_sequence_labled = []
for state in most_likely_sequence:
    most_likely_sequence_labled.append(ind_to_hidden_state_lable[state])

print(
    "Cea mai probabila secventa de stari bazat pe notele obervata este:",
    most_likely_sequence_labled,
)
print(
    "Probabilitatea acesteia:",
    np.exp(model.decode(sample)[0]),
)
