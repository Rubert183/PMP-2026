import numpy as np
from hmmlearn.hmm import CategoricalHMM

states = ["room 1", "room 2", "room 3", "room 4"]
observations = ["red", "green", "blue"]
initial_probabilities = np.array([0.25, 0.25, 0.25, 0.25])
transition_probabilities = np.array(
    [
        [0.2, 0.4, 0.0, 0.4],
        [0.4, 0.2, 0.4, 0.0],
        [0.0, 0.4, 0.2, 0.4],
        [0.4, 0.0, 0.4, 0.2],
    ]
)
emission_probabilities = np.array(
    [
        [0.7, 0.2, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6],
        [0.5, 0.2, 0.3],
    ]
)

model = CategoricalHMM(n_components=4, random_state=np.random.RandomState(42))
model.startprob_ = initial_probabilities
model.transmat_ = transition_probabilities
model.emissionprob_ = emission_probabilities

observation_sequence = ["red", "green", "blue", "red"]
observation_indices = np.array(
    [[observations.index(item)] for item in observation_sequence]
)

viterbi_log_probability, best_state_indices = model.decode(
    observation_indices,
    algorithm="viterbi",
)

best_states = [states[index] for index in best_state_indices]
print("Cea mai probabila secventa de camere:", best_states)
print("Probabilitatea acestei secvente:", np.exp(viterbi_log_probability))

new_trans_mat = np.array(
    [
        [0.0, 0.5, 0.0, 0.5],
        [0.5, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.5],
        [0.5, 0.0, 0.5, 0.0],
    ]
)

new_model = CategoricalHMM(n_components=4, random_state=np.random.RandomState(42))
new_model.startprob_ = initial_probabilities
new_model.transmat_ = new_trans_mat
new_model.emissionprob_ = emission_probabilities

viterbi_log_probability, best_state_indices = new_model.decode(
    observation_indices,
    algorithm="viterbi",
)

best_states = [states[index] for index in best_state_indices]
print(
    "Cea mai probabila secventa de camere cu conditia ca robotul nu ramane pe loc:",
    best_states,
)
print(
    "Probabilitatea acestei secvente cu conditia ca robotul nu ramane pe loc:",
    np.exp(viterbi_log_probability),
)
