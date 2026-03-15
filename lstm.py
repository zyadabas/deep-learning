import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lstm_forward(x_t, h_prev, c_prev, weights, biases):
    Wf, Whi, Wc, Wo = weights['input']
    Whf, Whi_h, Whc, Who = weights['hidden']
    bf, bi, bc, bo = biases

    f_t = sigmoid(Wf * x_t + Whf * h_prev + bf)

    i_t = sigmoid(weights['input'][1] * x_t + weights['hidden'][1] * h_prev + bi)

    c_tilde_t = np.tanh(Wc * x_t + Whc * h_prev + bc)

    c_t = f_t * c_prev + i_t * c_tilde_t

    o_t = sigmoid(Wo * x_t + Who * h_prev + bo)

    h_t = o_t * np.tanh(c_t)

    return h_t, c_t

params = {
    'input': [0.5, 0.6, 0.7, 0.8],
    'hidden': [0.1, 0.2, 0.3, 0.4],
}
biases = [0, 0, 0, 0]

X = [1, 2, 3]

h_t = 0.0
c_t = 0.0

print(f"{'Step':<10} | {'Input (x_t)':<12} | {'Hidden State (h_t)':<18} | {'Cell State (c_t)':<18}")
print("-" * 70)

for i, x in enumerate(X):
    h_t, c_t = lstm_forward(x, h_t, c_t, params, biases)
    print(f"t={i+1:<8} | {x:<12} | {h_t:<18.4f} | {c_t:<18.4f}")