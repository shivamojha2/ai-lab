"""
Attention implementation from scratch in numpy
"""
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # for stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def attention(Q, K, V):
    # Q: (seq_len, d_k)
    # K: (seq_len, d_k)
    # V: (seq_len, d_v)
    
    d_k = Q.shape[-1]
    
    # Compute similarity (dot product)
    scores = np.dot(Q, K.T) / np.sqrt(d_k) # (seq_len, seq_len)
    
    # Turn into probabilities
    weights = softmax(scores) # (seq_len, seq_len)
    
    # Weighted sum of values
    output = np.dot(weights, V) # (seq_len, d_v)
    
    return output, weights

if __name__ == "__main__":
    np.random.seed(0)
    
    Q = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    K = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    V = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    out, attn = attention(Q, K, V)
    print("Output:\n", out)
    print("Attention Weights:\n", attn)
