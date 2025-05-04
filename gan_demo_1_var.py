import numpy as np
from sklearn.neural_network import MLPClassifier

# Number of iterations for GAN
ITER = 1000
RUNS = 100

# True probability for heads (parameter we want GAN to estimate)
true_p_heads = 0.7

# Function simulating a biased coin toss
def simulate_coin_toss(p_heads, num_flips=100):
    return (np.random.rand(num_flips) < p_heads).astype(int).reshape(-1, 1)

# Gradient function based on given interpolation gradient formula
def compute_gradient(p, p1, p2, q11, q12, q21, q22):
    return (1 / ((p2 - p1)**2)) * (
        -q11 * (p2 - p) + q21 * (p2 - p)
        - q12 * (p - p1) + q22 * (p - p1)
    )

# Run GAN estimation multiple times and append to text file
with open("gan_estimates.txt", "a") as f:
    for run in range(RUNS):
        real_data = simulate_coin_toss(true_p_heads)
        estimated_p_heads = np.random.uniform(0.3, 0.6)  # random starting point
        discriminator = MLPClassifier(hidden_layer_sizes=(5,), max_iter=ITER, tol=1e-4)

        for epoch in range(ITER):
            simulated_data = simulate_coin_toss(estimated_p_heads)
            X = np.vstack([real_data, simulated_data])
            y = np.hstack([np.ones(len(real_data)), np.zeros(len(simulated_data))])
            discriminator.fit(X, y)

            p1, p2 = 0.01, 0.99
            q11 = discriminator.predict_proba([[p1]])[0, 1]
            q21 = discriminator.predict_proba([[p2]])[0, 1]
            q12, q22 = q11, q21

            gradient = compute_gradient(estimated_p_heads, p1, p2, q11, q12, q21, q22)
            learning_rate = 0.1
            estimated_p_heads += learning_rate * gradient
            estimated_p_heads = np.clip(estimated_p_heads, 0.01, 0.99)

        f.write(f"{estimated_p_heads}\n")
