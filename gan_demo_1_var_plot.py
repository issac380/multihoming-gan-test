import matplotlib.pyplot as plt

true_p_heads = 0.7

ITER = 2000

with open(f"gan_estimates{ITER}.txt", "r") as f:
    estimates = [float(line.strip()) for line in f if line.strip()]

plt.hist(
    estimates,
    bins=10,
    alpha=0.7,
    color='skyblue',
    edgecolor='black'
)

plt.axvline(
    true_p_heads,
    color='red',
    linestyle='--',
    linewidth=2,
    label='True Probability'
)

plt.xlabel('Estimated Probability of Heads')
plt.ylabel('Frequency')
plt.title('Distribution of GAN Estimates from Text File')
plt.xlim(0, 1)
plt.legend()
plt.show()
