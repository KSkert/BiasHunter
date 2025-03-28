import numpy as np
import matplotlib.pyplot as plt
import random

def generate_1D_groups(n=30):

    x0 = np.random.normal(loc=2, scale=0.5, size=n)
    s0 = np.zeros(n)

    x1 = np.random.normal(loc=6, scale=0.5, size=n)
    s1 = np.ones(n)

    x = np.concatenate([x0, x1])
    s = np.concatenate([s0, s1])
    return x, s

def perturb_value(val, all_vals, strength=0.3):
    sorted_vals = sorted(set(all_vals))
    if val not in sorted_vals:
        return val
    idx = sorted_vals.index(val)
    options = []
    if idx > 0:
        options.append(sorted_vals[idx - 1])
    if idx < len(sorted_vals) - 1:
        options.append(sorted_vals[idx + 1])
    options.append(val)
    return random.choice(options)

def main():
    np.random.seed(0)
    random.seed(0)

    x, s = generate_1D_groups(n=30)

    # Plot original points
    plt.figure(figsize=(10, 4))
    plt.scatter(x[s == 0], np.zeros_like(x[s == 0]), color='blue', label='Original s=0', alpha=0.7)
    plt.scatter(x[s == 1], np.ones_like(x[s == 1]), color='red', label='Original s=1', alpha=0.7)

    # naively pick one point from s=0 to do a counterfactual flip
    i = random.choice(np.where(s == 0)[0])
    x_orig = x[i]
    y_orig = 0
    y_flip = 1

    # same x, different s
    plt.scatter([x_orig], [y_flip], color='green', marker='s', s=120, edgecolor='black', label='Counterfactual flip')
    plt.plot([x_orig, x_orig], [y_orig, y_flip], 'k--', linewidth=1.5)

    # Pick one from each group for perturbation-based pair
    i0 = random.choice(np.where(s == 0)[0])
    i1 = random.choice(np.where(s == 1)[0])
    x0_orig, x1_orig = x[i0], x[i1]

    x0_pert = perturb_value(x0_orig, x)
    x1_pert = perturb_value(x1_orig, x)

    plt.scatter([x0_pert], [0], marker='X', color='blue', s=150, edgecolor='white', label='Perturbed s=0')
    plt.scatter([x1_pert], [1], marker='X', color='red', s=150, edgecolor='white', label='Perturbed s=1')
    plt.plot([x0_pert, x1_pert], [0, 1], color='purple', linestyle='-', linewidth=2, label='Pair perturbation link')
    plt.yticks([0, 1], ['s=0', 's=1'])
    plt.xlabel("Non-sensitive feature (x)")
    plt.title("Counterfactual Flip vs Pairwise Perturbation (1D Example)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()