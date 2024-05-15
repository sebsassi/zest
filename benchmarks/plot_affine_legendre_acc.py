import numpy as np
import matplotlib.pyplot as plt

data = np.load("affine_legendre_lmax_accuracy_200.npz")

accurate_shifted = data["accurate_shifted"]
accurate_scaled = data["accurate_scaled"]
shifted = data["shifted"]
scaled = data["scaled"]

shifted_err = np.abs(accurate_shifted - shifted)
scaled_err = np.abs(accurate_scaled - scaled)

shifted_err[accurate_shifted != 0] /= np.abs(accurate_shifted[accurate_shifted != 0])
scaled_err[accurate_scaled != 0] /= np.abs(accurate_scaled[accurate_scaled != 0])

quantile = 0.90
quantile_shifted_err = [np.quantile(e, quantile) for e in shifted_err]
quantile_scaled_err = [np.quantile(e, quantile) for e in scaled_err]

max_scaled = [np.max(np.abs(s)) for s in scaled]
max_shifted = [np.max(np.abs(s)) for s in shifted]

fig, ax = plt.subplots(2, 2)

ax[0][0].semilogy(quantile_shifted_err)
ax[1][0].semilogy(quantile_scaled_err)
ax[0][1].semilogy(max_shifted)
ax[1][1].semilogy(max_scaled)
plt.show()