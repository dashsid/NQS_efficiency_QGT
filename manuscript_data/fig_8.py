import matplotlib.pyplot as plt
import numpy as np




L = 8
alpha_list = np.flip(np.array([2, 4, 6, 8, 10])) # gapless


cmap = plt.cm.plasma
colors = cmap(np.linspace(0, 1, len(alpha_list)))
fontsize_suptitle = 16+2+2
fontsize_title = 14+2+2
fontsize_labels = 13+2+2
fontsize_legend = 13+2
transparency = 0.8

infidelity_list = []
relE_list = []
rank_list = []

fig, axes = plt.subplots(2,4,figsize=(15,11))
#plt.suptitle("Spectrum of the Hessian, for the spin-1 BLBQ chain with L=8", fontsize=fontsize_suptitle)

iterations = 15000
neg_limit = 25
model = "AFH"
axes[0,1].set_title(r"(a) AFH ($\theta=0$)                                               ", fontsize=fontsize_title)
for ia,alp in enumerate(alpha_list):
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    infidelity = np.load(infidelity_file)[0]

    hess_eigvals_file = f"./data_infidelity_optimization/hessian_eigenvalues/hess_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    hess_eigvals = np.load(hess_eigvals_file)[:,0]
    hess_eigvals /= hess_eigvals.max()
    positive_hess_logevals = np.log10(hess_eigvals[hess_eigvals>0])
    negative_hess_logevals = np.log10(abs(hess_eigvals[hess_eigvals<0]))

    axes[0,0].hist(negative_hess_logevals, bins=70, alpha=transparency, label=rf"$\alpha={alp}$, "+r"$\mathcal{I}$"+f"={infidelity:.2e}", color=colors[-ia-1])#, edgecolor='black')
    axes[0,1].hist(positive_hess_logevals, bins=70, alpha=transparency, label=rf"$\alpha={alp}$, "+r"$\mathcal{I}$"+f"={infidelity:.2e}", color=colors[-ia-1])#, edgecolor='black')
axes[0,0].legend(loc="upper right", fontsize=fontsize_legend)
axes[0,0].set_xlabel(r'$\log_{10}\left(|\lambda_{\rm Hess.}|\right)\Theta\left(-\lambda_{\rm Hess.}\right)$', fontsize=fontsize_labels)
axes[0,1].set_xlabel(r'$\log_{10}\left(|\lambda_{\rm Hess.}|\right)\Theta\left( \lambda_{\rm Hess.}\right)$', fontsize=fontsize_labels)
axes[0,1].set_xlim(-neg_limit,0)
axes[0,0].set_xlim(0,-neg_limit)


model = "AKLT"
axes[0,3].set_title(r"(b) AKLT ($\theta=\arctan(1/3)$)                                               ", fontsize=fontsize_title)
for ia,alp in enumerate(alpha_list):
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    infidelity = np.load(infidelity_file)[0]

    hess_eigvals_file = f"./data_infidelity_optimization/hessian_eigenvalues/hess_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    hess_eigvals = np.load(hess_eigvals_file)[:,0]
    hess_eigvals /= hess_eigvals.max()
    positive_hess_logevals = np.log10(hess_eigvals[hess_eigvals>0])
    negative_hess_logevals = np.log10(abs(hess_eigvals[hess_eigvals<0]))

    axes[0,2].hist(negative_hess_logevals, bins=70, alpha=transparency, label=rf"$\alpha={alp}$, "+r"$\mathcal{I}$"+f"={infidelity:.2e}", color=colors[-ia-1])#, edgecolor='black')
    axes[0,3].hist(positive_hess_logevals, bins=70, alpha=transparency, label=rf"$\alpha={alp}$, "+r"$\mathcal{I}$"+f"={infidelity:.2e}", color=colors[-ia-1])#, edgecolor='black')
axes[0,2].legend(loc="upper right", fontsize=fontsize_legend)
axes[0,2].set_xlabel(r'$\log_{10}\left(|\lambda_{\rm Hess.}|\right)\Theta\left(-\lambda_{\rm Hess.}\right)$', fontsize=fontsize_labels)
axes[0,3].set_xlabel(r'$\log_{10}\left(|\lambda_{\rm Hess.}|\right)\Theta\left( \lambda_{\rm Hess.}\right)$', fontsize=fontsize_labels)
axes[0,3].set_xlim(-neg_limit,0)
axes[0,2].set_xlim(0,-neg_limit)



model = "LS"
axes[1,1].set_title(r"(c) critical point ($\theta=\pi/4$)                                               ", fontsize=fontsize_title)
for ia,alp in enumerate(alpha_list):
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    infidelity = np.load(infidelity_file)[0]

    hess_eigvals_file = f"./data_infidelity_optimization/hessian_eigenvalues/hess_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    hess_eigvals = np.load(hess_eigvals_file)[:,0]
    hess_eigvals /= hess_eigvals.max()
    positive_hess_logevals = np.log10(hess_eigvals[hess_eigvals>0])
    negative_hess_logevals = np.log10(abs(hess_eigvals[hess_eigvals<0]))

    axes[1,0].hist(negative_hess_logevals, bins=70, alpha=transparency, label=rf"$\alpha={alp}$, "+r"$\mathcal{I}$"+f"={infidelity:.2e}", color=colors[-ia-1])#, edgecolor='black')
    axes[1,1].hist(positive_hess_logevals, bins=70, alpha=transparency, label=rf"$\alpha={alp}$, "+r"$\mathcal{I}$"+f"={infidelity:.2e}", color=colors[-ia-1])#, edgecolor='black')
axes[1,0].legend(loc="upper right", fontsize=fontsize_legend)
axes[1,0].set_xlabel(r'$\log_{10}\left(|\lambda_{\rm Hess.}|\right)\Theta\left(-\lambda_{\rm Hess.}\right)$', fontsize=fontsize_labels)
axes[1,1].set_xlabel(r'$\log_{10}\left(|\lambda_{\rm Hess.}|\right)\Theta\left( \lambda_{\rm Hess.}\right)$', fontsize=fontsize_labels)
axes[1,1].set_xlim(-neg_limit,0)
axes[1,0].set_xlim(0,-neg_limit)



model = "gapless"
axes[1,3].set_title(r"(d) critical region ($\theta=\arctan(2)$)                                               ", fontsize=fontsize_title)
for ia,alp in enumerate(alpha_list):
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    infidelity = np.load(infidelity_file)[0]

    hess_eigvals_file = f"./data_infidelity_optimization/hessian_eigenvalues/hess_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    hess_eigvals = np.load(hess_eigvals_file)[:,0]
    hess_eigvals /= hess_eigvals.max()
    positive_hess_logevals = np.log10(hess_eigvals[hess_eigvals>0])
    negative_hess_logevals = np.log10(abs(hess_eigvals[hess_eigvals<0]))

    axes[1,2].hist(negative_hess_logevals, bins=70, alpha=transparency, label=rf"$\alpha={alp}$, "+r"$\mathcal{I}$"+f"={infidelity:.2e}", color=colors[-ia-1])#, edgecolor='black')
    axes[1,3].hist(positive_hess_logevals, bins=70, alpha=transparency, label=rf"$\alpha={alp}$, "+r"$\mathcal{I}$"+f"={infidelity:.2e}", color=colors[-ia-1])#, edgecolor='black')
    
axes[1,2].set_xlabel(r'$\log_{10}\left(|\lambda_{\rm Hess.}|\right)\Theta\left(-\lambda_{\rm Hess.}\right)$', fontsize=fontsize_labels)
axes[1,3].set_xlabel(r'$\log_{10}\left(|\lambda_{\rm Hess.}|\right)\Theta\left( \lambda_{\rm Hess.}\right)$', fontsize=fontsize_labels)
axes[1,3].set_xlim(-neg_limit,0)
axes[1,2].set_xlim(0,-neg_limit)

axes[1,2].legend(loc="upper right", fontsize=fontsize_legend)

axes[0,0].set_ylabel("Occurence of Hessian eigenvalues", fontsize=fontsize_labels)
axes[1,0].set_ylabel("Occurence of Hessian eigenvalues", fontsize=fontsize_labels)

for ax in axes.flat:
    ax.tick_params(axis='both', labelsize=fontsize_legend)
    ax.set_ylim(0,200)

axes[0, 1].yaxis.set_visible(False)
axes[0, 3].yaxis.set_visible(False)
axes[1, 1].yaxis.set_visible(False)
axes[1, 3].yaxis.set_visible(False)

plt.tight_layout()
plt.savefig("fig_8.pdf", dpi=300, bbox_inches="tight")

