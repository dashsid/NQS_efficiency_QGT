import matplotlib.pyplot as plt 
import numpy as np 


cmap = plt.cm.plasma

def compute_rank(qgt_evals, cutoff=1e-6, normalize=False):
    if not normalize:
        qgt_lgevals = np.log10(qgt_evals)
    else:
        qgt_evals /= qgt_evals.max()
        qgt_lgevals = np.log10(qgt_evals)
    rank_ = len(qgt_lgevals[qgt_lgevals>cutoff])
    return rank_, qgt_lgevals

#cutoff = -12
ix = 0.15 
iy = 0.62
lx = 0.35
ly = 0.35
y0 = 0
yf = 230
x0 = -24
xf = 10
y0_in = 1
yf_in = 4000
x0_in = -24
xf_in = 10
bin_edges = np.linspace(-24, 10, 100)
transparency = 0.8

fontsize_suptitle = 16+2
fontsize_title = 14+2
fontsize_labels = 13+2
fontsize_legend = 13

cutoff = -5
normalize_ = False
L=10

# AFH
fig, axes = plt.subplots(2,2,figsize=(15,11))
#plt.suptitle("Spectrum of the QGT for the spin-1 Bilinear-Biquadratic chain with L=10", fontsize=fontsize_suptitle)

flattened_axes = axes.flatten()
title_list = [r"(a) AFH ($\theta=0$)", r"(b) AKLT ($\theta=\arctan(1/3)$)", r"(c) critical ($\theta=\pi/4$)", r"(d) critical ($\theta=\arctan(2)$)"]
model_list = ["AFH", "AKLT", "LS", "gapless"]
lr_list = [1e-2, 1e-2, 1e-2, 9e-3]

for iax, ax0 in enumerate(flattened_axes):

    ax0.set_title(title_list[iax], fontsize=fontsize_title)
    axins = ax0.inset_axes([ix, iy, lx, ly])
    model = model_list[iax]
    lr = lr_list[iax]
    alpha_list=np.flip(np.array([4,8,12,16]))
    colors = cmap(np.linspace(0, 1, len(alpha_list)))

    for ia,alp in enumerate(alpha_list):
        Iterations = 20000
        samples = 4096

        infidelity_file = f"./data_VMC_optimization/infidelity/final_infidelity_vmc_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
        qgt_eigvals_file = f"./data_VMC_optimization/qgt_eigenvalues/qgt_eigvals_vmc_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
        relE_file = f"./data_VMC_optimization/converged_energy/relE_vmc_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

        infidelity = np.load(infidelity_file)[0]
        qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
        qgt_logevals = np.log10(qgt_eigvals)#/qgt_eigvals.max())
        rank = len(qgt_logevals[qgt_logevals>cutoff])
        relE = np.load(relE_file)[0]


        ax0.hist(qgt_logevals, bins=bin_edges, alpha=transparency, label=rf"$\alpha={alp}$, "+r"$\mathcal{I}$"+f"={infidelity:.2e}", color=colors[-ia-1])#, edgecolor='black')
        axins.hist(qgt_logevals, bins=bin_edges, alpha=transparency, cumulative=-1,color=colors[-ia-1])#, edgecolor='black')


    ax0.set_ylim(y0,yf)
    ax0.set_xlim(x0,xf)
    ax0.set_xlabel(r"$\log_{10}(\lambda_{\rm QGT})$", fontsize=fontsize_labels)

    if iax%2==0: ax0.set_ylabel("Occurence of QGT eigenvalues", fontsize=fontsize_labels)
    ax0.legend(loc="upper right", fontsize=fontsize_legend)
    ax0.axvline(cutoff, color='k', linestyle="--")
    axins.set_ylim(y0_in,yf_in)
    axins.set_xlim(x0_in,xf_in)
    axins.axvline(cutoff, color='k', linestyle="--")
    axins.set_xlabel(r"$\log_{10}(\lambda_{\rm QGT})$", fontsize=fontsize_legend)
    axins.set_ylabel("Cumulative occ.", fontsize=fontsize_legend)



for ax in axes.flat:
    ax.tick_params(axis='both', labelsize=fontsize_legend)

plt.tight_layout()
plt.savefig("fig_9.pdf",dpi=300)

