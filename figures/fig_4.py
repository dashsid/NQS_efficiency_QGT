import matplotlib.pyplot as plt 
import numpy as np 


cmap = plt.cm.plasma
# Create a list of 8 colors from the viridis colormap
def compute_rank(qgt_evals, cutoff=1e-6, normalize=False):
    if not normalize:
        qgt_lgevals = np.log10(qgt_evals)#/qgt_eigvals.max())
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
yf = 380
x0 = -24
xf = 10
y0_in = 1
yf_in = 8000
x0_in = -24
xf_in = 10
bin_edges = np.linspace(-24, 10, 100)
transparency = 0.8

fontsize_suptitle = 16+2
fontsize_title = 14+2
fontsize_labels = 13+2
fontsize_legend = 13

rank_cutoff = -5
normalize_ = False
L=10

# AFH
fig, axes = plt.subplots(2,2,figsize=(15,11))
#plt.suptitle("Spectrum of the QGT, for spin-1 Bilinear-Biquadratic chain with L=10", fontsize=fontsize_suptitle)
ax0 = axes[0,0]
ax0.set_title(r"(a) AFH ($\theta=0$)", fontsize=fontsize_title)
axins = ax0.inset_axes([ix, iy, lx, ly])
model="AFH"
alpha_list=np.flip(np.array([2,4,6,8,10,12,14,16,18,20,22]))
colors = cmap(np.linspace(0, 1, len(alpha_list)))
for ia,alp in enumerate(alpha_list):
    if alp==22:
        iterations = 5000
    else:
        iterations = 15000
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

    infidelity = np.load(infidelity_file)[0]
    qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
    rank, qgt_logevals = compute_rank(qgt_eigvals, cutoff=rank_cutoff, normalize=normalize_)
    relE = np.load(relE_file)[0]


    ax0.hist(qgt_logevals, bins=bin_edges, alpha=transparency, label=rf"$\alpha={alp}$, "+r"$\mathcal{I}$"+f"={infidelity:.2e}", color=colors[-ia-1])#, edgecolor='black')
    axins.hist(qgt_logevals, bins=bin_edges, alpha=transparency, cumulative=-1,color=colors[-ia-1])#, edgecolor='black')
ax0.set_ylim(y0,yf)
ax0.set_xlim(x0,xf)
ax0.set_xlabel(r"$\log_{10}(\lambda_{\rm QGT})$", fontsize=fontsize_labels)
ax0.set_ylabel("Occurence of QGT eigenvalues", fontsize=fontsize_labels)
ax0.legend(loc="upper right", fontsize=fontsize_legend)
ax0.axvline(rank_cutoff, color='k', linestyle="--")
axins.set_ylim(y0_in,yf_in)
axins.set_xlim(x0_in,xf_in)
axins.axvline(rank_cutoff, color='k', linestyle="--")
axins.set_xlabel(r"$\log_{10}(\lambda_{\rm QGT})$", fontsize=fontsize_legend)
axins.set_ylabel("Cumulative occ.", fontsize=fontsize_legend)


# AKLT
ax1 = axes[0,1]
ax1.set_title(r"(b) AKLT ($\theta=\arctan(1/3)$)", fontsize=fontsize_title)
axins = ax1.inset_axes([ix, iy, lx, ly])
model="AKLT"
alpha_list=np.flip(np.array([2,4,6,8,10,12,14,16,18,20]))
colors = cmap(np.linspace(0, 1, len(alpha_list)))
for ia,alp in enumerate(alpha_list):
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

    infidelity = np.load(infidelity_file)[0]
    qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
    rank, qgt_logevals = compute_rank(qgt_eigvals, cutoff=rank_cutoff, normalize=normalize_)
    relE = np.load(relE_file)[0]


    ax1.hist(qgt_logevals, bins=bin_edges, alpha=transparency, label=rf"$\alpha={alp}$, "+r"$\mathcal{I}$"+f"={infidelity:.2e}", color=colors[-ia-1])#, edgecolor='black')
    axins.hist(qgt_logevals, bins=bin_edges, alpha=transparency, cumulative=-1,color=colors[-ia-1])#, edgecolor='black')
ax1.set_ylim(y0,yf)
ax1.set_xlim(x0,xf)
ax1.set_xlabel(r"$\log_{10}(\lambda_{\rm QGT})$", fontsize=fontsize_labels)
#ax3.set_ylabel("Distribution of QGT eigenvalues", fontsize=fontsize_labels)
ax1.legend(loc="upper right", fontsize=fontsize_legend)
ax1.axvline(rank_cutoff, color='k', linestyle="--")
axins.set_ylim(y0_in,yf_in)
axins.set_xlim(x0_in,xf_in)
axins.axvline(rank_cutoff, color='k', linestyle="--")
axins.set_xlabel(r"$\log_{10}(\lambda_{\rm QGT})$", fontsize=fontsize_legend)
axins.set_ylabel("Cumulative occ.", fontsize=fontsize_legend)

# Critical
ax2 = axes[1,0]
ax2.set_title(r"(c) critical point ($\theta=\pi/4$)", fontsize=fontsize_title)
axins = ax2.inset_axes([ix, iy, lx, ly])
model="LS"
alpha_list=np.flip(np.array([2,4,6,8, 10, 12, 14, 16, 18]))
colors = cmap(np.linspace(0, 1, len(alpha_list)))
for ia,alp in enumerate(alpha_list):
    if alp>=10: Iterations=20000
    else: Iterations=15000
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

    infidelity = np.load(infidelity_file)[0]
    qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
    rank, qgt_logevals = compute_rank(qgt_eigvals, cutoff=rank_cutoff, normalize=normalize_)
    relE = np.load(relE_file)[0]


    ax2.hist(qgt_logevals, bins=bin_edges, alpha=transparency, label=rf"$\alpha={alp}$, "+r"$\mathcal{I}$"+f"={infidelity:.2e}", color=colors[-ia-1])#, edgecolor='black')
    axins.hist(qgt_logevals, bins=bin_edges, alpha=transparency, cumulative=-1,color=colors[-ia-1])#, edgecolor='black')
ax2.set_ylim(y0,yf)
ax2.set_xlim(x0,xf)
ax2.set_xlabel(r"$\log_{10}(\lambda_{\rm QGT})$", fontsize=fontsize_labels)
ax2.set_ylabel("Occurence of QGT eigenvalues", fontsize=fontsize_labels)
ax2.legend(loc="upper right", fontsize=fontsize_legend)
ax2.axvline(rank_cutoff, color='k', linestyle="--")
axins.set_ylim(y0_in,yf_in)
axins.set_xlim(x0_in,xf_in)
axins.axvline(rank_cutoff, color='k', linestyle="--")
axins.set_xlabel(r"$\log_{10}(\lambda_{\rm QGT})$", fontsize=fontsize_legend)
axins.set_ylabel("Cumulative occ.", fontsize=fontsize_legend)

# Gapless
ax3 = axes[1,1]
ax3.set_title(r"(d) critical region ($\theta=\arctan(2)$)", fontsize=fontsize_title)
axins = ax3.inset_axes([ix, iy, lx, ly])
model="gapless"
alpha_list=np.flip(np.array([2,4,6,8,10,12,14,16,18,20]))
colors = cmap(np.linspace(0, 1, len(alpha_list)))
for ia,alp in enumerate(alpha_list):
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

    infidelity = np.load(infidelity_file)[0]
    qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
    rank, qgt_logevals = compute_rank(qgt_eigvals, cutoff=rank_cutoff, normalize=normalize_)
    relE = np.load(relE_file)[0]


    ax3.hist(qgt_logevals, bins=bin_edges, alpha=transparency, label=rf"$\alpha={alp}$, "+r"$\mathcal{I}$"+f"={infidelity:.2e}", color=colors[-ia-1])#, edgecolor='black')
    axins.hist(qgt_logevals, bins=bin_edges, alpha=transparency, cumulative=-1,color=colors[-ia-1])#, edgecolor='black')
#ax.grid()
ax3.set_ylim(y0,yf)
ax3.set_xlim(x0,xf)
ax3.set_xlabel(r"$\log_{10}(\lambda_{\rm QGT})$", fontsize=fontsize_labels)
#ax3.set_ylabel("Distribution of QGT eigenvalues", fontsize=fontsize_labels)
ax3.legend(loc="upper right", fontsize=fontsize_legend)
ax3.axvline(rank_cutoff, color='k', linestyle="--")
axins.set_ylim(y0_in,yf_in)
axins.set_xlim(x0_in,xf_in)
axins.axvline(rank_cutoff, color='k', linestyle="--")
axins.set_xlabel(r"$\log_{10}(\lambda_{\rm QGT})$", fontsize=fontsize_legend)
axins.set_ylabel("Cumulative occ.", fontsize=fontsize_legend)
#axins.set_yscale("log")
for ax in axes.flat:
    ax.tick_params(axis='both', labelsize=fontsize_legend)
plt.tight_layout()
plt.savefig("fig_4.pdf",dpi=300)

