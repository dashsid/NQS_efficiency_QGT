import matplotlib.pyplot as plt 
import numpy as np 


cmap = plt.cm.plasma
def compute_rank(qgt_evals, cutoff=1e-6, normalize=False):
    if not normalize:
        qgt_lgevals = np.log10(qgt_evals)#/qgt_eigvals.max())
    else:
        qgt_evals /= qgt_evals.max()
        qgt_lgevals = np.log10(qgt_evals)
    rank_ = len(qgt_lgevals[qgt_lgevals>cutoff])
    return rank_, qgt_lgevals

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

transparency = 0.8

fontsize_suptitle = 16+2
fontsize_title = 14+2
fontsize_labels = 13+2
fontsize_legend = 13

cutoff = -5
normalize_ = False
L=10

legend_list = [r"AFH ($\theta=0$)", r"AKLT ($\theta=\arctan(1/3)$)", r"critical ($\theta=\pi/4$)", r"critical ($\theta=\arctan(2)$)"]
model_list = ["AFH", "AKLT", "LS", "gapless"]
marker_list = ["s-", "^-", "o-", "d-"]
colors_list= ["firebrick", "red", "darkblue", "blue"]
lr_list = [1e-2, 1e-2, 1e-2, 9e-3]
alpha_list = np.array([4,8,12,16])

rank_dict={}
relE_dict={}
infi_dict={}

fig, axes = plt.subplots(1,2,figsize=(13,5))
#plt.suptitle("Infidelity and Rank of the QGT for VMC solutions on the spin-1 Bilinear-Biquadratic chain with L=10", fontsize=fontsize_suptitle)
line_width = 2.5
marker_size = 9

for im, mphase in enumerate(model_list):

    model = model_list[im]
    lr = lr_list[im]
    marker = marker_list[im]
    colors = colors_list[im]
    legend=legend_list[im]
    #colors = cmap(np.linspace(0, 1, len(alpha_list)))

    rank_list = []
    relE_list = []
    infi_list = []
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

        rank_list.append(rank)
        infi_list.append(infidelity)
        relE_list.append(relE) 

    rank_dict[mphase] = np.array(rank_list)
    infi_dict[mphase] = np.array(infi_list)
    relE_dict[mphase] = np.array(relE_list)

    axes[0].plot(alpha_list,        infi_dict[mphase],marker, color=colors, label=legend,lw=line_width,ms=marker_size, mfc='w')
    axes[1].plot(rank_dict[mphase]/4477, infi_dict[mphase],marker, color=colors,lw=line_width,ms=marker_size, mfc='w')
    
axes[0].set_title(r"(a) Infidelity ($\mathcal{I}$) ~ $\alpha$", fontsize=fontsize_title)
axes[1].set_title(r"(b) Infidelity ($\mathcal{I}$) ~ $d_r/d_q$", fontsize=fontsize_title)
axes[0].set_ylabel(r"Infidelity ($\mathcal{I}$)", fontsize=fontsize_labels)
axes[0].legend(fontsize=fontsize_legend)
axes[0].set_xlabel(r"$\alpha$", fontsize=fontsize_labels)
axes[1].set_xlabel(r"$d_r/d_q$", fontsize=fontsize_labels)
for ax in axes.flat:
    ax.set_yscale("log")
    ax.tick_params(axis='both', labelsize=fontsize_legend)
    ax.grid()

plt.tight_layout()
plt.savefig("fig_5.pdf",dpi=300)

#plt.show()
#plt.clf()
