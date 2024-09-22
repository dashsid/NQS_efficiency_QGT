import matplotlib.pyplot as plt 
import numpy as np 
import json 

cmap = plt.cm.plasma

def compute_rank(qgt_evals, cutoff=1e-6, normalize=False):
    if not normalize:
        qgt_lgevals = np.log10(qgt_evals)#/qgt_eigvals.max())
    else:
        qgt_evals /= qgt_evals.max()
        qgt_lgevals = np.log10(qgt_evals)
    rank_ = len(qgt_lgevals[qgt_lgevals>cutoff])
    return rank_, qgt_lgevals

fontsize_suptitle = 16+4
fontsize_title = 14+4
fontsize_labels = 13+4
fontsize_legend = 12+4

ms1 = 11
ms2 = 10.5
ms3 = 9
ms4 = 8
line_width = 2
y01 = 100
yf1 = 2000

y02 = 500
yf2 = 7500

y03 = 1000
yf3 = 5000

color_p1= "firebrick" #"darkblue"
color_p2= "red" #"deepskyblue"
color_p3= "darkblue" #"darkgreen"
color_p4= "blue" #"limegreen"#"lightgreen"
line_width = 2.5

cutoff = -5
normalize_ = False
# ----------------------- Extracting QGT data for L=8 ----------------------
L=8
model="AFH"
alpha_list_p1_L08=np.flip(np.array([2,4,6,8,10]))
colors = cmap(np.linspace(0, 1, len(alpha_list_p1_L08)))
rank_list_p1_L08 = []
infidelity_list_p1_L08 = []
relE_list_p1_L08 = []
for ia,alp in enumerate(alpha_list_p1_L08):
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

    infidelity = np.load(infidelity_file)[0]
    qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
    rank, qgt_logevals = compute_rank(qgt_eigvals, cutoff=cutoff, normalize=normalize_)
    relE = np.load(relE_file)[0]

    rank_list_p1_L08.append(rank)
    infidelity_list_p1_L08.append(infidelity)
    relE_list_p1_L08.append(relE) 

#cutoff = -5
model="AKLT"
alpha_list_p2_L08=np.flip(np.array([2,4,6,8,10]))
colors = cmap(np.linspace(0, 1, len(alpha_list_p2_L08)))
rank_list_p2_L08 = []
infidelity_list_p2_L08 = []
relE_list_p2_L08 = []
for ia,alp in enumerate(alpha_list_p2_L08):
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

    infidelity = np.load(infidelity_file)[0]
    qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
    rank, qgt_logevals = compute_rank(qgt_eigvals, cutoff=cutoff, normalize=normalize_)
    relE = np.load(relE_file)[0]

    rank_list_p2_L08.append(rank)
    infidelity_list_p2_L08.append(infidelity)
    relE_list_p2_L08.append(relE) 


#cutoff=-15
model="LS"
alpha_list_p3_L08=np.flip(np.array([2,4,6,8]))
colors = cmap(np.linspace(0, 1, len(alpha_list_p3_L08)))
rank_list_p3_L08 = []
infidelity_list_p3_L08 = []
relE_list_p3_L08 = []
for ia,alp in enumerate(alpha_list_p3_L08):
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

    infidelity = np.load(infidelity_file)[0]
    qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
    rank, qgt_logevals = compute_rank(qgt_eigvals, cutoff=cutoff, normalize=normalize_)
    relE = np.load(relE_file)[0]

    rank_list_p3_L08.append(rank)
    infidelity_list_p3_L08.append(infidelity)
    relE_list_p3_L08.append(relE) 

model="gapless"
alpha_list_p4_L08=np.flip(np.array([2,4,6,8,10]))
colors = cmap(np.linspace(0, 1, len(alpha_list_p4_L08)))
rank_list_p4_L08 = []
infidelity_list_p4_L08 = []
relE_list_p4_L08 = []
for ia,alp in enumerate(alpha_list_p4_L08):
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

    infidelity = np.load(infidelity_file)[0]
    qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
    rank, qgt_logevals = compute_rank(qgt_eigvals, cutoff=cutoff, normalize=normalize_)
    relE = np.load(relE_file)[0]

    rank_list_p4_L08.append(rank)
    infidelity_list_p4_L08.append(infidelity)
    relE_list_p4_L08.append(relE) 

# ------------------- Extracting QGT data for L=10------------------------------------------------
L=10
model="AFH"
alpha_list_p1=np.flip(np.array([2,4,6,8,10,12,14,16,18,20,22]))
colors = cmap(np.linspace(0, 1, len(alpha_list_p1)))
rank_list_p1 = []
infidelity_list_p1 = []
relE_list_p1 = []
for ia,alp in enumerate(alpha_list_p1):
    if alp==22:
        Iterations=5000
    else:
        Iterations=15000
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

    infidelity = np.load(infidelity_file)[0]
    qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
    rank, qgt_logevals = compute_rank(qgt_eigvals, cutoff=cutoff, normalize=normalize_)
    relE = np.load(relE_file)[0]

    rank_list_p1.append(rank)
    infidelity_list_p1.append(infidelity)
    relE_list_p1.append(relE) 


model="AKLT"
alpha_list_p2=np.flip(np.array([2,4,6,8,10,12,14,16,18,20]))
colors = cmap(np.linspace(0, 1, len(alpha_list_p2)))
rank_list_p2 = []
infidelity_list_p2 = []
relE_list_p2 = []
for ia,alp in enumerate(alpha_list_p2):
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

    infidelity = np.load(infidelity_file)[0]
    qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
    rank, qgt_logevals = compute_rank(qgt_eigvals, cutoff=cutoff, normalize=normalize_)
    relE = np.load(relE_file)[0]

    rank_list_p2.append(rank)
    infidelity_list_p2.append(infidelity)
    relE_list_p2.append(relE) 



model="LS"
alpha_list_p3=np.flip(np.array([2,4,6,8, 10, 12, 14, 16, 18]))
colors = cmap(np.linspace(0, 1, len(alpha_list_p3)))
rank_list_p3 = []
infidelity_list_p3 = []
relE_list_p3 = []
for ia,alp in enumerate(alpha_list_p3):
    if alp>=10: Iterations=20000
    else: Iterations=15000
    
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

    infidelity = np.load(infidelity_file)[0]
    qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
    rank, qgt_logevals = compute_rank(qgt_eigvals, cutoff=cutoff, normalize=normalize_)
    relE = np.load(relE_file)[0]

    rank_list_p3.append(rank)
    infidelity_list_p3.append(infidelity)
    relE_list_p3.append(relE) 


model="gapless"
alpha_list_p4=np.flip(np.array([2,4,6,8,10,12,14,16,18,20]))
colors = cmap(np.linspace(0, 1, len(alpha_list_p4)))
rank_list_p4 = []
infidelity_list_p4 = []
relE_list_p4 = []
for ia,alp in enumerate(alpha_list_p4):
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

    infidelity = np.load(infidelity_file)[0]
    qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
    rank, qgt_logevals = compute_rank(qgt_eigvals, cutoff=cutoff, normalize=normalize_)
    relE = np.load(relE_file)[0]

    rank_list_p4.append(rank)
    infidelity_list_p4.append(infidelity)
    relE_list_p4.append(relE) 

# ---------------------------- L=12 -------------------------------------------
#cutoff = -15

L=12
model="AFH"
Iterations=20000
alpha_list_p1_L12=np.flip(np.array([2,4,6,8, 10, 12, 18, 20]))
colors = cmap(np.linspace(0, 1, len(alpha_list_p1_L12)))
rank_list_p1_L12 = []
infidelity_list_p1_L12 = []
relE_list_p1_L12 = []
for ia,alp in enumerate(alpha_list_p1_L12):
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

    infidelity = np.load(infidelity_file)[0]
    qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
    rank, qgt_logevals = compute_rank(qgt_eigvals, cutoff=cutoff, normalize=normalize_)
    relE = np.load(relE_file)[0]

    rank_list_p1_L12.append(rank)
    infidelity_list_p1_L12.append(infidelity)
    relE_list_p1_L12.append(relE) 

#cutoff = -10

model="AKLT"
alpha_list_p2_L12=np.flip(np.array([2,4,6,8, 12,16, 18, 20]))
colors = cmap(np.linspace(0, 1, len(alpha_list_p2_L12)))
rank_list_p2_L12 = []
infidelity_list_p2_L12 = []
relE_list_p2_L12 = []
for ia,alp in enumerate(alpha_list_p2_L12):
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

    infidelity = np.load(infidelity_file)[0]
    qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
    rank, qgt_logevals = compute_rank(qgt_eigvals, cutoff=cutoff, normalize=normalize_)
    relE = np.load(relE_file)[0]

    rank_list_p2_L12.append(rank)
    infidelity_list_p2_L12.append(infidelity)
    relE_list_p2_L12.append(relE) 


#cutoff = -20

model="LS"
alpha_list_p3_L12=np.flip(np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]))
colors = cmap(np.linspace(0, 1, len(alpha_list_p3_L12)))
rank_list_p3_L12 = []
infidelity_list_p3_L12 = []
relE_list_p3_L12 = []
for ia,alp in enumerate(alpha_list_p3_L12):    
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

    infidelity = np.load(infidelity_file)[0]
    qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
    rank, qgt_logevals = compute_rank(qgt_eigvals, cutoff=cutoff, normalize=normalize_)
    relE = np.load(relE_file)[0]

    rank_list_p3_L12.append(rank)
    infidelity_list_p3_L12.append(infidelity)
    relE_list_p3_L12.append(relE) 

#cutoff = -15

model="gapless"
alpha_list_p4_L12=np.flip(np.array([2,6,8,10,12, 16, 18, 20]))
colors = cmap(np.linspace(0, 1, len(alpha_list_p4_L12)))
rank_list_p4_L12 = []
infidelity_list_p4_L12 = []
relE_list_p4_L12 = []
for ia,alp in enumerate(alpha_list_p4_L12):
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"

    infidelity = np.load(infidelity_file)[0]
    qgt_eigvals = np.load(qgt_eigvals_file)[:,0]
    rank, qgt_logevals = compute_rank(qgt_eigvals, cutoff=cutoff, normalize=normalize_)
    relE = np.load(relE_file)[0]

    rank_list_p4_L12.append(rank)
    infidelity_list_p4_L12.append(infidelity)
    relE_list_p4_L12.append(relE) 


print(np.abs(np.array(infidelity_list_p1_L08)))

transp = 1
fig, axes = plt.subplots(1,3,figsize=(17,5))
#plt.suptitle("Rank of the Quantum Geometric tensor (QGT), for the spin-1 Bilinear Biquadratic chain", fontsize=fontsize_suptitle)
# ------------------- L=8 -----------------------------
axes[0].set_title("(a) L=8", fontsize=fontsize_title)
# cutoff = 1e-16
axes[0].plot(np.array(rank_list_p1_L08)/554, np.abs(np.array(infidelity_list_p1_L08)), "bs-", lw=line_width, mfc='w', alpha=transp, ms=ms1+2, color=color_p1)
axes[0].plot(np.array(rank_list_p2_L08)/554, infidelity_list_p2_L08, "b^-", lw=line_width, mfc='w', alpha=transp, ms=ms2+2, color=color_p2)
axes[0].plot(np.array(rank_list_p3_L08)/554, infidelity_list_p3_L08, "go-", lw=line_width, mfc='w', alpha=transp, ms=ms3+2, color=color_p3)
axes[0].plot(np.array(rank_list_p4_L08)/554, infidelity_list_p4_L08, "gd-", lw=line_width, mfc='w', alpha=transp, ms=ms4+2, color=color_p4)
axes[0].set_yscale("log")


#axes[0].set_xticks(np.arange(2,17,2), size=fontsize_legend)
axes[0].grid()
#axes[0].legend(fontsize=fontsize_legend)
axes[0].set_xlabel(r"$d_r/d_q$", fontsize=fontsize_labels)
axes[0].set_ylabel(r"Infidelity", fontsize=fontsize_labels)
#axes[0].set_yscale("log")
axes[0].set_ylim(1e-13, 1e0)
#axes[0].set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])

#axes[0].legend(fontsize=fontsize_legend, loc="upper left")#, framealpha=0.4)

# ------------------- L=10 ------------------------
axes[1].set_title("(b) L=10", fontsize=fontsize_title)
# cutoff = 1e-16
axes[1].plot(np.array(rank_list_p1)/4477, infidelity_list_p1, "bs-", lw=line_width, mfc='w', alpha=transp, ms=ms1+2, color=color_p1)
axes[1].plot(np.array(rank_list_p2)/4477, infidelity_list_p2, "b^-", lw=line_width, mfc='w', alpha=transp, ms=ms2+2, color=color_p2)
axes[1].plot(np.array(rank_list_p3)/4477, infidelity_list_p3, "go-", lw=line_width, mfc='w', alpha=transp, ms=ms3+2, color=color_p3)
axes[1].plot(np.array(rank_list_p4)/4477, infidelity_list_p4, "gd-", lw=line_width, mfc='w', alpha=transp, ms=ms4+2, color=color_p4)
axes[1].set_yscale("log")
# axes[1].set_xticks(np.arange(2,38,4), size=fontsize_legend)
axes[1].grid()
#axes[1].legend(fontsize=fontsize_legend, loc="upper left")
axes[1].set_xlabel(r"$d_r/d_q$", fontsize=fontsize_labels)
#axes[1].set_yscale("log")
axes[1].set_ylim(1e-12, 1e0)
# ------------------- L=12 ------------------------
axes[2].set_title("(c) L=12", fontsize=fontsize_title)
axes[2].plot(np.array(rank_list_p1_L12)/36895, np.abs(np.array(infidelity_list_p1_L12)), "bs-", lw=line_width, mfc='w', alpha=transp, ms=ms1+2, color=color_p1)
axes[2].plot(np.array(rank_list_p2_L12)/36895, infidelity_list_p2_L12, "b^-", lw=line_width, mfc='w', alpha=transp, ms=ms2+2, color=color_p2)
axes[2].plot(np.array(rank_list_p3_L12)/36895, infidelity_list_p3_L12, "go-", lw=line_width, mfc='w', alpha=transp, ms=ms3+2, color=color_p3)
axes[2].plot(np.array(rank_list_p4_L12)/36895, infidelity_list_p4_L12, "gd-", lw=line_width, mfc='w', alpha=transp, ms=ms4+2, color=color_p4)
axes[2].set_yscale("log")

# Attributes of the plot
#axes[2].set_xticks(np.arange(2,38,4), size=fontsize_legend)
axes[2].grid()
#axes[2].legend(fontsize=fontsize_legend)
axes[2].set_xlabel(r"$d_r/d_q$", fontsize=fontsize_labels)
axes[2].set_xlim(0,0.5)
#axes[2].set_yscale("log")
axes[2].set_ylim(1e-12, 1e0)

axes[2].plot([], [], "bs", mfc='w', alpha=1, ms=ms1, label=r"AFH ($\theta=0$)"                     , color=color_p1, lw=line_width)
axes[2].plot([], [], "b^", mfc='w', alpha=1, ms=ms2, label=r"AKLT ($\theta=\arctan(1/3)$)"         , color=color_p2, lw=line_width)
axes[2].plot([], [], "go", mfc='w', alpha=1, ms=ms3, label=r"critical ($\theta=\pi/4$)"      , color=color_p3, lw=line_width)
axes[2].plot([], [], "gd", mfc='w', alpha=1, ms=ms4, label=r"critical ($\theta=\arctan(2)$)", color=color_p4, lw=line_width)
axes[2].legend(fontsize=fontsize_legend)


for ax in axes.flat:
    ax.tick_params(axis='both', labelsize=fontsize_legend)
    ax.set_xlim(0.0,1.05)

plt.tight_layout()
#plt.savefig("fig_qgt_rank.pdf",dpi=300)
plt.savefig("fig_3.pdf",dpi=300)