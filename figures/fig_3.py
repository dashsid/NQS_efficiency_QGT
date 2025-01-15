import matplotlib.pyplot as plt 
import numpy as np 
import json 

cmap = plt.cm.plasma
def parameters_RBM(L, alpha):
    M = alpha*L
    params = 2*L + 2*M*L + M 
    return params

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
nparams_p1_L08 = parameters_RBM(8,alpha_list_p1_L08)
colors = cmap(np.linspace(0, 1, len(alpha_list_p1_L08)))
rank_list_p1_L08 = []
infidelity_list_p1_L08 = []
relE_list_p1_L08 = []
for ia,alp in enumerate(alpha_list_p1_L08):
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    qgt_eigvals_file = f"./data_infidelity_optimization/qgt_eigenvalues/qgt_eigvals_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    relE_file = f"./data_infidelity_optimization/energy_error/relE_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    
    # log_file = f"../../kugel/1D_spin-1_Heisenberg/massively_parallel_runs_flatiron/log_files/1D{model}_RBM_clark_MC_sz0_parity_infidl_alp{alp:d}_N{L:d}_samp2000_lr0.05_ds1e-08_seed549.log"
    # infid_opt_curve = json.load(open(log_file))["Infidelity"]["Mean"]
    # infidelity = infid_opt_curve[-1]

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
nparams_p2_L08 = parameters_RBM(8,alpha_list_p2_L08)
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
nparams_p3_L08 = parameters_RBM(8,alpha_list_p3_L08)
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
nparams_p4_L08 = parameters_RBM(8,alpha_list_p4_L08)
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
nparams_p1 = parameters_RBM(10,alpha_list_p1)
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
nparams_p2 = parameters_RBM(10,alpha_list_p2)
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
nparams_p3 = parameters_RBM(10,alpha_list_p3)
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
nparams_p4 = parameters_RBM(10,alpha_list_p4)
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
nparams_p1_L12 = parameters_RBM(12,alpha_list_p1_L12)
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
nparams_p2_L12 = parameters_RBM(12,alpha_list_p2_L12)
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
nparams_p3_L12 = parameters_RBM(12,alpha_list_p3_L12)
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
nparams_p4_L12 = parameters_RBM(12,alpha_list_p4_L12)
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
fig, axes = plt.subplots(2,3,figsize=(17,9))
print("Axes:   ", axes)
axes=axes.flatten()
#plt.suptitle("Rank of the Quantum Geometric tensor (QGT), for the spin-1 Bilinear Biquadratic chain", fontsize=fontsize_suptitle)
# ------------------- L=8 -----------------------------
axes[0].set_title("(a) L=8", fontsize=fontsize_title)
axes[3].set_title("(d) L=8", fontsize=fontsize_title)
# cutoff = 1e-16
axes[0].plot(np.array(alpha_list_p1_L08), np.array(rank_list_p1_L08), "bs-", lw=line_width, mfc='w', alpha=transp, ms=ms1+2, color=color_p1)
axes[0].plot(np.array(alpha_list_p2_L08), np.array(rank_list_p2_L08), "b^-", lw=line_width, mfc='w', alpha=transp, ms=ms2+2, color=color_p2)
axes[0].plot(np.array(alpha_list_p3_L08), np.array(rank_list_p3_L08), "go-", lw=line_width, mfc='w', alpha=transp, ms=ms3+2, color=color_p3)
axes[0].plot(np.array(alpha_list_p4_L08), np.array(rank_list_p4_L08), "gd-", lw=line_width, mfc='w', alpha=transp, ms=ms4+2, color=color_p4)

axes[3].plot(np.array(rank_list_p1_L08)/554, np.abs(np.array(infidelity_list_p1_L08)), "bs-", lw=line_width, mfc='w', alpha=transp, ms=ms1+2, color=color_p1)
axes[3].plot(np.array(rank_list_p2_L08)/554, infidelity_list_p2_L08, "b^-", lw=line_width, mfc='w', alpha=transp, ms=ms2+2, color=color_p2)
axes[3].plot(np.array(rank_list_p3_L08)/554, infidelity_list_p3_L08, "go-", lw=line_width, mfc='w', alpha=transp, ms=ms3+2, color=color_p3)
axes[3].plot(np.array(rank_list_p4_L08)/554, infidelity_list_p4_L08, "gd-", lw=line_width, mfc='w', alpha=transp, ms=ms4+2, color=color_p4)
axes[3].set_yscale("log")

axins = axes[0].inset_axes([0.17, 0.57, 0.35, 0.35])
axins.plot(np.array(alpha_list_p1_L08), np.array(rank_list_p1_L08)/nparams_p1_L08, "bs-", lw=line_width, mfc='w', alpha=transp, ms=ms1, color=color_p1)
axins.plot(np.array(alpha_list_p2_L08), np.array(rank_list_p2_L08)/nparams_p2_L08, "b^-", lw=line_width, mfc='w', alpha=transp, ms=ms2, color=color_p2)
axins.plot(np.array(alpha_list_p3_L08), np.array(rank_list_p3_L08)/nparams_p3_L08, "go-", lw=line_width, mfc='w', alpha=transp, ms=ms3, color=color_p3)
axins.plot(np.array(alpha_list_p4_L08), np.array(rank_list_p4_L08)/nparams_p4_L08, "gd-", lw=line_width, mfc='w', alpha=transp, ms=ms4, color=color_p4)
axins.grid()

#axins.set_yticks([0.25,0.5,0.75], size=fontsize_labels)
#axins.set_xticks(np.arange(2,10,4), size=fontsize_labels)
#axins.set_ylim(0, 1.1)
axins.set_xlabel(r"$\alpha$", fontsize=fontsize_labels-2)
axins.set_ylabel(r"$d_r/N_p$", fontsize=fontsize_labels-2)


#axes[0].set_yscale("log")

# # Dummy legends
# axes[0].plot([], [], "bs", mfc='w', alpha=1, ms=ms1, label=r"AFH ($\theta=0$)"                     , color=color_p1, lw=line_width)
# axes[0].plot([], [], "b^", mfc='w', alpha=1, ms=ms2, label=r"AKLT ($\theta=\arctan(1/3)$)"         , color=color_p2, lw=line_width)
# axes[0].plot([], [], "go", mfc='w', alpha=1, ms=ms3, label=r"critical point ($\theta=\pi/4$)"      , color=color_p3, lw=line_width)
# axes[0].plot([], [], "gd", mfc='w', alpha=1, ms=ms4, label=r"critical region ($\theta=\arctan(2)$)", color=color_p4, lw=line_width)


# Attributes of the plot

#axes[0].set_xticks(np.arange(2,17,2), size=fontsize_legend)
#axes[0].grid()
#axes[0].legend(fontsize=fontsize_legend)
axes[0].set_xlabel(r"$\alpha$", fontsize=fontsize_labels)
axes[0].set_ylabel(r"Rank of the QGT ($d_r$)", fontsize=fontsize_labels)
#axes[0].set_yscale("log")
#axes[0].set_ylim(1e-13, 1e0)
#axes[0].set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])

#axes[0].legend(fontsize=fontsize_legend, loc="upper left")#, framealpha=0.4)

# ------------------- L=10 ------------------------
axes[1].set_title("(b) L=10", fontsize=fontsize_title)
axes[4].set_title("(e) L=10", fontsize=fontsize_title)
# cutoff = 1e-16
axes[1].plot(np.array(alpha_list_p1), np.array(rank_list_p1), "bs-", lw=line_width, mfc='w', alpha=transp, ms=ms1+2, color=color_p1)
axes[1].plot(np.array(alpha_list_p2), np.array(rank_list_p2), "b^-", lw=line_width, mfc='w', alpha=transp, ms=ms2+2, color=color_p2)
axes[1].plot(np.array(alpha_list_p3), np.array(rank_list_p3), "go-", lw=line_width, mfc='w', alpha=transp, ms=ms3+2, color=color_p3)
axes[1].plot(np.array(alpha_list_p4), np.array(rank_list_p4), "gd-", lw=line_width, mfc='w', alpha=transp, ms=ms4+2, color=color_p4)

axes[4].plot(np.array(rank_list_p1)/4477, infidelity_list_p1, "bs-", lw=line_width, mfc='w', alpha=transp, ms=ms1+2, color=color_p1)
axes[4].plot(np.array(rank_list_p2)/4477, infidelity_list_p2, "b^-", lw=line_width, mfc='w', alpha=transp, ms=ms2+2, color=color_p2)
axes[4].plot(np.array(rank_list_p3)/4477, infidelity_list_p3, "go-", lw=line_width, mfc='w', alpha=transp, ms=ms3+2, color=color_p3)
axes[4].plot(np.array(rank_list_p4)/4477, infidelity_list_p4, "gd-", lw=line_width, mfc='w', alpha=transp, ms=ms4+2, color=color_p4)
axes[4].set_yscale("log")

#axes[1].set_yscale("log")
axins = axes[1].inset_axes([0.17, 0.57, 0.35, 0.35])
axins.plot(np.array(alpha_list_p1), np.array(rank_list_p1)/nparams_p1, "bs-", lw=line_width, mfc='w', alpha=transp, ms=ms1, color=color_p1)
axins.plot(np.array(alpha_list_p2), np.array(rank_list_p2)/nparams_p2, "b^-", lw=line_width, mfc='w', alpha=transp, ms=ms2, color=color_p2)
axins.plot(np.array(alpha_list_p3), np.array(rank_list_p3)/nparams_p3, "go-", lw=line_width, mfc='w', alpha=transp, ms=ms3, color=color_p3)
axins.plot(np.array(alpha_list_p4), np.array(rank_list_p4)/nparams_p4, "gd-", lw=line_width, mfc='w', alpha=transp, ms=ms4, color=color_p4)
axins.grid()

#axins.set_yticks([0.25,0.5,0.75], size=fontsize_labels)
#axins.set_xticks(np.arange(2,20,4), size=fontsize_labels)
#axins.set_ylim(0, 1.1)
axins.set_xlabel(r"$\alpha$", fontsize=fontsize_labels-2)
axins.set_ylabel(r"$d_r/N_p$", fontsize=fontsize_labels-2)

# Attributes of the plot
# axes[1].set_xticks(np.arange(2,38,4), size=fontsize_legend)
#axes[1].grid()
#axes[1].legend(fontsize=fontsize_legend, loc="upper left")
axes[1].set_xlabel(r"$\alpha$", fontsize=fontsize_labels)
#axes[1].set_yscale("log")
#axes[1].set_ylim(1e-12, 1e0)
# ------------------- L=12 ------------------------
axes[2].set_title("(c) L=12", fontsize=fontsize_title)
axes[5].set_title("(f) L=12", fontsize=fontsize_title)
axes[2].plot(np.array(alpha_list_p1_L12), np.array(rank_list_p1_L12), "bs-", lw=line_width, mfc='w', alpha=transp, ms=ms1+2, color=color_p1)
axes[2].plot(np.array(alpha_list_p2_L12), np.array(rank_list_p2_L12), "b^-", lw=line_width, mfc='w', alpha=transp, ms=ms2+2, color=color_p2)
axes[2].plot(np.array(alpha_list_p3_L12), np.array(rank_list_p3_L12), "go-", lw=line_width, mfc='w', alpha=transp, ms=ms3+2, color=color_p3)
axes[2].plot(np.array(alpha_list_p4_L12), np.array(rank_list_p4_L12), "gd-", lw=line_width, mfc='w', alpha=transp, ms=ms4+2, color=color_p4)
#axes[2].set_yscale("log")
axes[5].plot(np.array(rank_list_p1_L12)/36895, np.abs(np.array(infidelity_list_p1_L12)), "bs-", lw=line_width, mfc='w', alpha=transp, ms=ms1+2, color=color_p1)
axes[5].plot(np.array(rank_list_p2_L12)/36895, infidelity_list_p2_L12, "b^-", lw=line_width, mfc='w', alpha=transp, ms=ms2+2, color=color_p2)
axes[5].plot(np.array(rank_list_p3_L12)/36895, infidelity_list_p3_L12, "go-", lw=line_width, mfc='w', alpha=transp, ms=ms3+2, color=color_p3)
axes[5].plot(np.array(rank_list_p4_L12)/36895, infidelity_list_p4_L12, "gd-", lw=line_width, mfc='w', alpha=transp, ms=ms4+2, color=color_p4)
axes[5].set_yscale("log")

axins = axes[2].inset_axes([0.17, 0.57, 0.35, 0.35])
axins.plot(np.array(alpha_list_p1_L12), np.array(rank_list_p1_L12)/nparams_p1_L12, "bs-", lw=line_width, mfc='w', alpha=transp, ms=ms1, color=color_p1)
axins.plot(np.array(alpha_list_p2_L12), np.array(rank_list_p2_L12)/nparams_p2_L12, "b^-", lw=line_width, mfc='w', alpha=transp, ms=ms2, color=color_p2)
axins.plot(np.array(alpha_list_p3_L12), np.array(rank_list_p3_L12)/nparams_p3_L12, "go-", lw=line_width, mfc='w', alpha=transp, ms=ms3, color=color_p3)
axins.plot(np.array(alpha_list_p4_L12), np.array(rank_list_p4_L12)/nparams_p4_L12, "gd-", lw=line_width, mfc='w', alpha=transp, ms=ms4, color=color_p4)
axins.grid()

#axins.set_yticks([0.4,0.6,0.8], size=fontsize_labels)
#axins.set_xticks(np.arange(2,20,4), size=fontsize_labels)
#axins.set_ylim(0, 1.1)
axins.set_xlabel(r"$\alpha$", fontsize=fontsize_labels-2)
axins.set_ylabel(r"$d_r/N_p$", fontsize=fontsize_labels-2)

# Attributes of the plot
#axes[2].set_xticks(np.arange(2,38,4), size=fontsize_legend)
#axes[2].grid()
#axes[2].legend(fontsize=fontsize_legend)
#axes[2].set_ylabel(r"Rank of QGT $d_r$", fontsize=fontsize_labels)
axes[2].set_xlabel(r"$\alpha$", fontsize=fontsize_labels)
#axes[2].set_yscale("log")
#axes[2].set_ylim(1e-12, 1e0)

axes[4].plot([], [], "bs", mfc='w', alpha=1, ms=ms1, label=r"AFH ($\theta=0$)"                     , color=color_p1, lw=line_width)
axes[4].plot([], [], "b^", mfc='w', alpha=1, ms=ms2, label=r"AKLT ($\theta=\arctan(1/3)$)"         , color=color_p2, lw=line_width)
axes[4].plot([], [], "go", mfc='w', alpha=1, ms=ms3, label=r"critical ($\theta=\pi/4$)"      , color=color_p3, lw=line_width)
axes[4].plot([], [], "gd", mfc='w', alpha=1, ms=ms4, label=r"critical ($\theta=\arctan(2)$)", color=color_p4, lw=line_width)
axes[4].legend(fontsize=fontsize_legend)

# cutoff = 1e-16
# axes[2].plot([4, 8, 12, 16], [rank16_qgt_logevals_p1_L12_alpha04_infid, rank16_qgt_logevals_p1_L12_alpha08_infid, rank16_qgt_logevals_p1_L12_alpha12_infid, rank16_qgt_logevals_p1_L12_alpha16_infid], "bs-", mfc="w", alpha=transp, ms=ms1+2, color=color_p1, lw=line_width)
# axes[2].plot([4, 8, 12, 16], [rank16_qgt_logevals_p2_L12_alpha04_infid, rank16_qgt_logevals_p2_L12_alpha08_infid, rank16_qgt_logevals_p2_L12_alpha12_infid, rank16_qgt_logevals_p2_L12_alpha16_infid], "b^-", mfc="w", alpha=transp, ms=ms2+2, color=color_p2, lw=line_width)
# axes[2].plot([4, 8, 12, 16], [rank16_qgt_logevals_p3_L12_alpha04_infid, rank16_qgt_logevals_p3_L12_alpha08_infid, rank16_qgt_logevals_p3_L12_alpha12_infid, rank16_qgt_logevals_p3_L12_alpha16_infid], "go-", mfc="w", alpha=transp, ms=ms3+2, color=color_p3, lw=line_width)
# axes[2].plot([4, 8, 12, 16], [rank16_qgt_logevals_p4_L12_alpha04_infid, rank16_qgt_logevals_p4_L12_alpha08_infid, rank16_qgt_logevals_p4_L12_alpha12_infid, rank16_qgt_logevals_p4_L12_alpha16_infid], "gd-", mfc="w", alpha=transp, ms=ms4+2, color=color_p4, lw=line_width)

# Dummy legends

for iax,ax in enumerate(axes.flat):
    ax.tick_params(axis='both', labelsize=fontsize_legend)
    if iax<3 and iax>0:
        ax.set_ylim(100,6000)
    elif iax>2:
        ax.set_ylim(1e-13, 1e0)
        ax.set_xlabel(r"$d_r/d_q$", fontsize=fontsize_labels)
    ax.grid()
axes[0].set_ylim(100,1100)
axes[3].set_ylabel(r"Infidelity ($\mathcal{I}$)", fontsize=fontsize_labels)

plt.tight_layout()
#plt.savefig("fig_qgt_rank.pdf",dpi=300)
plt.savefig("fig_3.pdf",dpi=300)

