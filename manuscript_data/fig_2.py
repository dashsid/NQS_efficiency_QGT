import numpy as np
import matplotlib.pyplot as plt
import json


ms1 = 5+4
ms2 = 4+6
ms3 = 4+4
ms4 = 4+4

color_p1= "firebrick"#"darkblue"
color_p2= "red"#"deepskyblue"
color_p3= "darkblue"#"darkgreen"
color_p4= "blue"#"limegreen"#"lightgreen"
line_width = 2.5


fontsize_suptitle = 16+4
fontsize_title = 14+4
fontsize_labels = 13+4
fontsize_legend = 12+4

# The different phases are labelled as "AFH", "AKLT", "LS", and "gapless" for the four points marked in Fig.1 of the main text.

fig, axes = plt.subplots(1,3,figsize=(17,5))
#plt.suptitle("NQS optimization on the spin-1 Bilinear-Biquadratic chain", fontsize=fontsize_suptitle)

# --------------------Extracting data for L=10 -----------------------
final_infidelity_L08_newopt_p1 = []
final_infidelity_L08_newopt_p2 = []
final_infidelity_L08_newopt_p3 = []
final_infidelity_L08_newopt_p4 = []
L=8
alpha_list_AKLT_L08 = [2, 4, 6, 8, 10]
alpha_list_AFH_L08 = alpha_list_AKLT_L08
model = "AFH"
Iterations = 15000
for alp in alpha_list_AFH_L08:    #
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    infidelity = np.load(infidelity_file)[0]
    final_infidelity_L08_newopt_p1.append(infidelity)

model = "AKLT"
ds= 1e-6
for alp in alpha_list_AKLT_L08:    #
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    infidelity = np.load(infidelity_file)[0]
    final_infidelity_L08_newopt_p2.append(infidelity)

alpha_list_LS_L08=[2,4,6,8, 10]
model = "LS"
for alp in alpha_list_LS_L08:    #
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    infidelity = np.load(infidelity_file)[0]
    final_infidelity_L08_newopt_p3.append(infidelity)


alpha_list_gapless_L08=[ 2, 4, 6, 8,10]
model = "gapless"
for alp in alpha_list_gapless_L08:    #
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    infidelity = np.load(infidelity_file)[0]
    final_infidelity_L08_newopt_p4.append(infidelity)



# --------------------Extracting data for L=10 -----------------------
final_infidelity_L10_newopt_p1 = []
final_infidelity_L10_newopt_p2 = []
final_infidelity_L10_newopt_p3 = []
final_infidelity_L10_newopt_p4 = []
L=10
alpha_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
alpha_list_AFH = alpha_list+[22]
model = "AFH"
Iterations=15000
for alp in alpha_list+[22]:    #
    if alp==22:
        Iterations = 5000
    else:
        Iterations=15000
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    infidelity = np.load(infidelity_file)[0]
    final_infidelity_L10_newopt_p1.append(infidelity)

model = "AKLT"
Iterations=15000
for alp in alpha_list:    #
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    infidelity = np.load(infidelity_file)[0]
    final_infidelity_L10_newopt_p2.append(infidelity)

Iterations=15000
alpha_list_LS=[2, 4, 6, 8, 10, 12, 14, 16, 18]
model = "LS"
for alp in alpha_list_LS:    #
    if alp<10: 
        Iterations=15000
    else: 
        Iterations=20000
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    infidelity = np.load(infidelity_file)[0]
    final_infidelity_L10_newopt_p3.append(infidelity)

Iterations=15000
alpha_list_gapless=[ 2, 4, 6, 8,10,12, 14,16,18,20]
model = "gapless"
for alp in alpha_list_gapless:    #
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    infidelity = np.load(infidelity_file)[0]
    final_infidelity_L10_newopt_p4.append(infidelity)

# --------------------------------------------------------
final_infidelity_L12_newopt_p1 = []
final_infidelity_L12_newopt_p2 = []
final_infidelity_L12_newopt_p3 = []
final_infidelity_L12_newopt_p4 = []

L=12

Iterations=20000
alpha_list_AFH_L12 = [2, 4, 6, 8, 10, 12, 18, 20]
model = "AFH"
for alp in alpha_list_AFH_L12:    #
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    infidelity = np.load(infidelity_file)[0]
    final_infidelity_L12_newopt_p1.append(infidelity)

model = "AKLT"
alpha_list_AKLT_L12 = [2, 4, 6, 8, 12, 16, 18, 20]
for alp in alpha_list_AKLT_L12:    #
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    infidelity = np.load(infidelity_file)[0]
    final_infidelity_L12_newopt_p2.append(infidelity)

alpha_list_LS_L12=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
model = "LS"
for alp in alpha_list_LS_L12:    #
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    infidelity = np.load(infidelity_file)[0]
    final_infidelity_L12_newopt_p3.append(infidelity)


alpha_list_gapless_L12=[ 2, 6, 8,10,12,16,18,20]
model = "gapless"
for alp in alpha_list_gapless_L12:    #
    infidelity_file = f"./data_infidelity_optimization/converged_infidelity/final_infidelity_model_{model}_L_{L:d}_alpha_{alp:d}.npy"
    infidelity = np.load(infidelity_file)[0]
    final_infidelity_L12_newopt_p4.append(infidelity)



axes[0].plot(alpha_list_AFH_L08, np.abs(np.array(final_infidelity_L08_newopt_p1)), 'bs-', ms=ms1, mfc='w', color=color_p1, lw=line_width, alpha=1)#, label="AFH")
axes[0].plot(alpha_list_AKLT_L08, final_infidelity_L08_newopt_p2, 'b^-', ms=ms2, mfc='w', color=color_p2, lw=line_width, alpha=1)#, label="AKLT")
axes[0].plot(alpha_list_LS_L08, final_infidelity_L08_newopt_p3, 'go-', ms=ms3, mfc='w', color=color_p3, lw=line_width, alpha=1)#, label="critical")
axes[0].plot(alpha_list_gapless_L08, np.abs(np.array(final_infidelity_L08_newopt_p4)), 'gd-', ms=ms4, mfc='w', color=color_p4, lw=line_width, alpha=1)#, label="gapless")

axes[0].plot([], [], 'bs', ms=ms1, mfc='w', label=r"AFH ($\theta=0$)"                     , color=color_p1, lw=line_width)
axes[0].plot([], [], 'b^', ms=ms2, mfc='w', label=r"AKLT ($\theta=\arctan(1/3)$)"         , color=color_p2, lw=line_width)
axes[0].plot([], [], 'go', ms=ms3, mfc='w', label=r"critical ($\theta=\pi/4$)"      , color=color_p3, lw=line_width)
axes[0].plot([], [], 'gd', ms=ms4, mfc='w', label=r"critical ($\theta=\arctan(2)$)", color=color_p4, lw=line_width)


axes[0].set_yscale("log")
axes[0].set_title("(a) L=8", fontsize=fontsize_title)
axes[0].set_xticks(np.arange(0,21,4), size=fontsize_legend)
axes[0].set_xlim(1,21)
axes[0].grid()
axes[0].legend(fontsize=fontsize_legend, loc='upper right')
axes[0].set_ylabel(r"Infidelity ($\mathcal{I}$)", fontsize=fontsize_labels)
axes[0].set_xlabel(r"$\alpha$", fontsize=fontsize_labels)
axes[0].set_ylim(1e-13, 1e0)
#axes[0].set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])

# --------------------------------- Panel 2: L=10 ------------------------------------

axes[1].plot(alpha_list_AFH, final_infidelity_L10_newopt_p1, 'bs-', ms=ms1, mfc='w', color=color_p1, lw=line_width, alpha=1)#, label="AFH")
axes[1].plot(alpha_list, final_infidelity_L10_newopt_p2, 'b^-', ms=ms2, mfc='w', color=color_p2, lw=line_width, alpha=1)#, label="AKLT")
axes[1].plot(alpha_list_LS, final_infidelity_L10_newopt_p3, 'go-', ms=ms3, mfc='w', color=color_p3, lw=line_width, alpha=1)#, label="critical")
axes[1].plot(alpha_list_gapless, final_infidelity_L10_newopt_p4, 'gd-', ms=ms4, mfc='w', color=color_p4, lw=line_width, alpha=1)#, label="gapless")


axes[1].plot([], [], 'bs', ms=ms1, mfc='w', label=r"AFH ($\theta=0$)"                     , color=color_p1, lw=line_width)
axes[1].plot([], [], 'b^', ms=ms2, mfc='w', label=r"AKLT ($\theta=\arctan(1/3)$)"         , color=color_p2, lw=line_width)
axes[1].plot([], [], 'go', ms=ms3, mfc='w', label=r"critical ($\theta=\pi/4$)"      , color=color_p3, lw=line_width)
axes[1].plot([], [], 'gd', ms=ms4, mfc='w', label=r"critical ($\theta=\arctan(2)$)", color=color_p4, lw=line_width)

axes[1].set_yscale("log")
axes[1].set_title("(b) L=10", fontsize=fontsize_title)
axes[1].set_xticks(np.arange(4,32,4), size=fontsize_legend)
axes[1].grid()
# axes[1].legend(fontsize=fontsize_legend, loc='upper right')
#axes[1].set_ylabel("Infidelity (for different initializations)", fontsize=fontsize_labels)
axes[1].set_xlabel(r"$\alpha$", fontsize=fontsize_labels)
axes[1].set_ylim(1e-12, 1e0)
#axes[1].set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])

# ---------------------------------- Panel-3 : L=12 ---------------------------------------

axes[2].plot(alpha_list_AFH_L12, final_infidelity_L12_newopt_p1, 'bs-', ms=ms1, mfc='w', color=color_p1, lw=line_width, alpha=1)#, label="AFH")
axes[2].plot(alpha_list_AKLT_L12, final_infidelity_L12_newopt_p2, 'b^-', ms=ms2, mfc='w', color=color_p2, lw=line_width, alpha=1)#, label="AKLT")
axes[2].plot(alpha_list_LS_L12, final_infidelity_L12_newopt_p3, 'go-', ms=ms3, mfc='w', color=color_p3, lw=line_width, alpha=1)#, label="critical")
axes[2].plot(alpha_list_gapless_L12, final_infidelity_L12_newopt_p4, 'gd-', ms=ms4, mfc='w', color=color_p4, lw=line_width, alpha=1)#, label="gapless")


axes[2].set_yscale("log")
axes[2].set_title("(c) L=12", fontsize=fontsize_title)
axes[2].set_xticks(np.arange(4,25,4), fontsize=fontsize_legend)
axes[2].grid()
#axes[2].legend(fontsize=fontsize_legend)
#axes[2].set_ylabel("Infidelity (for different initializations)", fontsize=fontsize_labels)
axes[2].set_xlabel(r"$\alpha$", fontsize=fontsize_labels)
axes[2].set_ylim(1e-12, 1e0)
#axes[2].set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])


axes[2].plot([], [], 'bs', ms=ms1, mfc='w', label=r"AFH ($\beta=0$)"            , color=color_p1)
axes[2].plot([], [], 'b^', ms=ms2, mfc='w', label=r"AKLT ($\beta=1/3$)"         , color=color_p2)
axes[2].plot([], [], 'go', ms=ms3, mfc='w', label=r"critical point ($\beta=1$)" , color=color_p3)
axes[2].plot([], [], 'gd', ms=ms4, mfc='w', label=r"critical region ($\beta=2$)", color=color_p4)


for ax in axes.flat:
    ax.tick_params(axis='both', labelsize=fontsize_legend)
    
plt.tight_layout()
plt.savefig("fig_2.pdf",dpi=300)