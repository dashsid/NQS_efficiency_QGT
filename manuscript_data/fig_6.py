import matplotlib.pyplot as plt
import json 
import numpy as np 


fontsize_suptitle = 18+2+3
fontsize_title = 16+2+2
fontsize_labels = 16+2+2
fontsize_legend = 14+2
marker_size=12
ix = 0.27 
iy = 0.62
lx = 0.35
ly = 0.35

y0_in = 1e-6
yf_in = 1e-2
L=10 
Iterations = 15000
transparency = 0.7
marker_every = 3000

colors_list= ["firebrick", "red", "darkblue", "blue"] #"limegreen"#"lightgreen"
symbols = ["ks", "k^", "ko", "kd"]
line_width = 2.5
line_width = 4

model_list = ["AFH", "AKLT", "LS", "gapless"]
ds_list = [1e-8, 1e-8, 1e-8, 1e-8]
model_legend_list = [r"AFH ($\theta=0$)",r"AKLT ($\theta=\arctan(1/3)$)",r"critical ($\theta=\pi/4$)",r"critical ($\theta=\arctan(2)$)"]
alpha_list = [2,4,6,8,10,12,14,16,18,20]
fig, axes= plt.subplots(nrows=2,ncols=5,figsize=(20, 10), sharex=True,sharey=True)
flattened_axes = axes.flat
#plt.suptitle(r"Infidelity minimization curves for the four phases of the BLBQ chain with $L=10$", fontsize=fontsize_suptitle)
index_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)"]
for ia,alp in enumerate(alpha_list):
    ax = flattened_axes[ia]
    indl = index_labels[ia]
    for mi,mphase in enumerate(model_list):
        symb = symbols[mi]
        if mphase == "LS" and alp==20:
            continue
        color = colors_list[mi]
        if mphase=="LS" and alp>=10:
            ds = 1e-3
        else:
            ds = ds_list[mi]

        log_file = f"./data_infidelity_optimization/infidelity_optimization_curve/infid_opt_model_{mphase}_L_{L:d}_alpha_{alp:d}_ds_{str(ds)}.log"
        offset=0
        if alp==22 and mphase=="AFH":
            log_file1 = f"./data_infidelity_optimization/infidelity_optimization_curve/infid_opt_model_{mphase}_L_{L:d}_alpha_{alp:d}_ds_{str(ds)}_1.log"
            infid_opt_curve = json.load(open(log_file1))["Infidelity"]["Mean"]
            ax.plot(infid_opt_curve,color=color)
            ax.plot(infid_opt_curve[::marker_every], symb,color=color, mfc="w", ms = marker_size)
            offset =len(infid_opt_curve)

            log_file2 = f"./data_infidelity_optimization/infidelity_optimization_curve/infid_opt_model_{mphase}_L_{L:d}_alpha_{alp:d}_ds_{str(ds)}_2.log"
            infid_opt_curve = json.load(open(log_file2))["Infidelity"]["Mean"]
            Iterations = len(infid_opt_curve)
        else:
            offset=0
            infid_opt_curve = json.load(open(log_file))["Infidelity"]["Mean"]
            Iterations = len(infid_opt_curve)
        #print(mphase, alp, len(infid_opt_curve))
        ax.set_title(rf"{indl} $\alpha$={alp:d}", fontsize=fontsize_title)
        ax.plot(np.arange(Iterations)+offset, infid_opt_curve, "r-", color=color, lw=line_width, alpha=transparency)
        ax.plot(np.arange(Iterations)[::marker_every]+offset, infid_opt_curve[::marker_every], symb, color=color, mfc="w", ms = marker_size)

flattened_axes[0].plot([],[], f"{symbols[0]}-", color=colors_list[0], label = model_legend_list[0], ms=marker_size, mfc="w", lw=line_width)
flattened_axes[0].plot([],[], f"{symbols[1]}-", color=colors_list[1], label = model_legend_list[1], ms=marker_size, mfc="w", lw=line_width)
flattened_axes[0].plot([],[], f"{symbols[2]}-", color=colors_list[2], label = model_legend_list[2], ms=marker_size, mfc="w", lw=line_width)
flattened_axes[0].plot([],[], f"{symbols[3]}-", color=colors_list[3], label = model_legend_list[3], ms=marker_size, mfc="w", lw=line_width)
flattened_axes[0].legend(fontsize = fontsize_legend-1)
flattened_axes[0].set_ylabel(r"Infidelity ($\mathcal{I}$)", fontsize=fontsize_labels)
flattened_axes[5].set_ylabel(r"Infidelity ($\mathcal{I}$)", fontsize=fontsize_labels)

flattened_axes[5].set_xlabel(r"Iterations", fontsize=fontsize_labels)
flattened_axes[6].set_xlabel(r"Iterations", fontsize=fontsize_labels)
flattened_axes[7].set_xlabel(r"Iterations", fontsize=fontsize_labels)
flattened_axes[8].set_xlabel(r"Iterations", fontsize=fontsize_labels)
flattened_axes[9].set_xlabel(r"Iterations", fontsize=fontsize_labels)


for ax in axes.flat:
    ax.tick_params(axis='both', labelsize=fontsize_legend)

import matplotlib.ticker as mticker

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-1,5))
for ax in axes.flat:
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True)
    ax.set_yscale("log")



plt.tight_layout()
plt.savefig("fig_6.pdf", dpi=300)

