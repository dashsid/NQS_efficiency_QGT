import matplotlib.pyplot as plt
import json 
import numpy as np 

fontsize_suptitle = 16+2
fontsize_title = 14+2
fontsize_labels = 13+2
fontsize_legend = 13

marker_size=12
ix = 0.27 
iy = 0.62
lx = 0.35
ly = 0.35


L=10 
#Iterations = 20000
transparency = 0.7
marker_every = 3000

colors_list= list(reversed(["firebrick", "red", "darkblue", "blue"])) #"limegreen"#"lightgreen"
symbols = list(reversed(["ks", "k^", "ko", "kd"]))
line_width = 2.5
line_width = 4

model_list = list(reversed(["AFH", "AKLT", "LS", "gapless"]))

ED_energy = {"AFH":-12.894587, "AKLT":-5.3333364, "LS":-1.7317768, "gapless":-0.490775}
ds_list = list(reversed([0.0001, 0.0001, 0.0001, 0.0001]))
lr_list = list(reversed([1e-2, 1e-2, 1e-2, 9e-3]))
model_legend_list = list(reversed([r"AFH ($\theta=0$)",r"AKLT ($\theta=\arctan(1/3)$)",r"critical ($\theta=\pi/4$)",r"critical ($\theta=\arctan(2)$)"]))
alpha_list = [4,8,12,16]
index_list = ["(a)", "(b)", "(c)", "(d)"]

fig, axes= plt.subplots(nrows=2,ncols=2,figsize=(15, 11), sharex=True,sharey=True)
flattened_axes = axes.flat
#plt.suptitle(r"VMC optimization curves for the four phases of the BLBQ chain with $L=10$", fontsize=fontsize_suptitle)

for ia,alp in enumerate(alpha_list):
    ax = flattened_axes[ia]

    for mi,mphase in enumerate(model_list):
        symb = symbols[mi]
        ds = ds_list[mi]
        lr = lr_list[mi]
        color = colors_list[mi]
        E_ED = ED_energy[mphase]

        log_file = f"./data_VMC_optimization/energy_optimization_curve/vmc_opt_model_{mphase}_L_{L:d}_alpha_{alp:d}_ds_{str(ds)}.log"
        if alp==16 and mphase=="LS":
            print(log_file)
        offset=0    
        vmc_opt_curve = np.array(json.load(open(log_file))["Energy"]["Mean"]["real"])
        vmc_opt_curve = abs(vmc_opt_curve - E_ED)/abs(E_ED)
        Iterations = len(vmc_opt_curve)
        #print(mphase, alp, len(vmc_opt_curve))
        ax.set_title(rf"{index_list[ia]} $\alpha$={alp:d}", fontsize=fontsize_title)
        ax.plot(np.arange(Iterations)+offset, vmc_opt_curve, "r-", color=color, lw=line_width, alpha=transparency)
        ax.plot(np.arange(Iterations)[::marker_every]+offset, vmc_opt_curve[::marker_every], symb, color=color, mfc="w", ms = marker_size)

flattened_axes[0].plot([],[], f"{symbols[0]}-", color=colors_list[0], label = model_legend_list[0], ms=marker_size, mfc="w", lw=line_width)
flattened_axes[0].plot([],[], f"{symbols[1]}-", color=colors_list[1], label = model_legend_list[1], ms=marker_size, mfc="w", lw=line_width)
flattened_axes[0].plot([],[], f"{symbols[2]}-", color=colors_list[2], label = model_legend_list[2], ms=marker_size, mfc="w", lw=line_width)
flattened_axes[0].plot([],[], f"{symbols[3]}-", color=colors_list[3], label = model_legend_list[3], ms=marker_size, mfc="w", lw=line_width)

flattened_axes[0].legend(fontsize = fontsize_legend-1)
flattened_axes[0].set_ylabel(r"$|\langle E\rangle - E_{\rm ED}|/|E_{\rm ED}|$", fontsize=fontsize_labels)
flattened_axes[2].set_ylabel(r"$|\langle E\rangle - E_{\rm ED}|/|E_{\rm ED}|$", fontsize=fontsize_labels)

flattened_axes[2].set_xlabel(r"Iterations", fontsize=fontsize_labels)
flattened_axes[3].set_xlabel(r"Iterations", fontsize=fontsize_labels)


for ax in axes.flat:
    ax.tick_params(axis='both', labelsize=fontsize_legend)

import matplotlib.ticker as mticker

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-1,5))
for ax in axes.flat:
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True)
    ax.set_yscale("log")
    ax.set_xlim(0,20000)
    ax.set_ylim(1e-12, 1e2)



plt.tight_layout()
plt.savefig("fig_7.pdf", dpi=300)