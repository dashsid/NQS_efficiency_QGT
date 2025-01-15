import matplotlib.pyplot as plt
import numpy as np


cf_L10_alp16_p1_infid = np.load("./data_infidelity_optimization/correlation_function/s0sj_L10_alp16_p1_infid.npy")
cf_L10_alp16_p2_infid = np.load("./data_infidelity_optimization/correlation_function/s0sj_L10_alp16_p2_infid.npy")
cf_L10_alp16_p3_infid = np.load("./data_infidelity_optimization/correlation_function/s0sj_L10_alp16_p3_infid.npy")
cf_L10_alp16_p4_infid = np.load("./data_infidelity_optimization/correlation_function/s0sj_L10_alp16_p4_infid.npy")

ecf_L10_alp16_p1 = np.load("./data_infidelity_optimization/correlation_function/s0sj_L10_alp16_p1_exact.npy")
ecf_L10_alp16_p2 = np.load("./data_infidelity_optimization/correlation_function/s0sj_L10_alp16_p2_exact.npy")
ecf_L10_alp16_p3 = np.load("./data_infidelity_optimization/correlation_function/s0sj_L10_alp16_p3_exact.npy")
ecf_L10_alp16_p4 = np.load("./data_infidelity_optimization/correlation_function/s0sj_L10_alp16_p4_exact.npy")

cf_L10_alp16_p1_vmc = np.load("./data_VMC_optimization/correlation_function/s0sj_L10_alp16_p1_vmc.npy")
cf_L10_alp16_p2_vmc = np.load("./data_VMC_optimization/correlation_function/s0sj_L10_alp16_p2_vmc.npy")
cf_L10_alp16_p3_vmc = np.load("./data_VMC_optimization/correlation_function/s0sj_L10_alp16_p3_vmc.npy")
cf_L10_alp16_p4_vmc = np.load("./data_VMC_optimization/correlation_function/s0sj_L10_alp16_p4_vmc.npy")



fontsize_suptitle = 16+4
fontsize_title = 14+4
fontsize_labels = 13+4
fontsize_legend = 15

fig, axes = plt.subplots(2,2,figsize=(15,11),sharex=True,sharey=True)
ax1 = axes[0][0]
ax2 = axes[0][1]
ax3 = axes[1][0]
ax4 = axes[1][1]

marker_size=8
line_size=2

X = np.arange(1,10)
#plt.suptitle("spin-spin correlation function of the spin-1 Bilinear-Biquadratic chain with L=10", fontsize=fontsize_suptitle)
ax1.set_title(r"(a) AFH ($\theta = 0$)", fontsize=fontsize_title)
ax1.plot(X, cf_L10_alp16_p1_infid, 'bs',color='darkblue', mfc='w', label="Infidelity optimization"      ,ms=marker_size)
ax1.plot(X, cf_L10_alp16_p1_vmc  , 'r^', label="Energy optimization (VMC)"            ,ms=marker_size)
ax1.plot(X,ecf_L10_alp16_p1, 'r.--', label="exact computation (ED)", color='black',ms=marker_size, lw=line_size)
ax1.legend(fontsize=fontsize_legend)
ax1.set_ylabel(r"$\left\langle S_{0z}S_{jz}\right\rangle$", fontsize=fontsize_labels)

ax2.set_title(r"(b) AKLT ($\theta = \arctan(1/3)$)", fontsize=fontsize_title)
ax2.plot(X, cf_L10_alp16_p2_infid, 'bs',mfc='w',color='darkblue'       , ms=marker_size)
ax2.plot(X, cf_L10_alp16_p2_vmc  , 'r^'               , ms=marker_size)
ax2.plot(X,ecf_L10_alp16_p2, 'r.--', color='black', ms=marker_size, lw=line_size)


ax3.set_title(r"(c) critical point ($\theta = \pi/4$)", fontsize=fontsize_title)
ax3.plot(X, cf_L10_alp16_p3_infid, 'bs',mfc='w',color='darkblue'       , ms=marker_size)
ax3.plot(X, cf_L10_alp16_p3_vmc  , 'r^'               , ms=marker_size)
ax3.plot(X,ecf_L10_alp16_p3, 'r.--', color='black', ms=marker_size, lw=line_size)
ax3.set_ylabel(r"$\left\langle S_{0z}S_{jz}\right\rangle$", fontsize=fontsize_labels)
ax3.set_xlabel("sites ($j$)", fontsize=fontsize_labels)

ax4.set_title(r"(d) critical region ($\theta = \arctan(2)$)", fontsize=fontsize_title)
ax4.plot(X, cf_L10_alp16_p4_infid, 'bs',mfc='w',color='darkblue'       , ms=marker_size)
ax4.plot(X, cf_L10_alp16_p4_vmc  , 'r^'               , ms=marker_size)
ax4.plot(X,ecf_L10_alp16_p4, 'r.--', color='black', ms=marker_size, lw=line_size)
ax4.set_xlabel("sites ($j$)", fontsize=fontsize_labels)
for ax in axes.flat:
    ax.tick_params(axis='both', labelsize=fontsize_legend)
    ax.grid()
    ax.set_xlim(0,10)


plt.tight_layout()
plt.savefig("fig_10.pdf", dpi=300)
#plt.savefig("fig_corr_func_L10.jpg", dpi=300)










