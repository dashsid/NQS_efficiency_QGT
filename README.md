This repository contains the datafiles for the article "Efficiency of neural quantum states in light of the quantum geometric tensor" (arXiv:2402.01565). In this work, we characterize the performance of a shallow neural quantum state (NQS) to represent the ground state of the spin-1 Bilinear Biquadratic (BLBQ) chain. We perform a supervised learning procedure (i.e. minimizing the infidelity of an NQS w.r.t. the exact ground state), as well as the Variational Monte Carlo (VMC) procedure, to learn the ground state of the BLBQ chain. 

The data files for the infidelity minimization runs are in the folder manuscript_data/data_infidelity_optimization/.

The data files for the VMC optimization runs are in the folder manuscript_data/data_VMC_optimization/.

The folder manuscript_data/example/ contains example scripts to run VMC optimization (run_test_VMC.py) and infidelity minimization (run_test_infid.py) for the spin-1 Bilinear Biquadratic chain. These scripts use the software packages https://github.com/netket/netket and https://github.com/netket/netket_fidelity.

The python scripts used to produce the plots used in the article are in the folder manuscript_data/ and are named as "fig_n.py" for nth figure in the article. 
