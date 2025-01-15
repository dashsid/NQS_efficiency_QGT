# Efficiency of neural quantum states in light of the quantum geometric tensor

[![arXiv](https://img.shields.io/badge/arXiv-2402.01565-b31b1b.svg)](https://arxiv.org/abs/2402.01565)

This repository contains the datafiles for the article "Efficiency of neural quantum states in light of the quantum geometric tensor".

In this work, we characterize the performance of a shallow neural quantum state (NQS) to represent the ground state of the spin-1 Bilinear Biquadratic (BLBQ) chain. We perform a supervised learning procedure (i.e. minimizing the infidelity of an NQS w.r.t. the exact ground state), as well as the Variational Monte Carlo (VMC) procedure, to learn the ground state of the BLBQ chain. 

## Content

 - ``data_infidelity_optimization/`` contains the data files for the infidelity minimisation runs analysed in the manuscript. The data is stored in the format ...
 - ``data_VMC_optimization/`` contains the data files for the VMC optimisation data analysed in the manuscript. The data is stored in the format ...
 - ``example/`` contains some example scripts to run the VMC optimization (run_test_VMC.py) and infidelity minimization (run_test_infid.py) for the spin-1 Bilinear Biquadratic chain.
   - The exact version of sofware used at the time of publication are stored in the file ``example/requirements.txt``.
 - ``figures/`` contains the scripts to produce the plots in the manuscript.
