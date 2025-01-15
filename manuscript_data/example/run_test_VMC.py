import os
#from mpi4py import MPI
import jax
import flax
import optax
import numpy as np
import netket as nk
import jax.numpy as jnp
import json
from scipy.sparse import linalg
import time
#import qutip as qt
from RBM_spin1 import *
import exchange_excitation
from sr_extended import ExtendedSR
# 1) -----------------------------Lattice and Hilbert space within Netket----------------------------------------------------------------------


def initialize_spin1_RBM(vs, L):
    init_samples = vs.samples.reshape(-1, L)
    init_samples_bool_p1 = init_samples==2
    init_samples_bool_m1 = init_samples==-2
    init_samples_bool_z = init_samples==0    
    # Initial coefficients
    total_samples = init_samples.shape[0]
    c_ze = jnp.sqrt(jnp.sum(init_samples_bool_z, axis=0)/total_samples)
    c_p1 = jnp.sqrt(jnp.sum(init_samples_bool_p1, axis=0)/total_samples)
    c_m1 = jnp.sqrt(jnp.sum(init_samples_bool_m1, axis=0)/total_samples)

    visible_biases = jnp.log(jnp.sqrt(c_p1*c_m1)/c_ze + 1e-16j)
    visible_biases = jnp.nan_to_num(visible_biases, nan=1e-4, posinf=1, neginf=-1)
    quadratic_visible_biases = jnp.log(jnp.sqrt(c_p1/c_m1) + 1e-16j)
    quadratic_visible_biases = jnp.nan_to_num(quadratic_visible_biases, nan=1e-4, posinf=1, neginf=-1)

    #resetting the parameters
    vstate._parameters["visible_bias"] = jnp.reshape(visible_biases, (L,1))
    vstate._parameters["quadratic_visible_bias"] = jnp.reshape(quadratic_visible_biases, (L,1))
    vstate._parameters["hidden_bias"] = 1e-4*vstate.parameters["hidden_bias"]




L = 8 
beta = 1 # \theta = \arctan(\beta)
samples = 2048 # 4096  
steps = 20000 
alpha = (6,)
lr = 0.01
parity_symm = True
parity_antisymmetry = True
ds = 1e-4
use_GPU =False


# Defining the lattice within Netket
g = nk.graph.Chain(length=L, pbc=False) # A 1D lattice with length L.

# Defining the Hilbert space
S=1

HilbertSpace = nk.hilbert.Spin(s=S, total_sz=0, N=g.n_nodes)

# 2) --------------------------Defining the Bilinear Biquadratic--------------------------------------------------------------------------
sz = np.array([[1,0,0],[0,0,0],[0,0,-1]], dtype=np.complex64)
sx = (1/np.sqrt(2))*np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.complex64)
sy = (1/np.sqrt(2))*np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]], dtype=np.complex64)
idt = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.complex64)
sp = np.sqrt(2)*np.array([[0,1,0],[0,0,1],[0,0,0]], dtype=np.complex64)
sm = np.sqrt(2)*np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.complex64)


tensor_operators_pm = {}
tensor_operators_pm["zz"] = np.kron(sz, sz)
tensor_operators_pm["pm"] = 0.5*np.kron(sp, sm)
tensor_operators_pm["mp"] = 0.5*np.kron(sm, sp)


Ham = 0
J = 1.0
for i in range(g.n_nodes - 1):
    #Bilinear term
    for key in tensor_operators_pm:
        Ham += J*nk.operator.LocalOperator(HilbertSpace, operators = tensor_operators_pm[key], acting_on=[i,i+1])
    #Biquadratic term
        for key2 in tensor_operators_pm:
            Ham += J*beta*nk.operator.LocalOperator(HilbertSpace, operators = tensor_operators_pm[key], acting_on=[i,i+1])*nk.operator.LocalOperator(HilbertSpace, operators = tensor_operators_pm[key2], acting_on=[i,i+1])
Ham = Ham+2/3
model = "AKLT"
parity = True 

if(beta==0):
    Ham = Ham - 2/3
    model = "AFH"
    parity = True
if(beta==1):
    Ham = Ham - 2/3 -4
    model = "LS"
    parity = True
if(beta==2):
    Ham = Ham - 2/3 -12
    model = "gapless"
    parity = True 





# 4) ---------------------------Exact diagonalization---------------------------------------------------------------------------------------------

evals, ket_gs = nk.exact.lanczos_ed(Ham, compute_eigenvectors=True, k=1)
psi0 = ket_gs[:,0]
print('Ground state energy, calculated with Netket lanczos: ',evals[0])

if use_GPU: number_of_chains = samples//2
else: number_of_chains = 4

# 5) --------------- Exact variational state ------------------------------------
rule1 = exchange_excitation.ExchangeExcitationRule(graph=g)
rule2 = nk.sampler.rules.ExchangeRule(graph=g)
rule = nk.sampler.rules.MultipleRules([rule1, rule2], [0.8, 0.2])
sa = nk.sampler.MetropolisSampler(HilbertSpace, rule=rule, n_chains=number_of_chains, n_sweeps=int(1.5*L))



ansatz = RBM_clark_parity_symm(alpha=alpha[0], antisymmetry=parity_antisymmetry, sign_rule=False)

vstate = nk.vqs.MCState(sa, ansatz, n_samples=samples, n_discard_per_chain=10)

vstate.init_parameters(init_fun = jax.nn.initializers.variance_scaling(scale=0.1, mode="fan_in", distribution="truncated_normal"))
vstate.sample(n_samples=samples, n_discard_per_chain=0)
log = nk.logging.JsonLog(f"test_VMC_RBM_L{L:d}_alpha{alpha[0]:d}", mode='write', save_params=True)
initialize_spin1_RBM(vs = vstate, L=L)

l_curve_optimization = False

lr = optax.linear_schedule(1e-2, 3e-3, 3000)
optimizer = optax.sgd(learning_rate=lr)

holomorphic = True
nksolve = nk.optimizer.solver.cholesky   
if l_curve_optimization:
    preconditioner = ExtendedSR(
        solver=nksolve,
        holomorphic=holomorphic,
        diag_shift=0,
        autotune_diag_shift_interval=5, # no. of iterations with a given diagshift
        fix_norm_curve=10, # cutoff for norm
        constrain_gradient_update=10,
        qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=holomorphic),
        lower_bound_diagshift = ds
    )
else:
    ds = optax.linear_schedule(1e-3, 1e-6, 5000)
    preconditioner = nk.optimizer.SR(diag_shift=ds, holomorphic=True)

driver = nk.driver.VMC(hamiltonian=Ham, optimizer=optimizer, preconditioner=preconditioner, variational_state=vstate)
driver.run(n_iter=steps, out=log)

log.flush(vstate)
del log


