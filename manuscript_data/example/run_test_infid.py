import jax
import flax
import optax
#os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np
import netket as nk
import jax.numpy as jnp
import json
from scipy.sparse import linalg
import time
import os
import netket_fidelity as nkf
import qutip as qt
from RBM_spin1 import *
from sr_extended import ExtendedSR
import exchange_excitation

def initialize_spin1_RBM(vs, L, HS):
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
    vstate_MC._parameters["visible_bias"] = jnp.reshape(visible_biases, (L,1))
    vstate_MC._parameters["quadratic_visible_bias"] = jnp.reshape(quadratic_visible_biases, (L,1))
    vstate_MC._parameters["hidden_bias"] = 1e-4*vstate_MC.parameters["hidden_bias"]

# 1) -----------------------------Lattice and Hilbert space within Netket----------------------------------------------------------------------
#Defining the lattice within Netket
L = 8
# Options for lattice: Grid,Hypercube,Cube,Square,Chain,BCC,FCC,Diamond,Pyrochlore,Triangular,Honeycomb,Kagome,KitaevHoneycomb.
g = nk.graph.Chain(length=L, pbc=False) # A 1D lattice with length L.

# Defining the Hilbert space
S=1
#Conserved
HilbertSpace = nk.hilbert.Spin(s=S, N=g.n_nodes, total_sz=0)
constrained_states = HilbertSpace.all_states()

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
beta = 1
J = 1.

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
antisymm = False


if(beta==0):
    Ham = Ham - 2/3
    model = "AFH"
    parity = True
    antisymm = False    
    
if(beta==1):
    Ham = Ham - 2/3 -4
    model = "LS"
    parity = True
    antisymm = True
    
if(beta==2):
    Ham = Ham - 2/3 -12
    model = "gapless"
    parity = True
    antisymm = True

if L==12:
    antisymm=False 

# 4) ---------------------------Exact diagonalization---------------------------------------------------------------------------------------------

#evals, ket_gs = nk.exact.lanczos_ed(Ham, compute_eigenvectors=True,k=1)
evals, ket_gs = linalg.eigsh(
    Ham.to_sparse(), k=1, which="SA", tol=1e-32)

psi0 = ket_gs[:,0]
print('Ground state energy, calculated with Netket lanczos: ',evals[0])

if parity:
    basis_states = HilbertSpace.all_states()
    spin_flipped_basis_states = -1*basis_states
    spin_flipped_indices = HilbertSpace.states_to_numbers(spin_flipped_basis_states)
    spin_flipped_psi0 = psi0[spin_flipped_indices]
    if not antisymm:
        symmetrized_psi0 = 0.5*(psi0+spin_flipped_psi0)
    else:
        symmetrized_psi0 = 0.5*(psi0-spin_flipped_psi0)  
                
    psi0 = symmetrized_psi0/np.sqrt(symmetrized_psi0.conj().T@symmetrized_psi0)
    Ham_matrix = Ham.to_sparse()
    energy_symm = ((psi0.conj().T)@Ham_matrix)@psi0
    print("symmetrized energy: ", energy_symm)


# 5) --------------- Exact variational state ------------------------------------
ma_target = nk.models.LogStateVector(HilbertSpace)
vs_target = nk.vqs.FullSumState(HilbertSpace, ma_target)
inf = nkf.InfidelityOperator(vs_target)
vs_target.parameters = {'logstate': np.log(psi0+0j)}
print("Testing energy: ", vs_target.expect(Ham).mean)

alpha=10
ansatz = RBM_clark_parity_symm(alpha=alpha, antisymmetry=antisymm,sign_rule=False)
vstate = nk.vqs.FullSumState(HilbertSpace, ansatz)
vstate.init_parameters(seed=549, init_fun = jax.nn.initializers.xavier_normal())
print("Infidelity average: ", vstate.expect(inf).mean)

# Setting up the initial VMC part
rule1 = exchange_excitation.ExchangeExcitationRule(graph=g)
rule2 = nk.sampler.rules.ExchangeRule(graph=g)
rule = nk.sampler.rules.MultipleRules([rule1, rule2], [0.8, 0.2])
sa = nk.sampler.MetropolisSampler(HilbertSpace, rule=rule, n_chains=8, n_sweeps=int(1.5*L))

vstate_MC = nk.vqs.MCState(sa, ansatz, n_samples=1024, n_discard_per_chain=10)
vstate_MC.init_parameters(init_fun = jax.nn.initializers.variance_scaling(scale=0.1, mode="fan_in", distribution="truncated_normal"))   
initialize_spin1_RBM(vs = vstate_MC, L=L, HS=HilbertSpace)


if L==8: iters_VMC = 200 #int(500 + (alpha[0]*0.5)*50)


log = nk.logging.JsonLog(f"test_RBM_L{L:d}_alpha{alpha:d}", mode='write', save_params=True)


holomorphic = True


if L==8: lr = 8e-3 # 8e-3
# Initial VMC optimization
optimizer = optax.sgd(learning_rate=lr)
preconditioner = nk.optimizer.SR(diag_shift=1e-4, holomorphic=True)
driver = nk.driver.VMC(hamiltonian=Ham, optimizer=optimizer, preconditioner=preconditioner, variational_state=vstate_MC)
driver.run(n_iter=iters_VMC, out=log)
vstate._parameters = vstate_MC.parameters

# Infidelity optimization
print("Check infidelity (MC, FS): ", vstate.expect(inf))
solver = nk.optimizer.solver.cholesky 
lr = 8e-3
lr = optax.linear_schedule(lr, 4e-3, 5000)
optimizer = optax.sgd(learning_rate=lr)
#preconditioner = nk.optimizer.SR(solver=solver, diag_shift=1e-4, holomorphic=True)
preconditioner = ExtendedSR(
    solver=nk.optimizer.solver.cholesky,
    holomorphic=holomorphic,
    diag_shift=0,
    autotune_diag_shift_interval=5, # no. of iterations with a given diagshift
    fix_norm_curve=5.0, # fixing the norm
    constrain_gradient_update=5.0, # 10 normally
    qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=holomorphic),
    lower_bound_diagshift = 1e-8 #p1, L=8, 1e-6 to 1e-8
)        
driver = nkf.driver.InfidelityOptimizer(vs_target, optimizer=optimizer, preconditioner=preconditioner, variational_state=vstate)
driver.run(n_iter=3000, out=log)#, callback=AdaptDiagshift2Loss(min_val=ds))
log.flush(vstate)
del log

    

