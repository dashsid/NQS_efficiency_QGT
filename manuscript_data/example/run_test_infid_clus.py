import os
uid_run = 8103
from mpi4py import MPI
mpi = MPI.COMM_WORLD
rank = mpi.Get_rank()
size = mpi.Get_size()
os.environ["MPLCONFIGDIR"]="/dev/shm/mpl."+str(uid_run)+f"{rank:d}"
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
import netket_fidelity as nkf
#import qutip as qt
from RBM_spin1 import *
from sr_extended import ExtendedSR, cb_monitor_ExtendedSR, AdaptDiagshift2Loss
#from infidelity_optimizer_nmlr import InfidelityOptimizer
from infidelity_optimizer_lr import InfidelityOptimizer
import exchange_excitation
# 1) -----------------------------Lattice and Hilbert space within Netket----------------------------------------------------------------------

#def cb_infid(step, logdata, driver):
    #inf_MC = nkf.InfidelityOperator(vs_target_MC)
    #logdata['infid'] = driver.state.expect(inf_MC).mean


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




L = 8 
beta = 1 
seeds = [549] 
samples = 2000 # doesnt do anything for full summation 
steps = 15000 
alpha = (10,)
lr = 0.05
use_xy_basis = False
parity_symm = True
use_GPU = True 
new_hop_exch_sampler = False
infidelity_optimization = True
do_full_sum_VMC = False
n_layers = 1
keyword = "None"
ds = 1e-8
continue_previous_run = False
continue_file = "../1DAFH_MC_sampler_sz0_parity_infidL_10-iterations_15000-samples_2000-alpha_22-seedset_0-lr_0.05000/results/1DAFH_RBM_clark_MC_sz0_parity_infidl_alp22_N10_samp2000_lr0.05_ds1e-08_seed549"

if use_GPU:
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
    print("CUDA_VISIBLE_DEVICES: ", cuda_visible_devices)


# Defining the lattice within Netket
# Options for lattice: Grid,Hypercube,Cube,Square,Chain,BCC,FCC,Diamond,Pyrochlore,Triangular,Honeycomb,Kagome,KitaevHoneycomb.
g = nk.graph.Chain(length=L, pbc=False) # A 1D lattice with length L.

# Defining the Hilbert space
S=1
#Conserved
HilbertSpace = nk.hilbert.Spin(s=S, total_sz=0, N=g.n_nodes)
#constrained_states = HilbertSpace.all_states()

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
parity_antisymmetry = False

if(beta==0):
    Ham = Ham - 2/3
    model = "AFH"
    parity = True
    parity_antisymmetry = False
if(beta==1):
    Ham = Ham - 2/3 -4
    model = "LS"
    parity = True
    parity_antisymmetry = True
if(beta==2):
    Ham = Ham - 2/3 -12
    model = "gapless"
    parity = True 
    parity_antisymmetry = True

if L==12:
    parity_antisymmetry=False


# ------------------------ things for the filenaming -----------------------------
if n_layers ==1:
    file_savename="1D"+model+"_RBM_clark_MC_sz0"
else:
    file_savename="1D"+model+"_"+str(int(n_layers))+"lRBM_clark_MC_sz0"

if parity_symm:
    file_savename += "_parity"
if use_xy_basis:
    file_savename += "_xy"

file_savename += "_infid"

if len(alpha)==1:
    alpha_str = str(int(alpha[0]))
else:
    alpha_str = str(int(alpha[0]))+"-"
    for il in range(1,len(alpha)):
        alpha_str += str(int(alpha[il]))+"-"
    alpha_str = alpha_str[:-1]
# ------------------------ Useless things for the filenaming -----------------------------

# 4) ---------------------------Exact diagonalization---------------------------------------------------------------------------------------------

#evals, ket_gs = nk.exact.lanczos_ed(Ham, compute_eigenvectors=True, k=1)
evals, ket_gs = linalg.eigsh(Ham.to_sparse(), k=1, which="SA")
psi0 = ket_gs[:,0]
print('Ground state energy, calculated with Netket lanczos: ',evals[0])
if parity_symm:
    basis_states = HilbertSpace.all_states()
    spin_flipped_basis_states = -1*basis_states
    spin_flipped_indices = HilbertSpace.states_to_numbers(spin_flipped_basis_states)
    spin_flipped_psi0 = psi0[spin_flipped_indices]
    if not parity_antisymmetry:
        symmetrized_psi0 = 0.5*(psi0+spin_flipped_psi0)
    else:
        symmetrized_psi0 = 0.5*(psi0-spin_flipped_psi0)  
        #print("ANTIPARITY")          
    psi0 = symmetrized_psi0/np.sqrt(symmetrized_psi0.conj().T@symmetrized_psi0)
    Ham_matrix = Ham.to_sparse()
    energy_symm = ((psi0.conj().T)@Ham_matrix)@psi0
    print("symmetrized energy: ", energy_symm)

# 5) --------------- Exact variational state ------------------------------------
ma_target = nk.models.LogStateVector(HilbertSpace)
vs_target = nk.vqs.FullSumState(HilbertSpace, ma_target)
vs_target.parameters = {'logstate': vs_target.parameters['logstate'].at[:].set(np.log(psi0+0j))}
inf = nkf.InfidelityOperator(vs_target)
print("Testing energy: ", vs_target.expect(Ham).mean)
print("Testing infidelity: ", vs_target.expect(inf).mean)
#ansatz = RBM_clark_multilayer(alpha=4, layers=2)
#ansatz = RBM_clark_multilayer_parity_symm(alpha=(4, 4), layers=2, antisymmetry=False)
#all_states = HilbertSpace.all_states()
#ansatz = RBM_clark_parity_symm(alpha=4, antisymmetry=True)
if parity_symm:
    ansatz = RBM_clark_parity_symm(alpha=alpha[0], antisymmetry=parity_antisymmetry, sign_rule=False)
else:
    ansatz = RBM_clark(alpha=alpha[0])


# VMC part
rule1 = exchange_excitation.ExchangeExcitationRule(graph=g)
rule2 = nk.sampler.rules.ExchangeRule(graph=g)
rule = nk.sampler.rules.MultipleRules([rule1, rule2], [0.8, 0.2])
sa = nk.sampler.MetropolisSampler(HilbertSpace, rule=rule, n_chains=4096//2, n_sweeps=int(1.5*L))



if L==8: iters_VMC = 200 #int(500 + (alpha[0]*0.5)*50)

for seed in seeds:
    log_filename = "./results/"+file_savename+"l_alp"+alpha_str+"_N"+str(L)+"_samp"+str(samples)+"_lr"+str(lr)+"_ds"+str(ds)+"_seed"+str(seed)
    if keyword != "None":
        log_filename += "keyword"
    vstate = nk.vqs.FullSumState(HilbertSpace, ansatz)

    # Intialize the ansatz
    if continue_previous_run:
        # Extract variational parameters
        #filename = param_file
        with open(continue_file+".mpack", 'rb') as file:
            #vState = flax.serialization.from_state_dict(vstate, file.read())
            param = flax.serialization.msgpack_restore(file.read())
            #vState = flax.serialization.from_bytes(vstate, file.read())
        vstate.parameters = param['params']
        log_filename += "_c"
    else:    
        #vstate.init_parameters(seed=seed, init_fun = jax.nn.initializers.xavier_normal())
        vstate.init_parameters(init_fun = jax.nn.initializers.variance_scaling(scale=0.1, mode="fan_in", distribution="truncated_normal"))#print("all wavefuncs: ", vstate.to_array()) # 0.1 for adaptive lr        
        #vstate.init_parameters(seed=seed, init_fun = jax.nn.initializers.lecun_normal())#print("all wavefuncs: ", vstate.to_array())

    # VMC ansatz
    vstate_MC = nk.vqs.MCState(sa, ansatz, n_samples=4096, n_discard_per_chain=10)
    vstate_MC.init_parameters(init_fun = jax.nn.initializers.variance_scaling(scale=0.1, mode="fan_in", distribution="truncated_normal"))   
    initialize_spin1_RBM(vs = vstate_MC, L=L, HS=HilbertSpace)

    log = nk.logging.JsonLog(log_filename, mode='write', save_params=True)

    holomorphic = True
    mixed_VMC_optimization = True
    
    if mixed_VMC_optimization:
        if L==8: lr = 8e-3 # 8e-3
        optimizer = optax.sgd(learning_rate=lr)
        preconditioner = nk.optimizer.SR(diag_shift=1e-4, holomorphic=True)
        driver = nk.driver.VMC(hamiltonian=Ham, optimizer=optimizer, preconditioner=preconditioner, variational_state=vstate_MC)
        driver.run(n_iter=iters_VMC, out=log)
        vstate._parameters = vstate_MC.parameters

        print("Check infidelity (MC, FS): ", vstate.expect(inf))

        solver = nk.optimizer.solver.cholesky 
        lr = 8e-4
        lr = optax.linear_schedule(lr, 6e-4, 5000)
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
            lower_bound_diagshift = 1e-6 #p1, L=8, 1e-6 to 1e-8
        )        
        driver = nkf.driver.InfidelityOptimizer(vs_target, optimizer=optimizer, preconditioner=preconditioner, variational_state=vstate)
        driver.run(n_iter=steps, out=log)#, callback=AdaptDiagshift2Loss(min_val=ds))
log.flush(vstate)
del log
# alpha = 4: "test_RBM2sig"


