import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import netket as nk 
import jax 
verbose = False
datatype = jnp.complex128
initializer = nn.initializers.normal()



class RBM_clark_parity_symm(nn.Module): 
    alpha: int # alpha*L is the number of hidden spins.
    antisymmetry : bool
    sign_rule: bool
    @nn.compact # Decorator for jit
    def __call__(self, s):# s is the input : 1D array of spins (could also be 2D)
        if verbose: print("Shape of input: ", s.shape)# (..., L)
        n_visible_units = s.shape[-1] # by default s is taken as a row vector
        n_hidden_units = int(self.alpha*n_visible_units) # No.of spins in the hidden layer
        # define variational parameters
        # the arguments are: name, initializer, shape, dtype
        W = self.param("weights", initializer, (n_hidden_units, n_visible_units), datatype) # (alpha*L, L)
        qW = self.param("quadratic_weights", initializer, (n_hidden_units, n_visible_units), datatype) # (alpha*L, L)
        if verbose: print("Shape of W, qW: ", W.shape, qW.shape)
        b = self.param("hidden_bias", initializer, (1, n_hidden_units), datatype)
        if verbose: print("Shape of b: ", b.shape) # (1, alpha*L)
        a = self.param("visible_bias", initializer, (n_visible_units, 1), datatype) # (L, 1)
        qa = self.param("quadratic_visible_bias", initializer, (n_visible_units, 1), datatype) # (L, 1)
        
        # Parity symmetry
        psi_parity = []
        for par in range(2):
            sp = ((-1)**par)*s
            y = jnp.einsum("ij,...j", W, sp)
            y += jnp.einsum("ij,...j", qW, jnp.square(sp))
            if verbose: print("Shape of y after acting s with W: ", y.shape) # (1, alpha*L)
            y += b # the linear transformation ends here

            if verbose: print("Shape of y after adding with hidden bias", y.shape)
            y = jnp.sum(nk.nn.activation.log_cosh(y), axis = -1) # apply the non-linear function log(cosh(y))

            if verbose: print("Shape of y after summing over hidden units", y.shape)
            #y = y.reshape((sp.shape[0],1))
            if verbose: print("Shape of y after correction", y.shape)
            y += jnp.einsum( "...k,k...", sp, a)
            y += jnp.einsum( "...k,k...", jnp.square(sp), qa)

            if verbose: print("Shape of y after second correction", y.shape)
            if verbose: print("Shape of y after adding s.a", y.shape)
            msign = self.marshall(sp)
            if not self.sign_rule:
                psi_parity.append(y)
            else:
                #print("Shape of y: ", y.shape)
                #print("Shape of msigns: ", msign.shape)
                psi_parity.append(y+1j*msign)

        psi_parity = jnp.asarray(psi_parity)
        if not self.antisymmetry:
            log_psi_parity = jax.scipy.special.logsumexp(psi_parity, b=0.5, axis=0)
        else:
            #bc = np.random.random(psi_parity.shape)+1j*np.random.random(psi_parity.shape)
            bc = 0.5*np.ones(psi_parity.shape, dtype=complex)
            bc[1,:] = -1*bc[1,:]
            #print(bc)
            log_psi_parity = jax.scipy.special.logsumexp(psi_parity, b=bc+1j*1e-16, axis=0)
        return jnp.reshape(log_psi_parity, y.shape) 
    
    def marshall(self,x):
        lchain = x.shape[-1]
        neel_signs = np.arange(lchain)*(np.pi)
        #print("shape of x: ", x.shape)
        #print("shape of signs: ", neel_signs.shape)
        return jnp.matmul(x,neel_signs)





