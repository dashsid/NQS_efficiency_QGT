from functools import partial
import jax
from netket.utils import timing, struct
from netket.utils.types import Scalar, PyTree, ScalarOrSchedule
from netket.optimizer import SR, solver as nksolver
from netket.vqs import VariationalState
from netket.jax import tree_ravel
import netket as nk
from jax import numpy as jnp
from jax.tree_util import Partial
from typing import Optional, Callable
from jax.tree_util import tree_map
import numpy as np 

from L_curve import (
    compute_l_curve_coordinates,
    auto_tikhonov_solver,
)

αlist = jnp.geomspace(1e-13, 1, 20)


class ExtendedSR(SR, mutable=True):
    r"""
    Extended version of the :class:`netket.optimizer.SR` preconditioner that allows for autotuning of the diag_shift parameter.

    This preconditioner has the following additional features:

        - Autotuning of the diag_shift parameter every ``autotune_diag_shift_interval`` steps by using the L-curve criterion
          (see :func:`netket_pro.optimizer.L_curve.auto_tikhonov_solver`). This is fairly expensive as it involves solving
          the linear system multiple times, so pick a reasonable interval.
        - Fixing the euclidean norm of the resulting gradient update to ``fix_norm_curve``. Effectively this rescales the
          resulting update to be :math:`\Delta \theta = \Delta \theta \frac{\text{fix_norm_curve}}{\|\Delta \theta\|}`.

    """

    _autotune_diag_shift_interval: Optional[int]
    _constrain_gradient_update: Optional[int]
    _fix_norm_curve: Optional[float]
    _norm: float
    _reimann_norm: float
    _lower_bound_diagshift : Optional[float]

    def __init__(
        self,
        qgt: Optional[Callable] = None,
        solver: Callable = nksolver.cholesky,
        *,
        diag_shift: ScalarOrSchedule = 0.0,
        diag_scale: Optional[ScalarOrSchedule] = None,
        autotune_diag_shift_interval: Optional[int] = None,
        fix_norm_curve: Optional[float] = None,
        constrain_gradient_update: Optional[int] = None,
        lower_bound_diagshift = None,
        **kwargs,
    ):
        """
        Constructs the extended SR preconditioner.

        Args:
            qgt: Quantum Geometric Tensor.
            solver: Solver to use for the linear system.
            diag_shift: Diagonal shift to add to the S matrix.
            diag_scale: Scaling factor to apply to the S matrix (should be 0)
            autotune_diag_shift_interval: Interval at which to autotune the diag_shift parameter.
            fix_norm_curve: Choose the diag_shift parameter such that the norm of the gradient update is fixed to this value
            constrain_gradient_update: Maximum norm of the gradient update.
        """

        super().__init__(
            qgt=qgt,
            solver=solver,
            diag_shift=diag_shift,
            diag_scale=diag_scale,
            **kwargs,
        )

        self._autotune_diag_shift_interval = autotune_diag_shift_interval
        self._constrain_gradient_update = constrain_gradient_update
        self._fix_norm_curve = fix_norm_curve
        self._norm = 0.0
        self._reimann_norm = 0.0
        self._lower_bound_diagshift = lower_bound_diagshift

    def concretize_solver(self, step):
        solver = self.solver
        if isinstance(solver, partial):
            # Extract the original function and keyword arguments
            func = solver.func
            keywords = solver.keywords.copy()  # Copy to avoid mutating the original

            for key, value in keywords.items():
                if callable(value):
                    keywords[key] = value(step)

            # Create a new partial function with the updated arguments
            solver = Partial(func, **keywords)
        return solver

    @timing.timed
    def __call__(
        self,
        vstate: VariationalState,
        gradient: PyTree,
        step: Optional[Scalar] = None,
        *args,
        **kwargs,
    ) -> PyTree:
        solver = self.concretize_solver(step)

        if (self._autotune_diag_shift_interval is None) or (
            step % self._autotune_diag_shift_interval != 0
        ):
            self._lhs = self.lhs_constructor(vstate, step)

        else:
            with timing.timed_scope("Autotuning diag_shift"):
                self.diag_shift = 0.0
                self.diag_scale = None
                self._lhs = self.lhs_constructor(vstate, step)

                if (
                    self._fix_norm_curve is not None
                ):  # Compute the diagonal shift parameter that fixes the norm of the gradient update
                    _, norm_list = compute_l_curve_coordinates(
                        self._lhs, gradient, αlist, solver=self.solver
                    )
                    norm_list = jnp.nan_to_num(norm_list, nan=jnp.inf)

                    idx = jnp.argmin(jnp.abs(norm_list - self._fix_norm_curve))
                    self.diag_shift = αlist[idx]

                else:  # Compute the diagonal shift parameter according to the L-curve criterion
                    self.diag_shift = auto_tikhonov_solver(
                        self._lhs,
                        gradient,
                        solver,
                        λ1=1e-12,
                        λ4=1e-1,
                        ϵ=1e-3,
                    )

                self.diag_shift = jnp.maximum(self.diag_shift, self._lower_bound_diagshift)
                self._lhs = self._lhs + self.diag_shift

        self.x0, self.info = self._lhs.solve(solver, gradient, x0=None)
        x0, unravel = tree_ravel(self.x0)        
        self._norm = jnp.abs(jnp.dot(x0.conj(), x0).real)

        if jnp.isnan(self._norm):
            #print("Norm is diverging")
            new_solver = jax.tree_util.Partial(nk.optimizer.solver.pinv, rcond=1e-10)
            for scale_grad in jnp.array([0.5, 0.1, 0.05, 0.01, 1, 5, 10, 100, 1e-6]):
                new_grad = tree_map(lambda x: scale_grad * x, gradient)
                self.x0, self.info = self._lhs.solve(new_solver, new_grad, x0=None)
                x0, unravel = tree_ravel(self.x0)        
                self._norm = jnp.abs(jnp.dot(x0.conj(), x0).real)
                #print(f"Trying scale {scale_grad:.2e}")
                if not jnp.isnan(self._norm): 
                    #print("Saved from divergence")
                    break
            
        S = self._lhs + (-self.diag_shift)
        S = S.to_dense()
        self._reimann_norm = jnp.abs(jnp.dot(x0.conj(), S@x0).real)

        if self._constrain_gradient_update is not None:
            not_one = self._constrain_gradient_update / self._norm
            not_one = jnp.nan_to_num(not_one, nan=1000, posinf=1000, neginf=0.0)
            scale = jnp.minimum(1, not_one)
            self.x0 = unravel(x0 * scale)
            #print(f"scale, norm_prev, norm_current: {scale:.2e}, {self._norm:.2e}, {self._norm*scale:.2e}")
        return self.x0


def cb_monitor_ExtendedSR(step, logdata, driver):
    sr = driver.preconditioner
    logdata['diag_shift'] = sr.diag_shift
    logdata['norm'] = sr._norm
    logdata['S_norm'] = sr._reimann_norm

    # if step%1000 == 0:
    #     import matplotlib.pyplot as plt
    #     #vs = driver.state
    #     #qgt = vs.quantum_geometric_tensor(nk.optimizer.qgt.QGTJacobianPyTree(holomorphic=True))#.to_dense()
    #     #F = vs.expect_and_grad(driver._I_op)
    #     lambda_vals = jnp.geomspace(1e-12, 1e-1, 20)
    #     S = driver.preconditioner._lhs
    #     S = S + (-driver.preconditioner.diag_shift)
    #     F = driver._loss_grad
    #     residuals, norms = compute_l_curve_coordinates(S, F, lambda_vals, solver=nksolver.cholesky)
    #     plt.plot(residuals, norms)

    #     for i,α in enumerate(lambda_vals):
    #         plt.annotate(f'{α:.2e}', (residuals[i]*1.1, norms[i]*1.2), fontsize=8)
    #     plt.yscale('log')
    #     plt.xscale('log')
    #     plt.savefig(f"l-curve_p2_{step:d}.png")

    return True



class AdaptDiagshift2Loss(struct.Pytree, mutable=True):
    r"""
    Callback to adapt the diagonal shift of the preconditioner to the infidelity.
    The diagonal shift is set to `scale*infidelity`, where `infidelity` is the mean of the loss function.

    Args:
        min_val: Minimum value of the diagonal shift.
        max_val: Maximum value of the diagonal shift.
        scale: Scaling factor for the infidelity
    """

    min_val: float = struct.field(pytree_node=False, default=1e-10)
    max_val: float = struct.field(pytree_node=False, default=1e-1)
    scale: float = struct.field(pytree_node=False, default=1.0)

    def __init__(
        self, min_val: float = 1e-10, max_val: float = 1e-1, scale: float = 1.0
    ):
        """
        Constructs the callback.

        Args:
            min_val: Minimum value of the diagonal shift.
            max_val: Maximum value of the diagonal shift.
            scale: Scaling factor for the infidelity
        """
        self.min_val = min_val
        self.max_val = max_val
        self.scale = scale

    def __call__(self, step, log_data, driver):
        infidelity = driver._loss_stats.mean
        diag_shift = np.maximum(
            self.min_val, np.minimum(self.max_val, self.scale * infidelity)
        )

        if hasattr(driver, "diag_shift"):
            driver.diag_shift = diag_shift
        elif hasattr(driver, "preconditioner") and hasattr(
            driver.preconditioner, "diag_shift"
        ):
            driver.preconditioner.diag_shift = diag_shift
        else:
            raise AttributeError(
                "No attribute diag_shift found in driver or preconditioner."
            )
        return True
