import numpy as np

from typing import Callable, Union

import jax
from jax import numpy as jnp
from jax.typing import ArrayLike

from netket.utils.types import PyTree
from netket.optimizer import LinearOperator, solver as nksolver
from netket.jax import tree_ravel

ETOL = 1e-8
ETOL_Pbar = 1e-8
φ = (1 + np.sqrt(5)) / 2.0


def get_λ2_λ3(λ1, λ4):
    x1 = np.log10(λ1)
    x4 = np.log10(λ4)
    x2 = (x4 + φ * x1) / (1 + φ)
    x3 = x1 + x4 - x2

    λ2 = 10**x2
    λ3 = 10**x3
    return λ2, λ3


def _solve_shifted_problem(
    A: Union[ArrayLike, LinearOperator],
    b: Union[ArrayLike, PyTree],
    λ: float,
    solver: Callable = nksolver.cholesky,
):
    if isinstance(A, LinearOperator):
        A_shift = A + λ
        x, _ = A_shift.solve(solver, b)
    else:
        A_shift = A + λ * jnp.identity(A.shape[0], dtype=A.dtype)
        x = solver(A_shift, b)
    return x


def compute_l_curve_coordinates(
    A: Union[ArrayLike, LinearOperator],
    b: Union[ArrayLike, PyTree],
    λ: Union[ArrayLike, float],
    solver: Callable = nksolver.cholesky,
):
    r"""
    Given a single value of the diagonal shift (or Tikhonov Regularisation parameter) λ,
    compute the L-curve point (ξ, η) for the given λ. This function solves the linear system
    :math:`\|Ax - b\|^2 + \lambda\|x\|^2`.

    The input matrix (or operator) A is assumed to have no shift, i.e. A = A + 0.

    When the input A is a matrix, b should be a vector.
    Conversly, when A is a LinearOperator, b should be a PyTree.

    In particular:
    - ξ is the residual norm, computed as :math:`\|Ax - b\|^2`
    - η is the norm of the solution, computed as :math:`\|x\|^2`

    This function works for both scalar values of lambda or arrays.

    Example:
        Compute the L curve for a given variational state

        >>> vs = nk.vqs.MCState(...)
        >>> qgt = vs.quantum_geometric_tensor()
        >>> F = vs.expect_and_forces(hamiltonian)
        >>> lambda_vals = jnp.geomspace(1e-12, 1e-1, 20)
        >>> residuals, norms = nkp.optimizer.compute_l_curve_coordinates(qgt, F, lambda_vals, nkp.optimizer.qgt.solve)
        >>>
        >>> plt.plot(residuals, norms)
        >>>
        >>> for i,α in enumerate(lambda_vals):
        >>>     plt.annotate(f'{α:.2e}', (residuals[i]*1.1, norms[i]*1.2), fontsize=8)
        >>> plt.yscale('log')
        >>> plt.xscale('log')


    Parameters:
        A: The linear operator (either a netket LinearOperator/QGT or a numpy array)
        b: The right-hand side of the linear system
        λ: The diagonal shift (or Tikhonov Regularisation parameter)
        solver: The linear solver to use to solve the linear system (default: :func:`netket.optimizer.solver.cholesky`)

    Returns:
        The tuple (ξ, η) for the given λ
    """
    λ = jnp.array(λ)

    # Since interested only in ξ, η, we convert the input to dense
    # and neglect the PyTree structure of b and of the solution
    if isinstance(A, LinearOperator):
        X = A.to_dense()
        y, _ = tree_ravel(b)
    else:
        X = A
        y = b

    if λ.ndim > 0:
        return jax.lax.map(lambda λ: compute_l_curve_coordinates(X, y, λ, solver), λ)
    else:
        x, _ = _solve_shifted_problem(X, y, λ, solver)
        ξ = jnp.linalg.norm(X @ x - y) ** 2
        η = jnp.linalg.norm(x) ** 2
        return ξ, η


@jax.jit
def menger(
    Pj: tuple[jnp.ndarray, jnp.ndarray],
    Pk: tuple[jnp.ndarray, jnp.ndarray],
    Pl: tuple[jnp.ndarray, jnp.ndarray],
) -> float:
    """
    Compute the Menger curvature of the triangle defined by the points Pj, Pk, Pl
    """
    ξj, ηj = Pj
    ξk, ηk = Pk
    ξl, ηl = Pl

    PjPk = (ξk - ξj) ** 2 + (ηk - ηj) ** 2
    PkPl = (ξl - ξk) ** 2 + (ηl - ηk) ** 2
    PlPj = (ξj - ξl) ** 2 + (ηj - ηl) ** 2

    Ck = 2 * (ξj * (ηk - ηl) + ξk * (ηl - ηj) + ξl * (ηj - ηk))
    Ck /= jnp.maximum(ETOL_Pbar, jnp.sqrt(PjPk * PkPl * PlPj))
    return Ck


def auto_tikhonov_solver(
    A: Union[ArrayLike, LinearOperator],
    b: Union[ArrayLike, PyTree],
    solver: Callable = nksolver.cholesky,
    λ1: float = 1e-12,
    λ4: float = 1e-1,
    ϵ: float = 1e-8,
):
    """
    Compute the optimal value of the Tikhonov Regularisation parameter λ and the associated solution.

    This function implements the algorithm described in https://doi.org/10.1088/2633-1357/abad0d.

    Parameters:
        A: The linear operator (either a netket LinearOperator/QGT or a numpy array)
        b: The right-hand side of the linear system
        solver: The linear solver to use to solve the linear system (default: :func:`netket.optimizer.solver.cholesky`)
        λ1: The lower bound of the search interval for λ
        λ4: The upper bound of the search interval for λ
        ϵ: The tolerance for the convergence of the algorithm

    Returns:
        The tuple (λ, x) for the optimal λ and the associated solution x
    """

    if isinstance(A, LinearOperator):
        X = A.to_dense()
        y, _ = tree_ravel(b)
    else:
        X = A
        y = b

    def get_P(λ):
        ξ, η = compute_l_curve_coordinates(X, y, λ, solver=solver)
        return jnp.log10(ξ), jnp.log10(η)

    λ2, λ3 = get_λ2_λ3(λ1, λ4)
    λl = [λ1, λ2, λ3, λ4]
    Pl = [get_P(λ) for λ in λl]

    while (λl[3] - λl[0]) / λl[3] >= ϵ:
        C2 = menger(*Pl[:-1])
        C3 = menger(*Pl[+1:])

        while C3 <= 0:
            λl[3] = λl[2]
            Pl[3] = Pl[2]

            λl[2] = λl[1]
            Pl[2] = Pl[1]

            λl[1], _ = get_λ2_λ3(λl[0], λl[3])
            Pl[1] = get_P(λl[1])

            C3 = menger(*Pl[1:])

        if C2 > C3:
            λmc = λl[1]

            λl[3] = λl[2]
            Pl[3] = Pl[2]

            λl[2] = λl[1]
            Pl[2] = Pl[1]

            λl[1], _ = get_λ2_λ3(λl[0], λl[3])
            Pl[1] = get_P(λl[1])

        else:
            λmc = λl[2]

            λl[0] = λl[1]
            Pl[0] = Pl[1]

            λl[1] = λl[2]
            Pl[1] = Pl[2]

            _, λl[2] = get_λ2_λ3(λl[0], λl[3])
            Pl[2] = get_P(λl[2])

    return λmc
