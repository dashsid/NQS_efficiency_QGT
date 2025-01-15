# Copyright 2023 Filippo Vicentini - All rights reserved.
#

from typing import Optional

import jax

from jax import numpy as jnp
#from flax import struct
from netket.utils import struct
from netket.graph import AbstractGraph

from netket.sampler.rules import MetropolisRule


def sample_to_decrease(
    key,
    config,
    *,
    Δ=1,
):
    """
    Given configuration `config`and a PRNGKey and a minimum allowed
    value of `n_min`, identify an index that can be decreased by 1.
    """
    # assuming that the real minimum value is 0, otherwise just sum
    # n_min to this.
    n_min = Δ - 1

    # Masks for decreaseable and increaseable values
    decreaseable_mask = config > n_min

    # Cumulative sums
    decreaseable_cumsum = jnp.cumsum(decreaseable_mask)

    # Count of eligible values
    num_decreaseable = decreaseable_cumsum[-1]

    # Generate a random number to select an index `i`
    i_random = jax.random.uniform(key, minval=0, maxval=num_decreaseable)
    i_idx = jnp.sum(decreaseable_cumsum <= i_random)
    return i_idx, num_decreaseable


def sample_to_increase(
    key,
    config,
    *,
    n_max,
):
    """
    Given configuration `config`and a PRNGKey and a minimum allowed
    value of `n_min`, identify an index that can be decreased by 1.
    """
    # Masks for increaseable values
    increasable_mask = config < n_max

    # Cumulative sums
    increasable_cumsum = jnp.cumsum(increasable_mask)

    # Count of eligible values
    num_increasable = increasable_cumsum[-1]

    # Generate a random number to select an index `i`
    i_random = jax.random.uniform(key, minval=0, maxval=num_increasable)
    i_idx = jnp.sum(increasable_cumsum <= i_random)
    return i_idx, num_increasable


def sample_neighbor_to_increase(
    key,
    neighbors,
    config,
    *,
    n_max,
    Δ=1,
):
    # assuming that the real minimum value is 0, otherwise just sum
    # n_min to this.
    n_max = n_max - (Δ - 1)

    if neighbors is None:
        return sample_to_increase(key, config, n_max=n_max)
    else:
        # Mask for valid neighbors that can be increased
        increaseable_neighbors_mask = (config[neighbors] < n_max) & (neighbors != -1)
        # count of invalid choices up to this index
        not_increaseable_neighbors_cumsum = jnp.cumsum(~increaseable_neighbors_mask)
        num_increaseable_neighbors = jnp.sum(increaseable_neighbors_mask)

        # Select one of the valid neighbors
        j_idx_random = jax.random.randint(
            key, shape=(), minval=0, maxval=num_increaseable_neighbors
        )
        # j_idx = neighbors[increaseable_neighbors_mask][j_idx_random]
        sentinel_adjustment = jnp.cumsum(neighbors == -1)
        j_idx_adjusted = j_idx_random + sentinel_adjustment[j_idx_random]
        j_idx_adjusted = (
            j_idx_adjusted + not_increaseable_neighbors_cumsum[j_idx_adjusted]
        )
        j_idx = neighbors[j_idx_adjusted]

        return j_idx, num_increaseable_neighbors


def count_decreasable_neighbors(
    neighbors,
    config,
):
    pass


@struct.dataclass
class ExchangeExcitationRule_(MetropolisRule):
    r""" """

    adjacency_matrix: Optional[jax.Array] = None
    """
    Adjacency matrix of the graph to be used for exchanging indices.

    """
    n_exchange: int = struct.field(pytree_node=False, default=1)

    def transition(self, sampler, machine, parameters, state, key, σ):
        hilbert = sampler.hilbert
        # Max population
        n_max = hilbert.local_size - 1

        # a
        self.n_exchange - 1
        n_max_eff = n_max - (self.n_exchange - 1)

        def single_kernel(σ, key):
            key1, key2 = jax.random.split(key)

            i_idx, num_decreaseable = sample_to_decrease(key1, σ, Δ=self.n_exchange)

            # Get the neighbors of `i` from the adjacency matrix
            if self.adjacency_matrix is not None:
                neighbors = self.adjacency_matrix[i_idx]
            else:
                neighbors = None

            # index to increase and
            # Number of valid neighbors of i_idx that can be increased
            j_idx, num_increaseable_neighbors_i = sample_neighbor_to_increase(
                key, neighbors, σ, n_max=n_max, Δ=self.n_exchange
            )

            # If the selected neighbor is the sentinel (-1), return unchanged σ
            σ_new = jnp.where(
                j_idx == -1,
                σ,
                σ.at[i_idx].add(-self.n_exchange).at[j_idx].add(self.n_exchange),
            )

            # Metropolis-Hastings log correction calculation

            # For the reverse move, we'll need the number of decreaseable neighbors of j_idx
            num_decreaseable_new = jnp.sum(σ_new > 0)

            if self.adjacency_matrix is not None:
                neighbors_j = self.adjacency_matrix[j_idx]
                increaseable_neighbors_mask_new = (σ_new[neighbors_j] < n_max_eff) & (
                    neighbors_j != -1
                )
                num_increaseable_neighbors_j_new = jnp.sum(
                    increaseable_neighbors_mask_new
                )
            else:
                num_increaseable_neighbors_j_new = jnp.sum(σ_new < n_max_eff)

            log_correction = -jnp.log(
                (num_decreaseable_new * num_increaseable_neighbors_j_new)
                / (num_decreaseable * num_increaseable_neighbors_i)
            )
            # if it's nan it's an invalid move (0/0) so just reject it
            log_correction = jnp.where(
                jnp.isnan(log_correction), -jnp.inf, log_correction
            )

            # jax.debug.print(
            #     """
            #    decreasing: {}[i={}] => {}  increasing[j={}] => {}
            #      neighbors {} increasable {}
            #      new state {} ({}) log_correction {}
            #      num_decreaseable_new {} num_increaseable_neighbors_j_new {}
            #      num_decreaseable {} num_increaseable_neighbors_i {}
            #
            #      neighbors_j{} increaseable_neighbors_mask_new {}
            #    """,
            #     σ,
            #     i_idx,
            #     σ[i_idx],
            #     j_idx,
            #     σ[j_idx],
            #     neighbors,
            #     num_increaseable_neighbors_i,
            #     σ_new,
            #     σ_new.sum(),
            #     log_correction,
            #     num_decreaseable_new,
            #     num_increaseable_neighbors_j_new,
            #     num_decreaseable,
            #     num_increaseable_neighbors_i,
            #     neighbors_j,
            #     increaseable_neighbors_mask_new,
            # )

            return σ_new, log_correction

        kernel = jax.vmap(single_kernel)

        # Convert to 0,1...N basis
        σ_idx = hilbert.states_to_local_indices(σ)
        keys = jax.random.split(key, σ_idx.shape[0])

        σp, log_probs = kernel(σ_idx, keys)

        σp = jnp.array(hilbert.local_states, dtype=σ.dtype)[σp]

        return σp, log_probs

    def __repr__(self):
        return "ExchangeExcitationRule()"


def edge_list_to_adjacency_list(edge_list, num_nodes):
    adj_list = {}
    for i, j in edge_list:
        adj_list.setdefault(i, []).append(j)
        adj_list.setdefault(j, []).append(i)  # If undirected

    max_neighbors = max(len(neighbors) for neighbors in adj_list.values())
    adjacency_matrix = -1 * jnp.ones((num_nodes, max_neighbors), dtype=jnp.int64)

    for node, neighbors in adj_list.items():
        adjacency_matrix = adjacency_matrix.at[node, : len(neighbors)].set(neighbors)

    return adjacency_matrix


def ExchangeExcitationRule(
    *,
    n_exchange: int = 1,
    graph: Optional[AbstractGraph] = None,
    d_max: int = 1,
):
    r"""
    A Rule exchanging the state on a random couple of sites, chosen from a list of
    possible couples (clusters).

    This rule acts on two local degree of freedom :math:`s_i` and :math:`s_j`,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s^\prime_j \dots s_N`,
    where in general :math:`s^\prime_i \neq s_i` and :math:`s^\prime_j \neq s_j`.
    The sites :math:`i` and :math:`j` are also chosen to be within a maximum graph
    distance of :math:`d_{\mathrm{max}}`.

    The transition probability associated to this sampler can
    be decomposed into two steps:

    1. A pair of indices :math:`i,j = 1\dots N`, and such
       that :math:`\mathrm{dist}(i,j) \leq d_{\mathrm{max}}`,
       is chosen with uniform probability.
    2. The sites are exchanged, i.e. :math:`s^\prime_i = s_j` and :math:`s^\prime_j = s_i`.

    Notice that this sampling method generates random permutations of the quantum
    numbers, thus global quantities such as the sum of the local quantum numbers
    are conserved during the sampling.
    This scheme should be used then only when sampling in a
    region where :math:`\sum_i s_i = \mathrm{constant}` is needed,
    otherwise the sampling would be strongly not ergodic.

    Args:
        clusters: The list of clusters that can be exchanged.
        graph: A graph, from which the edges determine the clusters that can be exchanged.
        d_max: Only valid if a graph is passed in. The maximum distance between two sites
    """
    if graph is not None:
        adjacency_matrix = edge_list_to_adjacency_list(graph.edges(), graph.n_nodes)
    else:
        adjacency_matrix = None
    #print(adjacency_matrix)
    return ExchangeExcitationRule_(adjacency_matrix, n_exchange)
