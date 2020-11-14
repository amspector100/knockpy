"""
Functions for computing treewidth decomposition. This code is
adapted from the metro package and the paper https://arxiv.org/abs/1903.00434.

*This code is taken from the NetworkX package source code.*

Treewidth of an undirected graph is a number associated with the graph.
It can be defined as the size of the largest vertex set (bag) in a tree
decomposition of the graph minus one.

`Wikipedia: Treewidth <https://en.wikipedia.org/wiki/Treewidth>`_

The notions of treewidth and tree decomposition have gained their
attractiveness partly because many graph and network problems that are
intractable (e.g., NP-hard) on arbitrary graphs become efficiently
solvable (e.g., with a linear time algorithm) when the treewidth of the
input graphs is bounded by a constant [1]_ [2]_.

There are two different functions for computing a tree decomposition:
:func:`treewidth_min_degree` and :func:`treewidth_min_fill_in`.

.. [1] Hans L. Bodlaender and Arie M. C. A. Koster. 2010. "Treewidth
      computations I.Upper bounds". Inf. Comput. 208, 3 (March 2010),259-275.
      http://dx.doi.org/10.1016/j.ic.2009.03.008

.. [2] Hand L. Bodlaender. "Discovering Treewidth". Institute of Information
      and Computing Sciences, Utrecht University.
      Technical Report UU-CS-2005-018.
      http://www.cs.uu.nl

.. [3] K. Wang, Z. Lu, and J. Hicks *Treewidth*.
      http://web.eecs.utk.edu/~cphillip/cs594_spring2015_projects/treewidth.pdf

NetworkX is distributed with the 3-clause BSD license.

::

   Copyright (C) 2004-2019, NetworkX Developers
   Aric Hagberg <hagberg@lanl.gov>
   Dan Schult <dschult@colgate.edu>
   Pieter Swart <swart@lanl.gov>
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

     * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

     * Neither the name of the NetworkX Developers nor the names of its
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys
import numpy as np
import networkx as nx
from networkx.utils import not_implemented_for
from heapq import heappush, heappop, heapify
import itertools

def get_ordering(T):
    """ 
    Takes a junction tree and returns a variable ordering for the metro
    knockoff sampler.

    :param T: A networkx graph that is a junction tree.
    Nodes must be sets with elements 0,...,p-1. 
    e.g.: width, T = treewidth_decomp(G)

    Returns:
    order : a numpy array with unique elements 0,...,p-1
    active_frontier (list of lists) : a list of length p gwhere
    entry j is the set of entries > j that are in V_j. This specifies
    the conditional independence structure of the distribution given by
    lf. See page 34 of the paper.
    """ 
    # Initialize
    T = T.copy()
    order = []
    active_frontier = []
    
    while(T.number_of_nodes() > 0):
        # Loop through leaf nodes
        gen = (x for x in T.nodes() if T.degree(x) <= 1)
        active_node = next(gen)

        # Parent nodes of leaf nodes
        # active_vars get added to the order in this step
        # activated set are just the variables in the active node
        parents = list(T[active_node].keys())
        if(len(parents) == 0):
            active_vars = set(active_node)
            activated_set = active_vars.copy()
        else:
            active_vars = set(active_node.difference(parents[0]))
            activated_set = active_vars.union(parents[0]).difference(set(order))
        for i in list(active_vars)[::-1]: # This line was changed too
            order += [i]
            frontier = list(activated_set.difference(set(order)))
            active_frontier += [frontier]
        T.remove_node(active_node)
    
    return [np.array(order), active_frontier]

@not_implemented_for('directed')
@not_implemented_for('multigraph')
def treewidth_min_degree(G):
    """ Returns a treewidth decomposition using the Minimum Degree heuristic.

    The heuristic chooses the nodes according to their degree, i.e., first
    the node with the lowest degree is chosen, then the graph is updated
    and the corresponding node is removed. Next, a new node with the lowest
    degree is chosen, and so on.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    Treewidth decomposition : (int, Graph) tuple
          2-tuple with treewidth and the corresponding decomposed tree.
    """
    deg_heuristic = MinDegreeHeuristic(G)
    return treewidth_decomp(G, lambda graph: deg_heuristic.best_node(graph))



@not_implemented_for('directed')
@not_implemented_for('multigraph')
def treewidth_min_fill_in(G):
    """ Returns a treewidth decomposition using the Minimum Fill-in heuristic.

    The heuristic chooses a node from the graph, where the number of edges
    added turning the neighbourhood of the chosen node into clique is as
    small as possible.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    Treewidth decomposition : (int, Graph) tuple
        2-tuple with treewidth and the corresponding decomposed tree.
    """
    return treewidth_decomp(G,  min_fill_in_heuristic)



class MinDegreeHeuristic:
    """ Implements the Minimum Degree heuristic.

    The heuristic chooses the nodes according to their degree
    (number of neighbours), i.e., first the node with the lowest degree is
    chosen, then the graph is updated and the corresponding node is
    removed. Next, a new node with the lowest degree is chosen, and so on.
    """
    def __init__(self, graph):
        self._graph = graph

        # nodes that have to be updated in the heap before each iteration
        self._update_nodes = []

        self._degreeq = []  # a heapq with 2-tuples (degree,node)

        # build heap with initial degrees
        for n in graph:
            self._degreeq.append((len(graph[n]), n))
        heapify(self._degreeq)

    def best_node(self, graph):
        # update nodes in self._update_nodes
        for n in self._update_nodes:
            # insert changed degrees into degreeq
            heappush(self._degreeq, (len(graph[n]), n))

        # get the next valid (minimum degree) node
        while self._degreeq:
            (min_degree, elim_node) = heappop(self._degreeq)
            if elim_node not in graph or len(graph[elim_node]) != min_degree:
                # outdated entry in degreeq
                continue
            elif min_degree == len(graph) - 1:
                # fully connected: abort condition
                return None

            # remember to update nodes in the heap before getting the next node
            self._update_nodes = graph[elim_node]
            return elim_node

        # the heap is empty: abort
        return None


def min_fill_in_heuristic(graph):
    """ Implements the Minimum Degree heuristic.

    Returns the node from the graph, where the number of edges added when
    turning the neighbourhood of the chosen node into clique is as small as
    possible. This algorithm chooses the nodes using the Minimum Fill-In
    heuristic. The running time of the algorithm is :math:`O(V^3)` and it uses
    additional constant memory."""

    if len(graph) == 0:
        return None

    min_fill_in_node = None

    min_fill_in = sys.maxsize

    # create sorted list of (degree, node)
    degree_list = [(len(graph[node]), node) for node in graph]
    degree_list.sort()

    # abort condition
    min_degree = degree_list[0][0]
    if min_degree == len(graph) - 1:
        return None

    for (_, node) in degree_list:
        num_fill_in = 0
        nbrs = graph[node]
        for nbr in nbrs:
            # count how many nodes in nbrs current nbr is not connected to
            # subtract 1 for the node itself
            num_fill_in += len(nbrs - graph[nbr]) - 1
            if num_fill_in >= 2 * min_fill_in:
                break

        num_fill_in /= 2  # divide by 2 because of double counting

        if num_fill_in < min_fill_in:  # update min-fill-in node
            if num_fill_in == 0:
                return node
            min_fill_in = num_fill_in
            min_fill_in_node = node

    return min_fill_in_node


def treewidth_decomp(G, heuristic=min_fill_in_heuristic):
    """ Returns a treewidth decomposition using the passed heuristic.

    Parameters
    ----------
    G : NetworkX graph
    heuristic : heuristic function

    Returns
    -------
    Treewidth decomposition : (int, Graph) tuple
        2-tuple with treewidth and the corresponding decomposed tree.
    """

    # make dict-of-sets structure
    graph = {n: set(G[n]) - set([n]) for n in G}

    # stack containing nodes and neighbors in the order from the heuristic
    node_stack = []

    # get first node from heuristic
    elim_node = heuristic(graph)
    while elim_node is not None:
        # connect all neighbours with each other
        nbrs = graph[elim_node]
        for u, v in itertools.permutations(nbrs, 2):
            if v not in graph[u]:
                graph[u].add(v)

        # push node and its current neighbors on stack
        node_stack.append((elim_node, nbrs))

        # remove node from graph
        for u in graph[elim_node]:
            graph[u].remove(elim_node)

        del graph[elim_node]
        elim_node = heuristic(graph)

    # the abort condition is met; put all remaining nodes into one bag
    decomp = nx.Graph()
    first_bag = frozenset(graph.keys())
    decomp.add_node(first_bag)

    treewidth = len(first_bag) - 1

    while node_stack:
        # get node and its neighbors from the stack
        (curr_node, nbrs) = node_stack.pop()

        # find a bag all neighbors are in
        old_bag = None
        for bag in decomp.nodes():
            if nbrs <= bag:
                old_bag = bag
                break

        if old_bag is None:
            # no old_bag was found: just connect to the first_bag
            old_bag = first_bag

        # create new node for decomposition
        nbrs.add(curr_node)
        new_bag = frozenset(nbrs)

        # update treewidth
        treewidth = max(treewidth, len(new_bag) - 1)

        # add edge to decomposition (implicitly also adds the new node)
        decomp.add_edge(old_bag, new_bag)

    return treewidth, decomp