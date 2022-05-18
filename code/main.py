import heapq
import importlib
import math
import time
import sys
import os

import networkit as nk
import networkx as nx
import numpy as np
import ot

import pickle as pkl
import matplotlib.pyplot as plt

from cdlib import NodeClustering
from cdlib import algorithms
from cdlib import evaluation
import sys

from GraphRicciCurvature import OllivierRicci as OR
from GraphRicciCurvature import util

from sklearn import preprocessing, metrics
from itertools import groupby
import operator

import random

# --------------------------------------------------------------------------
import utils

############################## Nextrout ####################################
sys.path.append("../nextrout_tutorial/")
import main

file_path = os.path.dirname(os.path.realpath(__file__))
abs_path = file_path.split("ORC-Nextrout-dev")[0]
sys.path.append(abs_path + "/Nextrout/nextrout_core/")
print(abs_path)
import filtering as fn

##############################  MT  ######################################
import mt_tools

# --------------------------------------------------------------------------

EPSILON = 1e-7  # to prevent divided by zero

# ---Shared global variables for multiprocessing used.---
_Gk = nk.graph.Graph()
_alpha = 0.0
_weight = "weight"
_method = "Nextrout"
_base = math.e
_exp_power = 1
_shortest_path = "all_pairs"
_nbr_topk = 3000
_apsp = {}

# -------------------------------------------------------


def _get_all_pairs_shortest_path():
    # Taken from: https://github.com/saibalmars/GraphRicciCurvature/blob/master/GraphRicciCurvature/OllivierRicci.py
    """Pre-compute all pairs shortest paths of the assigned graph `_Gk`."""

    global _Gk

    t0 = time.time()
    apsp = nk.distance.APSP(_Gk).run().getDistances()

    return np.array(apsp)


def _distribute_densities(source, target, nextrout_in=False):
    # Taken from: https://github.com/saibalmars/GraphRicciCurvature/blob/master/GraphRicciCurvature/OllivierRicci.py
    """Get the density distributions of source and target node, and the cost (all pair shortest paths) between
    all source's and target's neighbors. Notice that only neighbors with top `_nbr_topk` edge weights.

    Parameters
    ----------
    source : int
        Source node index in Networkit graph `_Gk`.
    target : int
        Target node index in Networkit graph `_Gk`.
    Returns
    -------
    x : (m,) numpy.ndarray
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix.

    """

    # Distribute densities for source and source's neighbors as x
    t0 = time.time()
    x, source_topknbr = _get_single_node_neighbors_distributions(source)
    # Distribute densities for target and target's neighbors as y
    y, target_topknbr = _get_single_node_neighbors_distributions(target)
    # construct the cost dictionary from x to y
    t0 = time.time()

    if _shortest_path == "pairwise":
        d = []
        for src in source_topknbr:
            tmp = []
            for tgt in target_topknbr:
                tmp.append(_source_target_shortest_path(src, tgt))
            d.append(tmp)
        d = np.array(d)
    else:  # all_pairs
        d = _apsp[np.ix_(source_topknbr, target_topknbr)]  # transportation matrix

    x = np.array(x)  # the mass that source neighborhood initially owned
    y = np.array(y)  # the mass that target neighborhood needs to received

    # logger.debug("%8f secs density matrix construction for edge." % (time.time() - t0))

    return x, y, d


def _get_topknbr(source, target):
    """Get the neighbors of source and target node.
    Parameters
    ----------
    source : int
        Source node index in Networkit graph `_Gk`.
    target : int
        Target node index in Networkit graph `_Gk`.
    Returns
    -------
    source_topknbr : (m,) numpy array
        Source and source's neighbors.
    target_topknbr : (n,) numpy array
        Target and target's neighbors.
    """

    # Distribute densities for source and source's neighbors as x
    t0 = time.time()

    x, source_topknbr = _get_single_node_neighbors_distributions(source)

    # Distribute densities for target and target's neighbors as y
    y, target_topknbr = _get_single_node_neighbors_distributions(target)

    return source_topknbr, target_topknbr


def _nextrout_distance(
    small_G,
    x_dict,
    y_dict,
    d,
    source,
    target,
    isource,
    isink,
    beta_d,
    nx2nk_ndict,
    stopThres,
    MaxNumIter,
    tdens_gfvar="tdens",
    explicit_implicit="explicit",
):
    """Get the OT cost given by Nextrout.
    Parameters
    ----------
    small_G: A NetworkX graph.
        Neighborhood (bipartite) graph.
    x_dict : dictionary
        Source distribution extended to the small_G graph.
    y_dict : dictionary
        Target distribution extended to the small_G graph.
    d : numpy array
        Shortest-path information.
    source : int
        Source label.
    target : int
        Target label.
    isource : numpy array
        Source and source's neighbors.
    isink : numpy array
        Target and target's neighbors.
    beta_d : float
        Traffic rate for the DMK problem.
    nx2nk_dict : dictionary
        nx graph node keys to nk graph node keys.
    stopThres : float
        Accuracy used to solve the OT problem.
    MaxNumIter: int
        Max Number of iterations used to solve the OT problem.
    tdens_gfvar, explicit_implicit: str
        Schemes used to solve the DMK problem.
    Returns
    -------
    m : float
        beta-Wasserstein cost.
    """

    fplus = np.zeros(len(small_G))
    fminus = np.zeros(len(small_G))

    nodes = list(small_G.nodes())
    nodes.sort()

    for position, node in enumerate(nodes):
        fplus[position] = x_dict[node]
        fminus[position] = y_dict[node]
    edges = small_G.edges()
    distances = np.ones(len(edges))
    idx = -1
    for edge in edges:
        idx += 1

        node1 = edge[0]
        node2 = edge[1]
        # print(node1, node2)
        # accessing the proper indices
        shortest_path = False

        try:  # it is always true: e = (u,v), then either u neig of source and v neig of sink or viceversa
            idx1 = isource.index(node1)
            idx2 = isink.index(node2)
        except:
            try:
                idx1 = isource.index(node2)
                idx2 = isink.index(node1)
            except:
                shortest_path = True
                idx1 = 0
                idx2 = 0

                if True:
                    dst = nx.shortest_path_length(
                        small_G,
                        source=node1,
                        target=node2,
                        weight="weight",
                        method="dijkstra",
                    )
                else:
                    # print('accessing already computed apsp')
                    dst = _apsp[nx2nk_ndict[node1]][nx2nk_ndict[node2]]

        if not shortest_path:
            dst = d[idx1, idx2]

        # accessing d with these indices
        distances[
            idx
        ] = dst  # first index is related to the neigs of source, second to the target

    splus = sum(fplus)
    sminus = sum(fminus)

    fplus = np.array([e / splus for e in fplus])
    fminus = np.array([e / sminus for e in fminus])

    rhs = fplus - fminus

    distances += 0.00001

    fluxes = main.find_wass(
        small_G,
        rhs,
        weight_flag=distances,  # "unit",  # "shortest_path_distance", this should be changed? yes
        beta_d=beta_d,
        stopThres=stopThres,
        MaxNumIter=MaxNumIter,
        tdens_gfvar=tdens_gfvar,
        explicit_implicit=explicit_implicit,
    )

    fluxes = np.abs(fluxes)
    m = np.sum(np.multiply(fluxes, distances))

    return m


def _compute_nextrout_ricci_curvature_single_edge(
    source, target, G=None, beta_d=1, explicit_implicit="explicit", tdens_gfvar="tdens"
):
    # Adapted from: https://github.com/saibalmars/GraphRicciCurvature/blob/master/GraphRicciCurvature/OllivierRicci.py
    """Ricci curvature computation for a given single edge (modified).

    Parameters
    ----------
    source : int
        Source node index in Networkit graph `_Gk`.
    target : int
        Target node index in Networkit graph `_Gk`.

    Returns
    -------
    result : dict[(int,int), float]
        The Ricci curvature of given edge in dict format. E.g.: {(node1, node2): ricciCurvature}

    """

    # If the weight of edge is too small, return 0 instead.
    if _Gk.weight(source, target) < EPSILON:
        logger.trace(
            "Zero weight edge detected for edge (%s,%s), return Ricci Curvature as 0 instead."
            % (source, target)
        )
        return {(source, target): 0}

    # compute transportation distance
    m = 1  # assign an initial cost

    stopThres = 1e-3
    MaxNumIter = 10
    # alpha = 0
    t0 = time.time()

    # getting source distr, sink distr, cost matrix, neighbours of source, neighbours of sink
    (
        x,
        y,
        d,
    ) = _distribute_densities(source, target, nextrout_in=False)
    source_nbs, target_nbs = _get_topknbr(source, target)
    # getting subgraph
    source_un2ord = {}
    source_ord2un = {}

    for idx, node in enumerate(source_nbs):
        source_un2ord[node] = idx
        source_ord2un[idx] = node

    target_un2ord = {}
    target_ord2un = {}

    for idx, node in enumerate(target_nbs):
        target_un2ord[node] = idx + len(source_nbs)
        target_ord2un[idx + len(source_nbs)] = node

    source_nbs_rel = [source_un2ord[node] for node in source_nbs]
    target_nbs_rel = [target_un2ord[node] for node in target_nbs]

    small_G = nx.complete_bipartite_graph(source_nbs_rel, target_nbs_rel)

    for node in small_G:
        if node < len(source_nbs):
            small_G.nodes[node]["source"] = source_ord2un[node]
        else:
            small_G.nodes[node]["target"] = target_ord2un[node]

    # getting f+ and f- for all the nodes in small_G

    x_dict, y_dict = get_forcings(small_G, x, y, source_nbs_rel, target_nbs_rel)

    nx2nk_ndict = {}

    m = _nextrout_distance(
        small_G,
        x_dict,
        y_dict,
        d,
        source,
        target,
        source_nbs_rel,
        target_nbs_rel,
        beta_d,
        nx2nk_ndict,
        stopThres,
        MaxNumIter,
    )
    # compute Ricci curvature: k=1-(m_{x,y})/d(x,y)
    result = 1 - (m / _Gk.weight(source, target))  # Divided by the length of d(i, j)

    return {(source, target): result}


def get_forcings(small_G, x, y, source_nbs, target_nbs):

    """Compute forcing term for the DMK model (source minus target distribution).

    Parameters
    ----------
    small_G: A NetworkX graph.
        Neighborhood (bipartite) graph.
    x : numpy array
        Source distribution.
    y: numpy array
        Target distribution.
    source_nbs: numpy array
        Neighbors of first node (label not specified here).
    target_nbs:
        Neighbors of second node (label not specified here).
    Returns
    -------
    x_dict, y_dict: dictionaries
        x, y extended to the small_G, with keys in source_nbs U target_nbs.

    """

    nk2nx_ndict, nx2nk_ndict = {}, {}

    nodes_small_G = list(small_G.nodes())
    nodes_small_G.sort()

    for idx, n in enumerate(nodes_small_G):
        nk2nx_ndict[idx] = n
        nx2nk_ndict[n] = idx

    x_dict = {}
    idx_so = -1
    for node in source_nbs:
        idx_so += 1
        x_dict[node] = x[idx_so]

    y_dict = {}
    idx_si = -1
    for node in target_nbs:
        idx_si += 1
        y_dict[node] = y[idx_si]

    # filling out dict entries
    for node in small_G:
        try:
            test = x_dict[node]
        except:
            x_dict[node] = 0
        try:
            test = y_dict[node]
        except:
            y_dict[node] = 0

    return x_dict, y_dict


def _compute_nextrout_ricci_curvature_edges(
    G: nx.Graph,
    weight="weight",
    edge_list=[],
    alpha=0.0,
    method="OTDSinkhornMix",
    base=1,
    exp_power=0,
    shortest_path="all_pairs",
    nbr_topk=3000,
    beta_d=1,
    tdens_gfvar="tdens",
    explicit_implicit="explicit",
):
    # Adapted from: https://github.com/saibalmars/GraphRicciCurvature/blob/master/GraphRicciCurvature/OllivierRicci.py
    """Compute Ricci curvature for edges in  given edge lists.

    Parameters
    ----------
    G : NetworkX graph
        A given directional or undirectional NetworkX graph.
    weight : str
        The edge weight used to compute Ricci curvature. (Default value = "weight")
    edge_list : list of edges
        The list of edges to compute Ricci curvature, set to [] to run for all edges in G. (Default value = [])
    alpha : float
        The parameter for the discrete Ricci curvature, range from 0 ~ 1.
        It means the share of mass to leave on the original node.
        E.g. x -> y, alpha = 0.4 means 0.4 for x, 0.6 to evenly spread to x's nbr.
        (Default value = 0.5)
    method : {"OTD", "ATD", "Sinkhorn"}
        The optimal transportation distance computation method. (Default value = "OTDSinkhornMix")

        Transportation method:
            - "OTD" for Optimal Transportation Distance,
            - "ATD" for Average Transportation Distance.
            - "Sinkhorn" for OTD approximated Sinkhorn distance.
            - "OTDSinkhornMix" use OTD for nodes of edge with less than _OTDSinkhorn_threshold(default 2000) neighbors,
            use Sinkhorn for faster computation with nodes of edge more neighbors. (OTD is faster for smaller cases)
    base : float
        Base variable for weight distribution. (Default value = `math.e`)
    exp_power : float
        Exponential power for weight distribution. (Default value = 0)
    shortest_path : {"all_pairs","pairwise"}
        Method to compute shortest path. (Default value = `all_pairs`)
    nbr_topk : int
        Only take the top k edge weight neighbors for density distribution.
        Smaller k run faster but the result is less accurate. (Default value = 3000)

    Returns
    -------
    output : dict[(int,int), float]
        A dictionary of edge Ricci curvature. E.g.: {(node1, node2): ricciCurvature}.

    """

    if not nx.get_edge_attributes(G, weight):
        # logger.info(
        #    'Edge weight not detected in graph, use "weight" as default edge weight.'
        # )
        for (v1, v2) in G.edges():
            G[v1][v2][weight] = 1.0

    # ---set to global variable for multiprocessing used.---
    global _Gk
    global _alpha
    global _weight
    global _base
    global _exp_power
    global _shortest_path
    global _nbr_topk
    global _apsp
    global _beta_d
    global _tdens_gfvar
    global _explicit_implicit
    # -------------------------------------------------------

    _Gk = nk.nxadapter.nx2nk(G, weightAttr=weight)
    _alpha = alpha
    _weight = weight
    _method = method
    _base = base
    _exp_power = exp_power
    _shortest_path = shortest_path
    _nbr_topk = nbr_topk
    _beta_d = beta_d
    _tdens_gfvar = tdens_gfvar
    _explicit_implicit = explicit_implicit

    # Construct nx to nk dictionary
    nx2nk_ndict, nk2nx_ndict = {}, {}
    for idx, n in enumerate(G.nodes()):
        nx2nk_ndict[n] = idx
        nk2nx_ndict[idx] = n

    if _shortest_path == "all_pairs":
        # Construct the all pair shortest path dictionary
        # if not _apsp:
        _apsp = _get_all_pairs_shortest_path()

    if edge_list:
        edges = [
            (nx2nk_ndict[source], nx2nk_ndict[target], G, beta_d)
            for source, target in edge_list
        ]
    else:
        edges = [
            (nx2nk_ndict[source], nx2nk_ndict[target], G, beta_d)
            for source, target in G.edges()
        ]

    # Start compute edge Ricci curvature
    t0 = time.time()

    result = []
    i = 0
    for e in edges:
        i += 1
        source, target, G, beta_d = e
        edge_res = _compute_nextrout_ricci_curvature_single_edge(
            source,
            target,
            G=G,
            beta_d=beta_d,
            explicit_implicit=explicit_implicit,
            tdens_gfvar=tdens_gfvar,
        )
        result.append(edge_res)
    # Convert edge index from nk back to nx for final output
    output = {}
    for rc in result:
        for k in list(rc.keys()):
            output[(nk2nx_ndict[k[0]], nk2nx_ndict[k[1]])] = rc[k]

    return output


def _compute_nextrout_ricci_curvature(G: nx.Graph, weight="weight", **kwargs):
    # Adapted from: https://github.com/saibalmars/GraphRicciCurvature/blob/master/GraphRicciCurvature/OllivierRicci.py
    """Compute Ricci curvature of edges and nodes.
    The node Ricci curvature is defined as the average of node's adjacency edges.

    Parameters
    ----------
    G : NetworkX graph
        A given directional or undirectional NetworkX graph.
    weight : str
        The edge weight used to compute Ricci curvature. (Default value = "weight")
    **kwargs
        Additional keyword arguments passed to `_compute_ricci_curvature_edges`.

    Returns
    -------
    G: NetworkX graph
        A NetworkX graph with "ricciCurvature" on nodes and edges.
    """

    # compute Ricci curvature for all edges
    edge_ricci = _compute_nextrout_ricci_curvature_edges(G, weight=weight, **kwargs)

    # Assign edge Ricci curvature from result to graph G
    nx.set_edge_attributes(G, edge_ricci, "ricciCurvature")

    return G


def _compute_nextrout_ricci_flow(
    G: nx.Graph,
    weight="weight",
    iterations=20,
    step=1,
    delta=1e-4,
    surgery=(lambda G, *args, **kwargs: G, 100),
    **kwargs
):
    # Adapted from: https://github.com/saibalmars/GraphRicciCurvature/blob/master/GraphRicciCurvature/OllivierRicci.py
    """Compute the given Ricci flow metric of each edge of a given connected NetworkX graph.

    Parameters
    ----------
    G : NetworkX graph
        A given directional or undirectional NetworkX graph.
    weight : str
        The edge weight used to compute Ricci curvature. (Default value = "weight")
    iterations : int
        Iterations to require Ricci flow metric. (Default value = 20)
    step : float
        step size for gradient decent process. (Default value = 1)
    delta : float
        process stop when difference of Ricci curvature is within delta. (Default value = 1e-4)
    surgery : (function, int)
        A tuple of user define surgery function that will execute every certain iterations.
        (Default value = (lambda G, *args, **kwargs: G, 100))
    **kwargs
        Additional keyword arguments passed to `_compute_ricci_curvature`.

    Returns
    -------
    G: NetworkX graph
        A NetworkX graph with ``weight`` as Ricci flow metric.
    """

    if not nx.is_connected(G):
        # logger.info(
        #    "Not connected graph detected, compute on the largest connected component instead."
        # )
        G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))

    # Set normalized weight to be the number of edges.
    normalized_weight = float(G.number_of_edges())

    global _apsp

    # Start compute edge Ricci flow
    t0 = time.time()

    if nx.get_edge_attributes(G, "original_RC"):
        # logger.info("original_RC detected, continue to refine the ricci flow.")
        pass
    else:
        # logger.info("No ricciCurvature detected, compute original_RC...")
        _compute_nextrout_ricci_curvature(G, weight=weight, **kwargs)

        for (v1, v2) in G.edges():
            G[v1][v2]["original_RC"] = G[v1][v2]["ricciCurvature"]

        # clear the APSP since the graph have changed.
        _apsp = {}

    # Start the Ricci flow process
    for i in range(iterations):
        for (v1, v2) in G.edges():
            G[v1][v2][weight] -= (
                step * (G[v1][v2]["ricciCurvature"]) * G[v1][v2][weight]
            )

        # Do normalization on all weight to prevent weight expand to infinity
        w = nx.get_edge_attributes(G, weight)
        sumw = sum(w.values())
        for k, v in w.items():
            w[k] = np.abs(w[k] * (normalized_weight / sumw))
        nx.set_edge_attributes(G, values=w, name=weight)
        # logger.info(" === Ricci flow iteration %d === " % i)

        _compute_nextrout_ricci_curvature(G, weight=weight, **kwargs)

        rc = nx.get_edge_attributes(G, "ricciCurvature")
        diff = max(rc.values()) - min(rc.values())
        # print(diff)
        # logger.trace("Ricci curvature difference: %f" % diff)

        # logger.trace(
        #    "max:%f, min:%f | maxw:%f, minw:%f"
        #    % (max(rc.values()), min(rc.values()), max(w.values()), min(w.values()))
        # )

        if diff < delta:
            #    logger.trace("Ricci curvature converged, process terminated.")
            break

        # do surgery or any specific evaluation
        surgery_func, do_surgery = surgery
        if i != 0 and i % do_surgery == 0:
            G = surgery_func(G, weight)
            normalized_weight = float(G.number_of_edges())

        for n1, n2 in G.edges():
            # logger.debug("%s %s %s" % (n1, n2, G[n1][n2]))
            pass

        # clear the APSP since the graph have changed.
        _apsp = {}

    # logger.info("%8f secs for Ricci flow computation." % (time.time() - t0))

    return G


class ORC_Nextrout:
    # Adapted from: https://github.com/saibalmars/GraphRicciCurvature/blob/master/GraphRicciCurvature/OllivierRicci.py
    """A class to compute Ollivier-Ricci curvature for all nodes and edges in G.
    Node Ricci curvature is defined as the average of all it's adjacency edge.

    """

    def __init__(
        self,
        G: nx.Graph,
        weight="weight",
        alpha=0.0,
        method="OTDSinkhornMix",
        base=1,
        exp_power=0,
        shortest_path="all_pairs",
        nbr_topk=3000,
        beta_d=1,
        tdens_gfvar="tdens",
        explicit_implicit="explicit",
    ):
        """Initialized a container to compute Ollivier-Ricci curvature/flow.

        Parameters
        ----------
        G : NetworkX graph
            A given directional or undirectional NetworkX graph.
        weight : str
            The edge weight used to compute Ricci curvature. (Default value = "weight")
        alpha : float
            The parameter for the discrete Ricci curvature, range from 0 ~ 1.
            It means the share of mass to leave on the original node.
            E.g. x -> y, alpha = 0.4 means 0.4 for x, 0.6 to evenly spread to x's nbr.
            (Default value = 0.5)
        method : {"OTD", "ATD", "Sinkhorn"}
            The optimal transportation distance computation method. (Default value = "OTDSinkhornMix")

            Transportation method:
                - "OTD" for Optimal Transportation Distance,
                - "ATD" for Average Transportation Distance.
                - "Sinkhorn" for OTD approximated Sinkhorn distance.
                - "OTDSinkhornMix" use OTD for nodes of edge with less than _OTDSinkhorn_threshold(default 2000) neighbors,
                use Sinkhorn for faster computation with nodes of edge more neighbors. (OTD is faster for smaller cases)
        base : float
            Base variable for weight distribution. (Default value = `math.e`)
        exp_power : float
            Exponential power for weight distribution. (Default value = 2)
        shortest_path : {"all_pairs","pairwise"}
            Method to compute shortest path. (Default value = `all_pairs`)
        nbr_topk : int
            Only take the top k edge weight neighbors for density distribution.
            Smaller k run faster but the result is less accurate. (Default value = 3000)

        """
        self.G = G.copy()
        self.alpha = alpha
        self.weight = weight
        self.method = method
        self.base = base
        self.exp_power = exp_power
        self.shortest_path = shortest_path
        self.nbr_topk = nbr_topk
        self.beta_d = beta_d
        self.tdens_gfvar = tdens_gfvar
        self.explicit_implicit = explicit_implicit

        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}  # density distribution dictionary

        assert importlib.util.find_spec(
            "ot"
        ), "Package POT: Python Optimal Transport is required for Sinkhorn distance."

        if not nx.get_edge_attributes(self.G, weight):
            # logger.info(
            #    'Edge weight not detected in graph, use "weight" as default edge weight.'
            # )
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][weight] = 1.0

        self_loop_edges = list(nx.selfloop_edges(self.G))
        if self_loop_edges:
            self.G.remove_edges_from(self_loop_edges)

    def compute_nextrout_ricci_curvature_edges(self, edge_list=None):
        # Adapted from: https://github.com/saibalmars/GraphRicciCurvature/blob/master/GraphRicciCurvature/OllivierRicci.py
        """Compute Ricci curvature for edges in given edge lists.

        Parameters
        ----------
        edge_list : list of edges
            The list of edges to compute Ricci curvature, set to [] to run for all edges in G. (Default value = [])

        Returns
        -------
        output : dict[(int,int), float]
            A dictionary of edge Ricci curvature. E.g.: {(node1, node2): ricciCurvature}.
        """
        return _compute_nextrout_ricci_curvature_edges(
            G=self.G,
            weight=self.weight,
            edge_list=edge_list,
            alpha=self.alpha,
            method=self.method,
            base=self.base,
            exp_power=self.exp_power,
            shortest_path=self.shortest_path,
            nbr_topk=self.nbr_topk,
            beta_d=self.beta_d,
            tdens_gfvar=self.tdens_gfvar,
            explicit_implicit=self.explicit_implicit,
        )

    def compute_nextrout_ricci_curvature(self):
        # Adapted from: https://github.com/saibalmars/GraphRicciCurvature/blob/master/GraphRicciCurvature/OllivierRicci.py
        """Compute Ricci curvature of edges and nodes.
        The node Ricci curvature is defined as the average of node's adjacency edges.

        Returns
        -------
        G: NetworkX graph
            A NetworkX graph with "ricciCurvature" on nodes and edges.

        Examples
        --------
        To compute the Ollivier-Ricci curvature for karate club graph::

            >>> G = nx.karate_club_graph()
            >>> orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
            >>> orc.compute_ricci_curvature()
            >>> orc.G[0][1]
            {'weight': 1.0, 'ricciCurvature': 0.11111111071683011}
        """

        self.G = _compute_nextrout_ricci_curvature(
            G=self.G,
            weight=self.weight,
            alpha=self.alpha,
            method=self.method,
            base=self.base,
            exp_power=self.exp_power,
            shortest_path=self.shortest_path,
            nbr_topk=self.nbr_topk,
            beta_d=self.beta_d,
            tdens_gfvar=self.tdens_gfvar,
            explicit_implicit=self.explicit_implicit,
        )
        return self.G

    def compute_nextrout_ricci_flow(
        self,
        iterations=10,
        step=1,
        delta=1e-4,
        surgery=(lambda G, *args, **kwargs: G, 100),
    ):
        # Adapted from: https://github.com/saibalmars/GraphRicciCurvature/blob/master/GraphRicciCurvature/OllivierRicci.py
        """Compute the given Ricci flow metric of each edge of a given connected NetworkX graph.

        Parameters
        ----------
        iterations : int
            Iterations to require Ricci flow metric. (Default value = 10)
        step : float
            Step size for gradient decent process. (Default value = 1)
        delta : float
            Process stop when difference of Ricci curvature is within delta. (Default value = 1e-4)
        surgery : (function, int)
            A tuple of user define surgery function that will execute every certain iterations.
            (Default value = (lambda G, *args, **kwargs: G, 100))

        Returns
        -------
        G: NetworkX graph
            A graph with ``weight`` as Ricci flow metric.

        Examples
        --------
        To compute the Ollivier-Ricci flow for karate club graph::

            >>> G = nx.karate_club_graph()
            >>> orc_OTD = OllivierRicci(G, alpha=0.5, method="OTD", verbose="INFO")
            >>> orc_OTD.compute_ricci_flow(iterations=10)
            >>> orc_OTD.G[0][1]
            {'weight': 0.06399135316908759,
             'ricciCurvature': 0.18608249978652802,
             'original_RC': 0.11111111071683011}
        """
        self.G = _compute_nextrout_ricci_flow(
            G=self.G,
            weight=self.weight,
            iterations=iterations,
            step=step,
            delta=delta,
            surgery=surgery,
            alpha=self.alpha,
            method=self.method,
            base=self.base,
            exp_power=self.exp_power,
            shortest_path=self.shortest_path,
            nbr_topk=self.nbr_topk,
            beta_d=self.beta_d,
            tdens_gfvar=self.tdens_gfvar,
            explicit_implicit=self.explicit_implicit,
        )
        return self.G

    def ricci_community(self, cutoff_step=0.025, drop_threshold=0.01):
        # Adapted from: https://github.com/saibalmars/GraphRicciCurvature/blob/master/GraphRicciCurvature/OllivierRicci.py
        """Detect community clustering by Ricci flow metric.
        The communities are detected by the modularity drop while iteratively remove edge weight (Ricci flow metric)
        from large to small.

        Parameters
        ----------
        cutoff_step: float
            The step size to find the good cutoff points.
        drop_threshold: float
            At least drop this much to considered as a drop for good_cut.

        Returns
        -------
        cutoff: float
            Ricci flow metric weight cutoff for detected community clustering.
        clustering : dict
            Detected community clustering.

        Examples
        --------
        To compute the Ricci community for karate club graph::

            >>> G = nx.karate_club_graph()
            >>> orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
            >>> orc.compute_ricci_flow(iterations=50)
            >>> cc = orc.ricci_community()
            >>> print("The detected community label of node 0: %s" % cc[1][0])
            The detected community label of node 0: 0
        """

        cc = self.ricci_community_all_possible_clusterings(
            cutoff_step=cutoff_step, drop_threshold=drop_threshold
        )
        assert cc, "No clustering found!"

        number_of_clustering = len(set(cc[-1][1].values()))
        # logger.info("Communities detected: %d" % number_of_clustering)

        return cc[-1]

    def ricci_community_all_possible_clusterings(
        self, cutoff_step=0.025, drop_threshold=0.01
    ):
        # Adapted from: https://github.com/saibalmars/GraphRicciCurvature/blob/master/GraphRicciCurvature/OllivierRicci.py
        """Detect community clustering by Ricci flow metric (all possible clustering guesses).
        The communities are detected by Modularity drop while iteratively remove edge weight (Ricci flow metric)
        from large to small.

        Parameters
        ----------
        cutoff_step: float
            The step size to find the good cutoff points.
        drop_threshold: float
            At least drop this much to considered as a drop for good_cut.

        Returns
        -------
        cc : list of (float, dict)
            All detected cutoff and community clusterings pairs. Clusterings are detected by detected cutoff points from
            large to small. Usually the last one is the best clustering result.

        Examples
        --------
        To compute the Ricci community for karate club graph::

            >>> G = nx.karate_club_graph()
            >>> orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
            >>> orc.compute_ricci_flow(iterations=50)
            >>> cc = orc.ricci_community_all_possible_clusterings()
            >>> print("The number of possible clusterings: %d" % len(cc))
            The number of possible clusterings: 3
        """

        if not nx.get_edge_attributes(self.G, "original_RC"):
            # logger.info(
            #    "Ricci flow not detected yet, run Ricci flow with default setting first..."
            # )
            self.compute_ricci_flow()

        # logger.info("Ricci flow detected, start cutting graph into community...")
        cut_guesses = util.get_rf_metric_cutoff(
            self.G,
            weight=self.weight,
            cutoff_step=cutoff_step,
            drop_threshold=drop_threshold,
        )

        assert cut_guesses, "No cutoff point found!"

        Gp = self.G.copy()
        cc = []
        for cut in cut_guesses[::-1]:
            Gp = util.cut_graph_by_cutoff(Gp, cutoff=cut, weight=self.weight)
            # Get connected component after cut as clustering
            cc.append(
                (
                    cut,
                    {
                        c: idx
                        for idx, comp in enumerate(nx.connected_components(Gp))
                        for c in comp
                    },
                )
            )

        return cc


def _get_single_node_neighbors_distributions(node):
    # Adapted from: https://github.com/saibalmars/GraphRicciCurvature/blob/master/GraphRicciCurvature/OllivierRicci.py
    """Get the neighbor density distribution of given node `node`.

    Parameters
    ----------
    node : int
        Node index in Networkit graph `_Gk`.

    Returns
    -------
    distributions : lists of float
        Density distributions of neighbors up to top `_nbr_topk` nodes.
    nbrs : lists of int
        Neighbor index up to top `_nbr_topk` nodes.

    """

    neighbors = list(_Gk.iterNeighbors(node))

    # Get sum of distributions from x's all neighbors
    heap_weight_node_pair = []
    for nbr in neighbors:

        w = _base ** (-_Gk.weight(node, nbr) ** _exp_power)

        if len(heap_weight_node_pair) < _nbr_topk:
            heapq.heappush(heap_weight_node_pair, (w, nbr))
        else:
            heapq.heappushpop(heap_weight_node_pair, (w, nbr))

    nbr_edge_weight_sum = sum([x[0] for x in heap_weight_node_pair])

    if not neighbors:
        # No neighbor, all mass stay at node
        return [1], [node]

    if nbr_edge_weight_sum > EPSILON:
        # Sum need to be not too small to prevent divided by zero
        distributions = [
            (1.0 - _alpha) * w / nbr_edge_weight_sum for w, _ in heap_weight_node_pair
        ]
    else:
        # Sum too small, just evenly distribute to every neighbors
        distributions = [(1.0 - _alpha) / len(heap_weight_node_pair)] * len(
            heap_weight_node_pair
        )

    nbr = [x[1] for x in heap_weight_node_pair]
    return distributions + [_alpha], nbr + [node]


def find_wass(
    G,
    rhs,
    weight_flag,
    stopThres,
    beta_d=1,
    MaxNumIter=20,
    tdens_gfvar="tdens",
    explicit_implicit="explicit",
):

    """
    Function to compute the beta-Wasserstein measure.

    Parameters
    ----------
    G : NetworkX graph
        A given NetworkX graph with node attribute "clustering_label" as ground truth.
    rhs : numpy array of size equal to the number of nodes of G
        Forcing term with source and target distribution information.
    weight_flag: numpy array of size number of edges of G.
        Edge weight information.
    stopThres : float
        Accuracy used to solve the OT problem.
    beta: float
        (Optional) traffic rate needed for the Nextrout method.
    tdens_gfvar : str
        (Optional) string used to denote the dynamic method of the solver (either "tdens" or "gfvar")
    explicit_implicit: str
        (Optional) Numerical method used to solve the DMK system (either "implicit" or "explicit").
    Returns
    -------
    fluxes: numpy array
        Optimal transport path defined on the edges, solution of the OT problem.

    """
    mapping = {}
    pos = nx.spring_layout(G)
    for idx, node in enumerate(G.nodes()):
        mapping[node] = idx
    nx.set_node_attributes(G, pos, "pos")
    G_rel = nx.relabel_nodes(G, mapping, copy=True)

    Gf, fluxes, colors, inputs = fn.filtering(
        G_rel,
        beta_d=beta_d,
        weight_flag=weight_flag,
        threshold=0,
        BPweights="flux",
        rhs=rhs,
        stopping_threshold_f=stopThres,
        MaxNumIter=MaxNumIter,
        tdens_gfvar=tdens_gfvar,
        explicit_implicit=explicit_implicit,
    )

    return fluxes


def community_detection(
    g,
    graph_model,
    met,
    num_iter=10,
    beta=1,
    plotting=True,
    saving=True,
    path2save="../data/",
    K=2,
    alpha=0,
    base=1,
    exp_power=0,
):
    """
    Computes community hard membership function. The dictionary community_mapping maps nodes to community labels.

    Parameters
    ----------
    G : NetworkX graph
        A given NetworkX graph with node attribute "clustering_label" as ground truth.
    graph model : str
        Model used to generate the network ('SBM', 'LFR').
    met : str
        Method used to compute the memberships.
    num_iter: int
        (Optional) number of iterations need for the Ricci flow-based algorithms.
    beta: float
        (Optional) traffic rate needed for the Nextrout method.
    plotting: bool
        (Optional) show plots.
    path2save: str
        (Optional) path to save results.
    K: int
        (Not required for Nextrout, mandatory for MT) Number of communities
    alpha: float
        Amount of mass to be left at central nodes
    base, exp_power: float, int
        base^exp_power used to weigh node distributions.
    Returns
    -------
    [g, community_mapping, cutoff]: list
        g: NetworkX graph
            Input graph with modified edge weights if Ricci flow-based methods are used.
        communtiy_mapping: dict
            Dictionary from G.nodes to community labels.
        cutoff: float
            Threshold used to cut edge off the graph. Only used for Ricci flow-based methods.

    """
    g_copy = g.copy()

    if graph_model == "LFR":

        g_copy = set_block_label(g_copy)

    if met == "Nextrout":

        for itnum in range(1, num_iter + 1):

            # Cleaning the class info
            importlib.reload(OR)

            # Reinitializing the class with the generated graph
            orf = ORC_Nextrout(g, alpha=alpha, base=base, exp_power=exp_power)

            # We define the method (by default, SinkhornMix)
            orf.method = met

            # Defining beta (only makes a difference for 'Nextrout' method)
            orf.beta_d = beta

            # Do Ricci flow 'num_iter' times
            orf.compute_nextrout_ricci_flow(iterations=1)

            # Accesing the ORC attribute and making it initial condition
            g = orf.G

            edges_to_print = list(g.edges(data=True))
            # print(edges_to_print[:10])

            # Removing the long edges, and thus finding the communities
            cc = orf.ricci_community()

            # Accessing the communties, cc[0] is a 'good cutoff'
            cutoff = cc[0]
            community_mapping = cc[1]

            if saving:

                # Storing the results
                if "Nextrout" in met:
                    met_name = met + "_beta=" + str(beta)
                else:
                    met_name = met

                results = open(
                    path2save
                    + "/communities_dict_"
                    + met_name
                    + "_num_iter_"
                    + str(itnum)
                    + "_"
                    + graph_model
                    + ".pkl",
                    "wb",
                )
                pkl.dump([g, community_mapping, cutoff], results)

                print(
                    "iteration # {:} is stored at {:}".format(
                        itnum,
                        path2save
                        + "communities_dict_"
                        + met_name
                        + "_num_iter_"
                        + str(itnum)
                        + "_"
                        + graph_model
                        + ".pkl",
                    )
                )

    elif met == "MT":

        name = "adj_{:}.dat".format(graph_model)
        # print('Looking for ' +str(K)+' communities')

        comm_label = utils.get_label(graph_model)

        u_norm, u_gt = mt_tools.MT_comm_dect(g, K, comm_label, name=name)
        print("=====K", K)
        community_mapping = u_norm  # renaming
        g = u_gt  # renaming
        cutoff = None

    else:
        assert False

    if plotting:
        # plotting the found communities
        try:
            test = pos[list(g.nodes())[0]]
        except:
            pos = nx.spectral_layout(g)
        colors = list(community_mapping.values())
        nx.draw(g, node_color=colors, edge_color="gray", with_labels=False)
        plt.show()
        print("here")
    if saving:
        # Storing the results
        if "Nextrout" in met:
            met_name = met + "_beta=" + str(beta)
        else:
            met_name = met

        results = open(
            path2save
            + "/communities_dict_"
            + met_name
            + "_num_iter_"
            + str(num_iter)
            + "_"
            + graph_model
            + ".pkl",
            "wb",
        )
        pkl.dump([g, community_mapping, cutoff], results)

    return [g, community_mapping, cutoff]
