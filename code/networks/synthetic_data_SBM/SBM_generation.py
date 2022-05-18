import math
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from scipy.stats import poisson
import random


EPS = 1e-12

class SyntNetSBM(object):

    def __init__(self, N = 100, K = 2, seed = 10, avg_degree = 10.,
                structure = 'assortative', label = 'test', a=0.1,
                verbose = 0, folder = '../../data/input', output_parameters = False,
                output_adj = False, outfile_adj = 'None'):

        # Set network size (node number)
        self.N = N
        # Set seed random number generator
        self.seed = seed
        # Set label (associated uniquely with the set of inputs)
        self.label = label
        # Initialize data folder path
        self.folder = folder
        # Set flag for storing the parameters
        self.output_parameters = output_parameters
        # Set flag for storing the generated adjacency matrix
        self.output_adj = output_adj
        # Set required average degree
        self.avg_degree = avg_degree
        self.a = a # outgroup probability

        # Set verbosity flag
        if verbose > 2 and not isinstance(verbose, int):
            raise ValueError('The verbosity parameter can only assume values in {0,1,2}!')
        self.verbose = verbose

        ### Set MT inputs
        # Set the affinity matrix structure
        if structure not in ['assortative', 'disassortative']:
            raise ValueError('The available structures for the affinity matrix w '
                            'are: assortative, disassortative!')
        self.structure = structure
        # Set number of communities
        self.K = K
        

    def synthetic_network(self, parameters = None):

        # Set seed random number generator
        prng = np.random.RandomState(self.seed)

        ### Latent variables
        if parameters is None:
            # Generate latent variables
            self.u, self.w = self._generate_lv(prng,seed = self.seed)
        else:
            # Set latent variables
            self.u, self.w = parameters


        # Compute M_ij
        M = np.einsum('ik,jq->ijkq', self.u, self.u)
        M = np.einsum('ijkq,kq->ij', M, self.w)
        
        c = ( float(self.N) * self.avg_degree * 0.5 ) / M.sum() 
        #c = 1.
        # Generate edges
        A = prng.poisson( c * M )
        A[A>0] = 1 # binarize the adjecancy matrix
        np.fill_diagonal(A, 0)
        A = np.triu(A) + np.triu(A).T

        assert np.allclose(A,A.T,rtol=1e-8,atol=1e-8)

        G = nx.to_networkx_graph(A, create_using=nx.DiGraph)
        nodes = list(G.nodes())

        # Keep largest connected component
        Gc = max(nx.weakly_connected_components(G), key=len)
        nodes_to_remove = set(G.nodes()).difference(Gc)
        G.remove_nodes_from(list(nodes_to_remove))

        # Update quantities by keeping only connected components
        nodes = list(G.nodes())
        self.N = len(nodes)

        A = nx.to_scipy_sparse_matrix(G, nodelist=nodes, weight='weight') #
        
        self.u = self.u[nodes]
        self.w *= c
        
        if self.verbose > 0:
            ave_deg = np.round(2 * G.number_of_edges() / float(G.number_of_nodes()), 3)
            print(f'Number of nodes: {G.number_of_nodes()} \n'
                f'Number of edges: {G.number_of_edges()}')
            print(f'Average degree (2E/N): {ave_deg}')

        if self.output_parameters:
            self._output_results(nodes)

        if self.output_adj:
            self._output_adjacency(G, outfile = self.outfile_adj)

        if self.verbose == 2:
            self._plot_A(A)
            self._plot_M(c * M,title='M')

        return A,G

    def _generate_lv(self, prng=None,seed = 10):
        if prng is None: prng = np.random.RandomState(seed)
        
        # Generate u, v for overlapping communities
        u = membership_vectors(seed, self.K,self.N)
        # Generate w
        w = affinity_matrix(self.structure, self.N, self.K, self.avg_degree, a = self.a)
        return u, w

    def _output_results(self, nodes):
        """
            Output results in a compressed file.
            INPUT
            ----------
            nodes : list
                    List of nodes IDs.
        """
        output_parameters = self.folder + 'N' + str(self.N) +'K' + str(self.K)  +'_theta_syn_' + self.label + '_' + str(self.seed) 
        # print(self.z.count_nonzero())
        np.savez_compressed(output_parameters + '.npz',  u=self.u, w=self.w, nodes=nodes)
        if self.verbose:
            print()
            print(f'Parameters saved in: {output_parameters}.npz')
            print('To load: theta=np.load(filename), then e.g. theta["u"]')

    def _output_adjacency(self, G, outfile = None):
        """
            Output the adjacency matrix. Default format is space-separated .csv
            with 3 columns: node1 node2 weight
            INPUT
            ----------
            G: Digraph
            DiGraph NetworkX object.
            outfile: str
                    Name of the adjacency matrix.
        """
        if outfile is None:
            outfile = 'N' + str(self.N) + 'K' + str(self.K) + '_syn_' + self.label + '_' + str(self.seed) + '.dat'
        else:
            outfile = 'N' + str(self.N) + 'K' + str(self.K) + '_syn_' + self.label + '_' + str(self.seed) + '_' + outfile + '.dat'

        edges = list(G.edges(data=True))
        try:
            data = [[u, v, d['weight']] for u, v, d in edges]
        except:
            data = [[u, v, 1] for u, v, d in edges]

        df = pd.DataFrame(data, columns=['source', 'target', 'w'], index=None)
        df.to_csv(self.folder + outfile, index=False, sep=' ')
        if self.verbose:
            print(f'Adjacency matrix saved in: {self.folder + outfile}')

    def _plot_A(self, A, cmap = 'PuBuGn',title='Adjacency matrix'):
        """
            Plot the adjacency matrix produced by the generative algorithm.
            INPUT
            ----------
            A : Scipy array
                Sparse version of the NxN adjacency matrix associated to the graph.
            cmap : Matplotlib object
                Colormap used for the plot.
        """
        Ad = A.todense()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(Ad, cmap = plt.get_cmap(cmap))
        ax.set_title(title, fontsize = 15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()


    def _plot_M(self, M, cmap = 'PuBuGn',title='MT means matrix'):
        """
            Plot the M matrix produced by the generative algorithm. Each entry is the
            poisson mean associated to each couple of nodes of the graph.
            INPUT
            ----------
            M : Numpy array
                NxN M matrix associated to the graph. Contains all the means used
                for generating edges.
            cmap : Matplotlib object
                Colormap used for the plot.
        """
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.matshow(M, cmap = plt.get_cmap(cmap))
        ax.set_title(title, fontsize = 15)
        for PCM in ax.get_children():
            if isinstance(PCM, plt.cm.ScalarMappable):
                break
        plt.colorbar(PCM, ax=ax)
        plt.show()


def membership_vectors(prng = 10, K = 2, N = 100):
    prng = np.random.RandomState(prng)
    # Generate equal-size unmixed group membership
    size = int(N / K)
    u = np.zeros((N, K))
    for i in range(N):
        q = int(math.floor(float(i) / float(size)))
        if q == K:
            u[i, K - 1] = 1
        else:
            u[i, q] = 1
    return u

def affinity_matrix(structure = 'assortative', N = 100, K = 2, avg_degree = 10., a = 0.1):
    """
        Compute the KxK affinity matrix w with probabilities between and within groups.
        INPUT
        ----------
        structure : string
                    Structure of the network.
        N : int
            Number of nodes.
        K : int
            Number of communities.
        a : float
            Parameter for secondary probabilities.
        OUTPUT
        -------
        p : Numpy array
            Array with probabilities between and within groups. Element (k,h)
            gives the density of edges going from the nodes of group k to nodes of group h.
    """

    p1 = avg_degree * K / N

    if structure == 'assortative':
        p = p1 * a * np.ones((K,K))  # secondary-probabilities
        np.fill_diagonal(p, p1 * np.ones(K))  # primary-probabilities
    elif structure == 'disassortative':
        p = p1 * np.ones((K,K))   # primary-probabilities
        np.fill_diagonal(p, a * p1 * np.ones(K))  # secondary-probabilities

    return p


def normalize_nonzero_membership(u):
    """
        Given a matrix, it returns the same matrix normalized by row.
        INPUT
        ----------
        u: Numpy array
        Numpy Matrix.
        OUTPUT
        -------
        The matrix normalized by row.
    """

    den1 = u.sum(axis=1, keepdims=True)
    nzz = den1 == 0.
    den1[nzz] = 1.

    return u / den1
