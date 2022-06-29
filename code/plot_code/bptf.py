"""
Bayesian Poisson tensor factorization with variational inference.
Adapted from Aaron Schein's code
"""
import sys
import time
import numpy as np
import numpy.random as rn
import scipy.special as sp
import sktensor as skt
from sklearn.base import BaseEstimator, TransformerMixin
from time import sleep
import pickle
# from path import path
from argparse import ArgumentParser
# from utils import *
import utils as tl



class BPTF(BaseEstimator, TransformerMixin):
    def __init__(self, n_modes=3, n_components=100,  max_iter=200, tol=0.0001,undirected=False,end_file=None,
                 smoothness=100, verbose=True, alpha=0.1, debug=False,seed=10,N_real=1):
        self.n_modes = n_modes
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.verbose = verbose
        self.debug = debug
        self.N_real = N_real
        self.undirected = undirected
        self.end_file = end_file

        self.alpha = alpha                                      # shape hyperparameter
        self.beta_M = np.ones(self.n_modes, dtype=float)        # rate hyperparameter (inferred)

        self.gamma_DK_M = np.empty(self.n_modes, dtype=object)  # variational shapes
        self.delta_DK_M = np.empty(self.n_modes, dtype=object)  # variational rates

        self.E_DK_M = np.empty(self.n_modes, dtype=object)      # arithmetic expectations
        self.G_DK_M = np.empty(self.n_modes, dtype=object)      # geometric expectations

        self.E_DK_M_f = np.empty(self.n_modes, dtype=object)      # arithmetic final expectations
        self.G_DK_M_f = np.empty(self.n_modes, dtype=object)      # geometric final expectations
        self.elbo_f = -10000000 #final elbo value
        # Inference cache
        self.sumE_MK = np.empty((self.n_modes, self.n_components), dtype=float)
        self.zeta = None
        self.nz_recon_I = None
        self.elbo=[]
        self.seed=seed

    def _reconstruct_nz(self, subs_I_M):
        """Computes the reconstruction for only non-zero entries."""
        I = subs_I_M[0].size
        K = self.n_components
        nz_recon_IK = np.ones((I, K))
        for m in range(self.n_modes):
            nz_recon_IK *= self.G_DK_M[m][subs_I_M[m], :]
        self.nz_recon_I = nz_recon_IK.sum(axis=1)
        return self.nz_recon_I

    def _bound(self, data, mask=None, preprocess_data=False):
        """Computes the Evidence Lower Bound (ELBO)."""
        if preprocess_data==True:
            data = preprocess(data)
        if mask is None:
            uttkrp_K = self.sumE_MK.prod(axis=0)
        elif isinstance(mask, skt.dtensor):
            uttkrp_DK = mask.uttkrp(self.E_DK_M, 0)
            uttkrp_K = (self.E_DK_M[0] * uttkrp_DK).sum(axis=0)
        elif isinstance(mask, skt.sptensor):
            uttkrp_DK = sp_uttkrp(mask.vals, mask.subs, 0, self.G_DK_M)
            uttkrp_K = (self.E_DK_M[0] * uttkrp_DK).sum(axis=0)

        bound = uttkrp_K.sum()

        if isinstance(data, skt.dtensor):
            subs_I_M = data.nonzero()
            vals_I = data[subs_I_M]
        elif isinstance(data, skt.sptensor):
            subs_I_M = data.subs
            vals_I = data.vals 
        nz_recon_I = self._reconstruct_nz(subs_I_M)

        bound -= np.log(vals_I + 1).sum()
        bound += (vals_I * np.log(nz_recon_I)).sum()

        K = self.n_components
        for m in range(self.n_modes):
            D = self.mode_dims[m]
            shp = self.alpha
            rte = self.alpha * self.beta_M[m]
            gamma_DK = self.gamma_DK_M[m]
            delta_DK = self.delta_DK_M[m]

            bound += (shp - 1.) * (np.log(self.G_DK_M[m]).sum())
            bound -= rte * (self.sumE_MK[m, :].sum())
            bound -= K * D * (sp.gammaln(shp) - shp * np.log(rte))
            bound += (-(gamma_DK - 1.) * sp.psi(gamma_DK) - np.log(delta_DK)
                      + gamma_DK + sp.gammaln(gamma_DK)).sum()
        return bound

    def _init_all_components(self, mode_dims):
        assert len(mode_dims) == self.n_modes
        self.mode_dims = mode_dims
        for m, D in enumerate(mode_dims):
            self._init_component(m, D)

    def _init_component(self, m, dim):
        np.random.seed(self.seed)
        assert self.mode_dims[m] == dim
        K = self.n_components
        if not self.debug:
            s = self.smoothness
            gamma_DK = s * rn.gamma(s, 1. / s, size=(dim, K))
            delta_DK = s * rn.gamma(s, 1. / s, size=(dim, K))
        else:
            gamma_DK = np.ones((dim, K))
            delta_DK = np.ones((dim, K))
        self.gamma_DK_M[m] = gamma_DK
        self.delta_DK_M[m] = delta_DK
        self.E_DK_M[m] = gamma_DK / delta_DK
        self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self.G_DK_M[m] = np.exp(sp.psi(gamma_DK) - np.log(delta_DK))
        self.beta_M[m] = 1. / self.E_DK_M[m].mean()

    def _check_component(self, m):
        assert np.isfinite(self.E_DK_M[m]).all()
        assert np.isfinite(self.G_DK_M[m]).all()
        assert np.isfinite(self.gamma_DK_M[m]).all()
        assert np.isfinite(self.delta_DK_M[m]).all()

    def _update_gamma(self, m, data):
        if isinstance(data, skt.dtensor):
            tmp = data.astype(float)
            subs_I_M = data.nonzero()
            tmp[subs_I_M] /= self._reconstruct_nz(subs_I_M)
            uttkrp_DK = tmp.uttkrp(self.G_DK_M, m)

        elif isinstance(data, skt.sptensor):
            tmp = data.vals / self._reconstruct_nz(data.subs)
            uttkrp_DK = sp_uttkrp(tmp, data.subs, m, self.G_DK_M)

        self.gamma_DK_M[m][:, :] = self.alpha + self.G_DK_M[m] * uttkrp_DK

    def _update_delta(self, m, mask=None):
        if mask is None:
            self.sumE_MK[m, :] = 1.
            uttrkp_DK = self.sumE_MK.prod(axis=0)
        else:
            uttrkp_DK = mask.uttkrp(self.E_DK_M, m)
        self.delta_DK_M[m][:, :] = self.alpha * self.beta_M[m] + uttrkp_DK

    def _update_cache(self, m):
        gamma_DK = self.gamma_DK_M[m]
        delta_DK = self.delta_DK_M[m]
        self.E_DK_M[m] = gamma_DK / delta_DK
        self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self.G_DK_M[m] = np.exp(sp.psi(gamma_DK)) / delta_DK

    def _update_beta(self, m):
        self.beta_M[m] = 1. / self.E_DK_M[m].mean()

    def _update(self, data, mask=None, modes=None):
        if modes is not None:
            modes = list(set(modes))
        else:
            modes = range(self.n_modes)
        assert all(m in range(self.n_modes) for m in modes)

        curr_elbo = -np.inf
        for itn in range(self.max_iter):
            s = time.time()
            for m in modes:
                self._update_gamma(m, data)
                self._update_delta(m, mask)
                self._update_cache(m)
                self._update_beta(m)  # must come after cache update!
                self._check_component(m)
            bound = self._bound(data, mask=mask)
            delta = (bound - curr_elbo) / abs(curr_elbo) if itn > 0 else np.nan
            e = time.time() - s
            if self.verbose:
                print('ITERATION %d:\t\
                       Time: %f\t\
                       Objective: %.2f\t\
                       Change: %.5f\t'\
                       % (itn, e, bound, delta))
            # if delta<0.0 and itn>0:
            #     break
            # assert ((delta >= 0.0) or (itn == 0))
            curr_elbo = bound
            self.elbo.append(bound)
            if abs(delta) < self.tol:
                break

    def _update_optimal_parameters(self, it=None):
        self.G_DK_M_f=np.copy(self.G_DK_M)
        self.E_DK_M_f=np.copy(self.E_DK_M)
        if it is None: it=self.max_iter
        self.it_f = it
        self.elbo_f = self.elbo[-1]

    def set_component(self, m, E_DK, G_DK, gamma_DK, delta_DK):
        assert E_DK.shape[1] == self.n_components
        self.E_DK_M[m] = E_DK.copy()
        self.sumE_MK[m, :] = E_DK.sum(axis=0)
        self.G_DK_M[m] = G_DK.copy()
        self.gamma_DK_M[m] = gamma_DK.copy()
        self.delta_DK_M[m] = delta_DK.copy()
        self.beta_M[m] = 1. / E_DK.mean()

    def set_component_like(self, m, model, subs_D=None):                           
        assert model.n_modes == self.n_modes
        assert model.n_components == self.n_components
        D = model.E_DK_M[m].shape[0]
        if subs_D is None:
            subs_D = np.arange(D)
        assert min(subs_D) >= 0 and max(subs_D) < D
        E_DK = model.E_DK_M[m][subs_D, :].copy()
        G_DK = model.G_DK_M[m][subs_D, :].copy()
        gamma_DK = model.gamma_DK_M[m][subs_D, :].copy()
        delta_DK = model.delta_DK_M[m][subs_D, :].copy()
        self.set_component(m, E_DK, G_DK, gamma_DK, delta_DK)

    def fit(self, data, mask=None):
        assert data.ndim == self.n_modes
        data = preprocess(data)
        if mask is not None:
            mask = preprocess(mask)
            assert data.shape == mask.shape
            assert is_binary(mask)
            assert np.issubdtype(mask.dtype, int)
        '''
        Cycle_over_realizations
        '''
        maxL=-1000000000;
        for r in range(self.N_real):
            self._init_all_components(data.shape)
            self._update(data, mask=mask)
          
            if(maxL<self.elbo[-1]): 
                self._update_optimal_parameters(it=len(self.elbo))
                maxL=self.elbo[-1]
            self.elbo=[]
            self.seed+=1

        return self

    def transform(self, modes, data, mask=None, version='geometric'):
        """Transform new data given a pre-trained model."""
        assert all(m in range(self.n_modes) for m in modes)
        assert (version == 'geometric') or (version == 'arithmetic')

        assert data.ndim == self.n_modes
        data = preprocess(data)
        if mask is not None:
            mask = preprocess(mask)
            assert data.shape == mask.shape
            assert is_binary(mask)
            assert np.issubdtype(mask.dtype, int)
        self.mode_dims = data.shape
        for m, D in enumerate(self.mode_dims):
            if m not in modes:
                if self.E_DK_M[m].shape[0] != D:
                    raise ValueError('Pre-trained components dont match new data.')
            else:
                self._init_component(m, D)
        self._update(data, mask=mask, modes=modes)

        if version == 'geometric':
            return [self.G_DK_M[m] for m in modes]
        elif version == 'arithmetic':
            return [self.E_DK_M[m] for m in modes]

    def fit_transform(self, modes, data, mask=None, version='geometric'):
        assert all(m in range(self.n_modes) for m in modes)
        assert (version == 'geometric') or (version == 'arithmetic')

        self.fit(data, mask=mask)

        if version == 'geometric':
            return [self.G_DK_M[m] for m in modes]
        elif version == 'arithmetic':
            return [self.E_DK_M[m] for m in modes]

    def reconstruct(self, mask=None, version='geometric'):
        """Reconstruct data using point estimates of latent factors.

        Currently supported only up to 5-way tensors.
        """
        assert (version == 'geometric') or (version == 'arithmetic')
        if version == 'geometric':
            tmp = [G_DK.copy() for G_DK in self.G_DK_M]
        elif version == 'arithmetic':
            tmp = [E_DK.copy() for E_DK in self.E_DK_M]

        if weights.keys():
            assert all(m in range(self.n_modes) for m in weights.keys())
            for m, weight_matrix in weights.iteritems():
                tmp[m] = weight_matrix
        Y_pred = parafac(tmp)
        if drop_diag:
            diag_idx = np.identity(Y_pred.shape[0]).astype(bool)
            Y_pred[diag_idx] = 0
        return Y_pred


def main():
    p = ArgumentParser()
    p.add_argument('-d', '--data', type=path, required=True)
    p.add_argument('-o', '--out', type=path, required=True)
    p.add_argument('-m', '--mask', type=path, default=None)
    p.add_argument('-k', '--n_components', type=int, required=True)
    p.add_argument('-n', '--max_iter', type=int, default=200)
    p.add_argument('-t', '--tol', type=float, default=1e-4)
    p.add_argument('-s', '--smoothness', type=int, default=100)
    p.add_argument('-a', '--alpha', type=float, default=0.1)
    p.add_argument('-v', '--verbose', action="store_true", default=False)
    p.add_argument('--debug', action="store_true", default=False)
    args = p.parse_args()

    args.out.makedirs_p()
    assert args.data.exists() and args.out.exists()
    if args.data.ext == '.npz':
        data_dict = np.load(args.data)
        if 'data' in data_dict.files:
            data = data_dict['data']
        elif 'Y' in data_dict.files:
            data = data_dict['Y']
        if data.dtype == 'object':
            assert data.size == 1
            data = data[0]
    else:
        data = np.load(args.data)

    valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
    assert any(isinstance(data, vt) for vt in valid_types)

    mask = None
    if args.mask is not None:
        if args.mask.ext == '.npz':
            mask = np.load(args.mask)['mask']
            if mask.dtype == 'object':
                assert mask.size == 1
                mask = mask[0]
        else:
            mask = np.load(args.mask)

        assert any(isinstance(mask, vt) for vt in valid_types)
        assert mask.shape == data.shape

    bptf = BPTF(n_modes=data.ndim,
                n_components=args.n_components,
                max_iter=args.max_iter,
                tol=args.tol,
                smoothness=args.smoothness,
                verbose=args.verbose,
                alpha=args.alpha,
                debug=args.debug)

    bptf.fit(data, mask=mask)
    serialize_bptf(bptf, args.out, num=None, desc='trained_model')


if __name__ == '__main__':
    main()

def serialize_bptf(model, out_dir, num=None, desc=None):
    if desc is None:
        desc = 'model'
    out_dir = path(out_dir)
    assert out_dir.exists()

    if num is None:
        sleep(rn.random() * 5)
        curr_files = out_dir.files('*_%s.npz' % desc)
        curr_nums = [int(f.namebase.split('_')[0]) for f in curr_files]
        num = max(curr_nums + [0]) + 1

    with open(out_dir.joinpath('%d_%s.dat' % (num, desc)), 'wb') as f:
        pickle.dump(model.get_params(), f)

    out_path = out_dir.joinpath('%d_%s.npz' % (num, desc))
    np.savez(out_path,
             E_DK_M=model.E_DK_M,
             G_DK_M=model.G_DK_M,
             gamma_DK_M=model.gamma_DK_M,
             delta_DK_M=model.delta_DK_M,
             beta_M=model.beta_M)
    print(out_path)
    return num



def sp_uttkrp(vals, subs, m, U):
    """Alternative implementation of the sparse version of the uttkrp.
    ----------
    subs : n-tuple of array-likes
        Subscripts of the nonzero entries in the tensor.
        Length of tuple n must be equal to dimension of tensor.
    vals : array-like
        Values of the nonzero entries in the tensor.
    U : list of array-likes
        Matrices for which the Khatri-Rao product is computed and
        which are multiplied with the tensor in mode `mode`.
    m : int
        Mode in which the Khatri-Rao product of `U` is multiplied
        with the tensor.
    Returns
    -------
    out : np.ndarray
        Matrix which is the result of the matrix product of the unfolding of
        the tensor and the Khatri-Rao product of `U`.
    """
    D, K = U[m].shape
    out = np.zeros_like(U[m])
    for k in range(K):
        tmp = vals.copy()
        for mode, matrix in enumerate(U):
            if mode == m:
                continue
            tmp *= matrix[subs[mode], k].astype(tmp.dtype)
        out[:, k] += np.bincount(subs[m],
                                 weights=tmp,
                                 minlength=D)
    return out

def preprocess(X):
    """Preprocesses input data tensor.

    If data is sparse, returns an int sptensor.
    Otherwise, returns an int dtensor.
    """
    if not X.dtype == np.dtype(int).type:
        X = X.astype(int)
    if isinstance(X, np.ndarray) and is_sparse(X):
        X = sptensor_from_dense_array(X)
    else:
        X = skt.dtensor(X)
    return X

def sptensor_from_dense_array(X):
    """Creates an sptensor from an ndarray or dtensor."""
    subs = X.nonzero()
    vals = X[subs]
    return skt.sptensor(subs, vals, shape=X.shape, dtype=X.dtype)

def is_binary(X):
    """Checks whether input is a binary integer tensor."""
    if np.issubdtype(X.dtype, int):
        if isinstance(X, skt.sptensor):
            return (X.vals == 1).all()
        else:
            return (X <= 1).all() and (X >= 0).all()
    else:
        return False

def is_sparse(X):
    """Checks whether input tensor is sparse.

    Implements a heuristic definition of sparsity.

    A tensor is considered sparse if:

    M = number of modes
    S = number of entries
    I = number of non-zero entries

            N > M(I + 1)
    """
    M = X.ndim
    S = X.size
    I = X.nonzero()[0].size
    return S > (I + 1) * M