import numpy as np
from sklearn.base import BaseEstimator
from numbers import Integral, Real
from sklearn.utils._param_validation import Hidden, Interval, StrOptions

def proj_simplex(y, a=1):
    """ project the vector onto the simplex whose sum is a[=1]"""
    y = np.array(y)
    p = y.shape[0]
    zeros = np.zeros(p)
    u = np.sort(y)[::-1]
    t = u + (a - np.cumsum(u)) / (np.arange(p) + 1)
    rho = np.sum(t > 0)
    lam = (t - u)[rho - 1]
    return np.maximum(y+lam, zeros)

class SimplexSolver(object):
    def __init__(
        self,
        w_init=None,
        L_init=1,
        gamma_u=2,
        gamma_d=2,
        tol=1e-7,
        min_iter=10,
        max_iter=1e3,
        verbose=-1,
    ):
        self.w_init = w_init
        self.L_init = L_init
        self.gamma_u = gamma_u
        self.gamma_d = gamma_d
        self.tol = tol
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.verbose = verbose

    def get_loss(self, X, y, w):
        loss = np.mean((X @ w - y) ** 2) / 2
        return loss
    
    def get_grad(self, X, y, w):
        n = len(y)
        grad = X.T @ (X @ w - y) / n
        return grad

    def grad_iter(self, X, y, w, M):
        r"""
        Parameters
        -----------
        X, y : data
        w : current iterate
        M : current guess of Lipschitz constant

        Return
        -------
        T : next iterate
        L : a new guess of Lipschitz constant
        """
        n, p = X.shape
        L = M
        loss = self.get_loss(X, y, w)
        grad = self.get_grad(X, y, w)
        while True:
            T = proj_simplex(w - grad / L)  # projected gradient descent using the current guess of Lipschitz constant
            obj_val = self.get_loss(X, y, T)  # value of the loss fuction
            model_val = loss + grad @ (T-w) + L * np.sum((T-w)**2) / 2  # value of quadratic model
            if obj_val > model_val:  # not decrease sufficiently
                L *= self.gamma_u  # increase the guess of Lipschitz constant
            else:
                break
        return T, L
    
    def solve(self, X, y):
        n, p = X.shape
        if self.w_init is None:
            w_init = np.ones(p) / p
        else:
            w_init = self.w_init
        L = self.L_init
        w = w_init
        i = 0
        self.iterates, self.diffs = [], []
        while True:
            i += 1
            T, M = self.grad_iter(X, y, w, L)  # iterate once
            diff = np.linalg.norm(T-w, ord=2)  
            if self.verbose > 0:
                self.iterates.append(w)
                self.diffs.append(diff)
            
            w = T  # update the iterate
            L = np.max([self.L_init, M / self.gamma_d])  # update the guess of Lipschitz constant
            if (diff < self.tol) and (i > self.min_iter):
                break
            if (i > self.max_iter) & (self.verbose > 0):
                print("The iterates may not convergent.")
                break
        self.iterates = np.array(self.iterates)
        self.diffs = np.array(self.diffs)
        self.coef_ = w
        self.loss = self.get_loss(X, y, w)
        return self

class TailScreening(BaseEstimator):
    r"""
    Tail screening
    """

    _parameter_constraints: dict = {
        "sparsity": [Interval(Integral, 1, None, closed="left"), None],
        "L_init": [Interval(Real, 0, None, closed="neither")],
        "gamma_u": [Interval(Real, 1, None, closed="neither")],
        "gamma_d": [Interval(Real, 1, None, closed="neither")],
        "min_iter": [Interval(Integral, 1, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        sparsity=None,
        L_init=1,
        gamma_u=2,
        gamma_d=2,
        min_iter=10,
        max_iter=1e3,
        tol=None,
        tol_multiple=1e-4,
    ):
        self.sparsity = sparsity
        self.L_init = L_init
        self.gamma_u = gamma_u
        self.gamma_d = gamma_d
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.tol = tol
        self.tol_multiple = tol_multiple
        self.name=None
    
    def __str__(self):
        if self.name is None:
            return 'PERMITS'
        else:
            return self.name
    
    def refresh(self, w, supp):
        r"""
        w : subvector of shape (s,)
        supp : bool vector of shape (p,) but sum(supp) == s
        """
        w, supp = np.array(w), np.array(supp)
        assert np.sum(supp) == len(w)
        p, s = len(supp), len(w)
        w_new = np.zeros(p)
        w_new[supp] = w
        min_val = np.where(w_new==0, np.inf, w_new).min()
        w_new[w_new <= min_val] = 0
        w_new /= w_new.sum()
        supp_new = np.zeros(p).astype(bool)
        supp_new[np.nonzero(w_new)[0]] = True
        return w_new, supp_new

    
    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        n, p = X.shape
        if self.sparsity is None:
            s_min = 1
        elif type(self.sparsity) == int and self.sparsity >=1:
            s_min = self.sparsity
        else:
            raise Exception('sparsity should be a positive integer.')
        
        if self.tol is None:
            tol = self.tol_multiple * (1 + np.log(p)) / n
            self.tol =  tol
        else: 
            tol = self.tol
        
        w = np.ones(p) / p
        loss = np.mean((X @ w - y) ** 2) / 2
        supp = np.ones(p).astype(bool) # flag of support
        w_best, supp_best = w.copy(), supp.copy()
        ic_best = n * np.log(loss) + np.log(p) * np.log(np.log(n)) * supp.sum()
        
        while True:
            X_tmp = X[:, supp]
            w_tmp_init = w[supp] / w[supp].sum()
            solver = SimplexSolver(w_init=w_tmp_init, min_iter=self.min_iter, tol=tol)
            solver = solver.solve(X_tmp, y)
            w_tmp = solver.coef_
            loss = solver.loss
            ic = n * np.log(loss) + np.log(p) * np.log(np.log(n)) * supp.sum()
            if ic < ic_best:
                ic_best = ic
                w_best = w.copy()
                supp_best = supp.copy()
            if supp.sum() <= s_min:
                break
            else:
                w, supp = self.refresh(w_tmp, supp)  # update: drop the tail
        self.coef_ = w_best
        self.support_set = np.arange(p)[supp_best]
        return self
    
    def score(self, X, y):
        loss = np.mean((X @ self.coef_ - y) ** 2) / 2
        return - loss
    
