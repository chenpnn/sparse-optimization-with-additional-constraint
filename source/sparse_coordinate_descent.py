import numpy as np
import cvxpy as cp
from projected_gradient_descent import SimplexSolver


class SCD(object):
    r"""
    Sparse Coordinate Descent

    Parameters
    --------
    sparsity: int
        the sparsity level of the solution
    loss_fn: callable
        the loss function, loss_fn(X, y, w)
    grad_fn: callable
        the gradient function, grad_fn(X, y, w)
    tol: float, optional=1e-3
        tolerance for the sub-solver
    supp_init: list
        initialzed support set
    """
    def __init__(
        self,
        sparsity,
        loss_fn,
        grad_fn,
        tol=1e-3,
        supp_init=None,
    ):  
        self.s = sparsity
        # self.tol = tol
        self.supp_init = supp_init
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.sub_solver = SimplexSolver(loss_fn, grad_fn, tol=tol)

    def __str__(self):
        return "SCD"

    def solve(self, X, y):
        n, d = X.shape
        
        if self.supp_init is None:
            supp_set = np.random.choice(range(d), self.s, replace=False)
        else:
            supp_set = self.supp_init
        while True:
            non_supp_set = np.setdiff1d(np.arange(d), supp_set)
            w = np.zeros(d)
            w[supp_set] = self.sub_solver.solve(X[:, supp_set], y).w
            loss = self.sub_solver.loss
            i1 = supp_set[np.argmin(w[supp_set])]
            i0 = non_supp_set[np.argmin(self.grad_fn(X, y, w)[non_supp_set])]

            supp_set_new = np.setdiff1d(supp_set, [i1])
            supp_set_new = np.union1d(supp_set_new, [i0])
            # non_supp_set = np.setdiff1d(np.arange(d), supp_set)
            w_new = np.zeros(d)
            w_new[supp_set_new] = self.sub_solver.solve(X[:, supp_set_new], y).w
            loss_new = self.sub_solver.loss

            if loss_new < loss:
                supp_set = supp_set_new
            else:
                break
        self.supp_set = np.sort(supp_set)
        self.coef_ = w
        return self

if __name__ == "__main__":
    n, p, s = 500, 100000, 5
    X = np.random.randn(n, p)
    w_true = np.zeros(p)
    supp_true = np.sort(np.random.choice(range(p), s, replace=False))
    print(f"supp_true: {supp_true}")
    w_true[supp_true] = 1 / s
    noise = np.random.randn(n) * 0.1
    y = X @ w_true + noise
    

    def loss_fn(X, y, w):
        loss = np.mean((X @ w - y) ** 2) / 2
        return loss
    
    def grad_fn(X, y, w):
        n = len(y)
        grad = X.T @ (X @ w - y) / n
        return grad
    
    solver = SCD(sparsity=s, loss_fn=loss_fn, grad_fn=grad_fn)

    solver = solver.solve(X, y)
    print(f"supp_scd: {solver.supp_set}")
    solver.coef_[solver.supp_set]
    np.sum(solver.coef_)
    
    import abess
    model = abess.LinearRegression(support_size=s)
    model.fit(X, y)
    supp_abess = np.nonzero(model.coef_)[0]
    print(f"supp_abess: {supp_abess}")
    model.coef_[supp_abess]
    np.sum(model.coef_)




