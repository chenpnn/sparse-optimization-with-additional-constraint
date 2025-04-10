import numpy as np

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
        loss_fn,
        grad_fn,
        w_init=None,
        L_init=1,
        gamma_u=2,
        gamma_d=2,
        tol=1e-5,
        min_iter=10,
        max_iter=5e3,
        verbose=-1,
    ):
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.w_init = w_init
        self.L_init = L_init
        self.gamma_u = gamma_u
        self.gamma_d = gamma_d
        self.tol = tol
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.verbose = verbose

    # def loss_fn(self, X, y, w):
    #     loss = np.mean((X @ w - y) ** 2) / 2
    #     return loss
    
    # def grad_fn(self, X, y, w):
    #     n = len(y)
    #     grad = X.T @ (X @ w - y) / n
    #     return grad

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
        loss = self.loss_fn(X, y, w)
        grad = self.grad_fn(X, y, w)
        while True:
            T = proj_simplex(w - grad / L)  # projected gradient descent using the current guess of Lipschitz constant
            obj_val = self.loss_fn(X, y, T)  # value of the loss fuction
            model_val = loss + grad @ (T-w) + L * np.sum((T-w)**2) / 2  # value of quadratic model
            if obj_val > model_val:  # not decrease sufficiently
                L *= self.gamma_u  # increase the guess of Lipschitz constant
            else:
                break
        grad_new = self.grad_fn(X, y, T)
        res = np.max(np.minimum((grad_new - np.min(grad_new)) / L, T)) 
        # print(res.round(5), obj_val.round(5))
        return T, L, res
    
    def solve(self, X, y):
        n, p = X.shape
        if self.w_init is None:
            w_init = np.ones(p) / p
        else:
            w_init = self.w_init
        L = self.L_init
        w = w_init
        i = 0
        while True:
            i += 1
            T, M, res = self.grad_iter(X, y, w, L)  # iterate once
            
            
            w = T  # update the iterate
            L = np.max([self.L_init, M / self.gamma_d])  # update the guess of Lipschitz constant
            if res < self.tol:
                break
            elif i > self.max_iter:
                print("The iterates may not convergent.")
                break
        self.w = w
        self.loss = self.loss_fn(X, y, w)
        return self

if __name__ == "__main__":
    n, p, s = 500, 100000, 5
    X = np.random.randn(n, p)
    w_true = np.zeros(p)
    w_true[:s] = 1 / s
    noise = np.random.randn(n) * 0.1
    y = X @ w_true + noise
    

    def loss_fn(X, y, w):
        loss = np.mean((X @ w - y) ** 2) / 2
        return loss
    
    def grad_fn(X, y, w):
        n = len(y)
        grad = X.T @ (X @ w - y) / n
        return grad

    solver = SimplexSolver(loss_fn, grad_fn)
    solver = solver.solve(X, y)
    solver.w[:s]

    