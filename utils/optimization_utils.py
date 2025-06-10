import numpy as np
import time
import torch
from torch.utils.data import TensorDataset, random_split
import cvxpy as cp
# from cvxpylayers.torch import CvxpyLayer
import casadi as ca
from qpth.qp import QPFunction

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float64)


###################################################################
# Base PROBLEM
###################################################################
class BaseProblem:
    def __init__(self, dataset, val_size, test_size, seed):
        self.input_L = torch.tensor(dataset['XL'])
        self.input_U = torch.tensor(dataset['XU'])
        self.L = torch.tensor(dataset['YL'])
        self.U = torch.tensor(dataset['YU'])
        self.X = torch.tensor(dataset['X'])
        self.Y = torch.tensor(dataset['Y'])
        self.num = dataset['X'].shape[0]
        self.device = DEVICE

        total_size = self.X.shape[0]
        train_size = int(total_size  - val_size - test_size)
        full_dataset = TensorDataset(self.X, self.Y)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [train_size, val_size, test_size],
                                                                                generator=torch.Generator().manual_seed(seed))
        
    def eq_grad(self, X, Y):
        # Create a copy of Y that requires gradients for the whole batch
        Y_grad = Y.clone().detach().requires_grad_(True)
        
        # Compute equality residuals and their squares for the whole batch
        eq_resid = self.eq_resid(X, Y_grad) ** 2
        eq_penalty = torch.sum(eq_resid, dim=1, keepdim=True)
        
        # Compute gradients for all samples at once
        grad = torch.autograd.grad(
            outputs=eq_penalty,
            inputs=Y_grad,
            grad_outputs=torch.ones_like(eq_penalty),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]
        
        return grad

    def ineq_grad(self, X, Y):
        # Create a copy of Y that requires gradients
        Y_grad = Y.clone().detach().requires_grad_(True)
        
        # Compute inequality residuals and their squares for the whole batch
        ineq_resid = self.ineq_resid(X, Y_grad) ** 2
        ineq_penalty = torch.sum(ineq_resid, dim=1, keepdim=True)
        
        # Compute gradients for all samples at once
        grad = torch.autograd.grad(
            outputs=ineq_penalty,
            inputs=Y_grad,
            grad_outputs=torch.ones_like(ineq_penalty),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]
        
        return grad

    def scale_full(self, Y):
        lower_bound = self.L.view(1, -1)
        upper_bound = self.U.view(1, -1)
        # The last layer of NN is sigmoid, scale to Opt bound
        scale_Y = Y * (upper_bound - lower_bound) + lower_bound
        return scale_Y

    def scale_partial(self, Y):
        lower_bound = (self.L[self.partial_vars]).view(1, -1)
        upper_bound = (self.U[self.partial_vars]).view(1, -1)
        scale_Y = Y * (upper_bound - lower_bound) + lower_bound
        return scale_Y

    def scale(self, Y):
        if Y.shape[1] < self.ydim:
            Y_scale = self.scale_partial(Y)
        else:
            Y_scale = self.scale_full(Y)
        return Y_scale

    # def cal_penalty(self, X, Y):
    #     penalty = torch.cat([self.ineq_resid(X, Y), self.eq_resid(X, Y)], dim=1)
    #     return torch.abs(penalty)

    # def check_feasibility(self, X, Y):
    #     return self.cal_penalty(X, Y)


###################################################################
# QP PROBLEM
###################################################################
class QPProblem(BaseProblem):
    """
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   Gy <= h
                   L<= x <=U
    """

    def __init__(self, dataset, val_size, test_size, seed):
        super().__init__(dataset, val_size, test_size, seed)
        self.Q = torch.tensor(dataset['Q'])
        self.p = torch.tensor(dataset['p'])
        self.A = torch.tensor(dataset['A'])
        self.G = torch.tensor(dataset['G'])
        self.h = torch.tensor(dataset['h'])
        self.L = torch.tensor(dataset['YL'])
        self.U = torch.tensor(dataset['YU'])
        self.X = torch.tensor(dataset['X'])
        self.Y = torch.tensor(dataset['Y'])
        self.xdim = dataset['X'].shape[1]
        self.ydim = dataset['Q'].shape[0]
        self.neq = dataset['A'].shape[0]
        self.nineq = dataset['G'].shape[0]
        self.nknowns = 0

        best_partial = dataset['best_partial']
        self.partial_vars = best_partial
        self.partial_unknown_vars = best_partial
        self.other_vars = np.setdiff1d(np.arange(self.ydim), self.partial_vars)
        self.A_partial = self.A[:, self.partial_vars]
        self.A_other_inv = torch.inverse(self.A[:, self.other_vars])
        

    def __str__(self):
        return 'QPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )

    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * Y).sum(dim=1)
    
    def eq_resid(self, X, Y):
        return Y @ self.A.T - X
    
    def ineq_resid(self, X, Y):
        res = Y @ self.G.T - self.h.view(1, -1)
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)

    def complete_partial(self, X, Y):
        Y_full = torch.zeros(X.shape[0], self.ydim, device=X.device)
        Y_full[:, self.partial_vars] = Y
        Y_full[:, self.other_vars] = (X - Y @ self.A_partial.T) @ self.A_other_inv.T
        return Y_full   
    
    def qpth_projection(self, X, Y):
        batch_size = X.shape[0]
        n = self.ydim
        device = X.device
        dtype = X.dtype
        
        # Identity matrix for quadratic term (squared distance objective)
        Q = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, n, n)
        
        # Linear term: -y_pred
        p = -Y
                # Prepare inequality constraints: [G; I; -I] y <= [h; U; -L]
        G_top = self.G.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, self.nineq, n)
        G_middle = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, n, n)
        G_bottom = -torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, n, n)
        G_combined = torch.cat([G_top, G_middle, G_bottom], dim=1)
        
        h_top = self.h.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, self.nineq)
        h_middle = self.U.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, n)
        h_bottom = -self.L.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, n)
        h_combined = torch.cat([h_top, h_middle, h_bottom], dim=1)
        
        # Use QPFunction to solve the projection problem
        Y_proj = QPFunction(verbose=-1)(Q, p, G_combined, h_combined, self.A, X)
        
        return Y_proj


###################################################################
# QCQP Problem
###################################################################
class QCQPProblem(QPProblem):
    """
        minimize_y 1/2 * y^T Q y + p^Ty
        s.t.       Ay =  x
                   1/2 * y^T H y + G^T y <= h
                   L<= x <=U
    """
    def __init__(self, dataset, val_size, test_size, seed):
        super().__init__(dataset, val_size, test_size, seed)
        self.H = torch.tensor(dataset['H'])

    def __str__(self):
        return 'QCQPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    def ineq_resid(self, X, Y):
        res = []
        q = torch.matmul(self.H, Y.T).permute(2, 0, 1)
        q = (q * Y.view(Y.shape[0], 1, -1)).sum(-1)
        res = 0.5 * q + torch.matmul(Y, self.G.T) - self.h
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)
    

###################################################################
# SOCP Problem
###################################################################
class SOCPProblem(QPProblem):
    """
        minimize_y p^Ty
        s.t.       Ay =  x
                   ||G^T y + h||_2 <= c^Ty+d
                   L<= x <=U
    """

    def __init__(self, dataset, val_size, test_size, seed):
        super().__init__(dataset, val_size, test_size, seed)
        self.C = torch.tensor(dataset['C'] )
        self.d = torch.tensor(dataset['d'] )

    def __str__(self):
        return 'SOCPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    def ineq_resid(self, X, Y):
        res = []
        q = torch.norm(torch.matmul(self.G, Y.T).permute(2, 0, 1) + self.h.unsqueeze(0), dim=-1, p=2)
        p = torch.matmul(Y, self.C.T) + self.d
        res = q - p
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)


###################################################################
# NONCONVEX PROBLEM
###################################################################

class nonconvexQPProblem(QPProblem):
    def __str__(self):
        return 'QPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * torch.sin(Y)).sum(dim=1)
    
    def ineq_resid(self, X, Y):
        res = torch.sin(Y) @ self.G.T - self.h.view(1, -1)*(torch.cos(X))
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)


class nonconvexQCQPProblem(QCQPProblem):
    def __str__(self):
        return 'QCQPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * torch.sin(Y)).sum(dim=1) 
    
    def ineq_resid(self, X, Y):
        res = []    
        q = torch.matmul(self.H, Y.T).permute(2, 0, 1)
        q = (q * Y.view(Y.shape[0], 1, -1)).sum(-1)
        res = 0.5 * q + torch.matmul(torch.cos(Y), self.G.T) - self.h
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)


class nonconvexSOCPProblem(SOCPProblem):

    def __str__(self):
        return 'SOCPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * torch.sin(Y)).sum(dim=1)
    
    def ineq_resid(self, X, Y):
        res = []
        q = torch.norm(torch.matmul(self.G, torch.cos(Y).T).permute(2, 0, 1) + self.h.unsqueeze(0), dim=-1, p=2)
        p = torch.matmul(Y, self.C.T) + self.d
        res = q - p
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)
 

###################################################################
# NONSMOOTH NONCONVEX
###################################################################

class nonsmooth_nonconvexQPProblem(QPProblem):
    def __str__(self):
        return 'QPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * torch.sin(Y)).sum(dim=1) + 0.1*torch.norm(Y, dim=1)
    
    def ineq_resid(self, X, Y):
        res = torch.sin(Y) @ self.G.T - self.h.view(1, -1)*(torch.cos(X))
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)
    

class nonsmooth_nonconvexQCQPProblem(QCQPProblem):
    def __str__(self):
        return 'QCQPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    
    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * torch.sin(Y)).sum(dim=1) + 0.1*torch.norm(Y, dim=1)
    
    def ineq_resid(self, X, Y):
        res = []    
        q = torch.matmul(self.H, Y.T).permute(2, 0, 1)
        q = (q * Y.view(Y.shape[0], 1, -1)).sum(-1)
        res = 0.5 * q + torch.matmul(torch.cos(Y), self.G.T) - self.h
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)
    

class nonsmooth_nonconvexSOCPProblem(SOCPProblem):
    def __str__(self):
        return 'SOCPProblem-{}-{}-{}-{}'.format(
            str(self.ydim), str(self.nineq), str(self.neq), str(self.num)
        )
    def obj_fn(self, Y):
        return (0.5 * (Y @ self.Q) * Y + self.p * torch.sin(Y)).sum(dim=1) + 0.1*torch.norm(Y, dim=1)
    
    def ineq_resid(self, X, Y):
        res = []
        q = torch.norm(torch.matmul(self.G, torch.cos(Y).T).permute(2, 0, 1) + self.h.unsqueeze(0), dim=-1, p=2)
        p = torch.matmul(Y, self.C.T) + self.d
        res = q - p
        l = self.L - Y
        u = Y - self.U
        resids = torch.cat([res, l, u], dim=1)
        return torch.clamp(resids, 0)
    


#########################################################################
# For DC3 correction
def ineq_partial_grad(data, X, Y):
    # Extract partial variables and create a copy that requires gradients
    Y_pred = Y[:, data.partial_vars].clone().detach().requires_grad_(True)
    # Complete to get full Y values for the entire batch at once
    y = data.complete_partial(X, Y_pred)
    # Compute inequality residuals squared (penalty) for the entire batch
    ineq_penalty = data.ineq_resid(X, y) ** 2
    ineq_penalty = torch.sum(ineq_penalty, dim=-1, keepdim=True)
    # Get gradients with respect to Y_pred for the entire batch at once
    grad_pred = torch.autograd.grad(ineq_penalty.sum(), Y_pred)[0]
    # Create the full gradient tensor for all samples
    grad = torch.zeros(Y.shape[0], data.ydim, device=X.device)
    grad[:, data.partial_vars] = grad_pred
    grad[:, data.other_vars] = - (grad_pred @ data.A_partial.T) @ data.A_other_inv.T
    return grad

# Correction for DC3 
def grad_steps(data, X, Y, config):
    lr = config['DC3']['corr_lr']
    max_corr_steps = config['DC3']['max_corr_steps']
    momentum = config['DC3']['corr_momentum']    
    Y_new = Y
    old_Y_step = 0
    for _ in range(max_corr_steps):
        Y_step = ineq_partial_grad(data, X, Y_new)    
        new_Y_step = lr * Y_step + momentum * old_Y_step
        Y_new = Y_new - new_Y_step
        
        old_Y_step = new_Y_step

    return Y_new

