import os
import numpy as np
import random
import time

import torch
import torch.nn as nn
import copy

from scipy.optimize import minimize_scalar 
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.sparse.linalg import cg, LinearOperator, gmres
from scipy.stats import t







class BilevelSolverSyntheticProb:
    """
    Class used to implement bilevel stochastic algorithms for synthetic quadratic bilevel problems

    Attributes
        ul_lr (real):                         Current upper-level stepsize (default 1)
        ll_lr (real):                         Current lowel-level stepsize (default 1)
        llp_iters (int):                      Current number of lower-level steps (default 1)
        func (obj):                           Object used to define an optimization problem to solve 
        algo (str):                           Name of the algorithm to run ('bsg' or 'darts')
        seed (int, optional):                 The seed used for the experiments (default 0)
        ul_lr_init (real, optional):          Initial upper-level stepsize (default 5)
        ll_lr_init (real, optional):          Initial lower-level stepsize (default 0.1)
        max_iter (int, optional):             Maximum number of iterations (default 500)
        normalize (bool, optional):           A flag to normalize the direction used for the upper-level update (default False)
        hess (bool, optional):                A flag to use either the true Hessians (hess = True), rank-1 approximations (hess = False), or CG with FD (hess = 'CG-FD') (default False)
        fullbatch (bool, optional):           A flag to compute full-batch gradient/Hessian estimates (default False)
        llp_iters_init (real, optional):      Initial number of lower-level steps (default 1)
        inc_acc (bool, optional):             A flag to use an increasing accuracy strategy for the lower-level problem (default False)
        true_func (bool, optional):           A flag to compute the true function (default True)
        constrained (bool, optional):       A flag to consider lower-level constraints in the bilevel problem         
        opt_stepsize (bool, optional):        A flag to compute the optimal stepsize when computing the true objective function of the bilevel problem (default True)
        use_stopping_iter (bool, optional):   A flag to use the total number of iterations as a stopping criterion (default True)
        stopping_time (real, optional):       Maximum running time (in sec) used when use_stopping_iter is False (default 0.5)
        iprint (int):                         Used to print info on the optimization (default 1): 0 --> no printing; 1 --> at the end of the optimization; 2 --> at each iteration 
    """    
    
    ul_lr = 1
    ll_lr = 1
    llp_iters = 1
    
    
    def __init__(self, func, algo, seed=0, ul_lr=5, ll_lr=0.1, max_iter=2000, normalize=False,\
                 hess=False, fullbatch=False, llp_iters=1, inc_acc=False, true_func=True, constrained=False, \
                 opt_stepsize = True, use_stopping_iter=True, stopping_time=0.5, iprint = 1):

        self.func = func
        self.algo = algo
        
        self.seed = seed
        self.ul_lr_init = ul_lr
        self.ll_lr_init = ll_lr
        self.max_iter = max_iter
        self.normalize = normalize
        self.hess = hess
        self.fullbatch = fullbatch
        self.llp_iters_init = llp_iters
        self.inc_acc = inc_acc
        self.true_func = true_func
        self.constrained = constrained
        self.opt_stepsize = opt_stepsize        
        self.use_stopping_iter = use_stopping_iter
        self.stopping_time = stopping_time
        self.iprint = iprint
        
        
    def set_seed(self, seed):
        """
        Sets the seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed) 
    
    
    def generate_ll_vars(self):
        """
        Creates a vector of LL variables 
        """
        ll_vars = np.random.uniform(0, 0.1, (self.func.y_dim, 1))
        return ll_vars
    
    
    def update_llp_constr(self, ul_vars, ll_vars): 
        """
        Updates the LL variables when the LL problem is constrained
        """        
        
        if self.algo == 'sigd': # For when we are using the SIGD algorithm, 
            # Define the functions to be used in solving the projection problem    
            def fun(y): # The QP function to be minimized
                    return np.linalg.norm( init_ll_vars - ll_vars )**2
            
            # # The A matrix components of the constraints
            # A1 = self.func.A1
            # A2 = self.func.A2
            # b = self.func.rhs # The rhs vector of the constraints
            # # Define the constraints
            # cons = LinearConstraint(A2, lb = -np.inf, ub = np.squeeze(b - np.dot(A1, ul_vars)))
            
            # The A matrix component of the constraint
            A = self.func.A
            b = self.func.rhs # The rhs vector of the constraints
            # Define the constraints
            cons = LinearConstraint(A, lb = -np.inf, ub = np.squeeze(b))
            
            for i in range(self.llp_iters):
                # Initial LL variables
                init_ll_vars = ll_vars
                # Gradient of the LLP wrt the LL variables
                grad_llp_ll_vars = self.func.grad_fl_ll_vars(ul_vars, ll_vars)
                # Update the LL variables
                ll_vars = ll_vars - self.ll_lr/(i+1) * grad_llp_ll_vars
                
                # Perform a projection onto the feasible region by solving a QP
                proj_sol = minimize(fun, np.squeeze(ll_vars), method='SLSQP', constraints=cons)
                ll_vars = np.asarray([proj_sol.x]).T # The projected solution
                
        else:
            for i in range(self.llp_iters):            
                # Obtain subgradient of the penalty function wrt LL variables
                grad_llp_ll_vars = self.func.grad_ll_pen_func_ll_vars(ul_vars, ll_vars)
                
                # Update the variables
                # alpha = self.backtracking_armijo_ll(ul_vars, ll_vars) 
                # ll_vars = ll_vars - alpha * grad_llp_ll_vars 
                ll_vars = ll_vars - self.ll_lr/(i+1) * grad_llp_ll_vars 
            
        return ll_vars    

    
    
    def update_llp(self, ul_vars, ll_vars): 
        """
        Updates the LL variables by taking a gradient descent step for the LL problem
        """ 
            
        for i in range(self.llp_iters):
        # for i in range(100):            
            # Obtain gradient of the LLP wrt LL variables
            grad_llp_ll_vars = self.func.grad_fl_ll_vars(ul_vars, ll_vars)
                
            # Update the variables
            # alpha = self.backtracking_armijo_ll(ul_vars, ll_vars) 
            # ll_vars = ll_vars - alpha * grad_llp_ll_vars 
            ll_vars = ll_vars - self.ll_lr/(i+1) * grad_llp_ll_vars 
            
        return ll_vars
    
 
    def backtracking_armijo_ul(self, x, y, d, eta = 10**-4, tau = 0.5):
        initial_rate = 1
        
        alpha_ul = initial_rate
        iterLS = 0
        y_tilde = self.update_llp_true_funct(x - alpha_ul * d, y, LL_iters_true_funct=10) 
        
        sub_obj = np.linalg.norm(d)**2
        while (self.func.f_u(x - alpha_ul * d, y_tilde) > self.func.f_u(x, y) - eta * alpha_ul * sub_obj):                                    
            iterLS = iterLS + 1
            alpha_ul = alpha_ul * tau
            if alpha_ul <= 10**-8:
                alpha_ul = 0 
                y_tilde = y;
                break  
            
            y_tilde = self.update_llp_true_funct(x - alpha_ul * d, y, LL_iters_true_funct=10) 
    
        return alpha_ul, y_tilde


    def backtracking_armijo_ll(self, x, y, eta = 10**-4, tau = 0.5):
        initial_rate = 1
        
        alpha = initial_rate
        iterLS = 0
        
        grad_f_l_y = self.func.grad_fl_ll_vars(x, y) 
        
        while (self.func.f_l(x, y - alpha * grad_f_l_y) > self.func.f_l(x,y) - eta * alpha * np.dot(grad_f_l_y.T, grad_f_l_y)):                                    
            iterLS = iterLS + 1
            alpha = alpha * tau
            if alpha <= 10**-8:
                alpha = 0 
                break 
          
        return alpha


    def backtracking_armijo_ul_constr(self, x, y, d, eta = 10**-4, tau = 0.5):
        initial_rate = 1
        
        alpha_ul = initial_rate
        iterLS = 0
        y_tilde = self.update_llp_true_funct_constr(x - alpha_ul * d, y, LL_iters_true_funct=10) 
        
        sub_obj = np.linalg.norm(d)**2
        while (self.func.f_u(x - alpha_ul * d, y_tilde) > self.func.f_u(x, y) - eta * alpha_ul * sub_obj):                                    
            iterLS = iterLS + 1
            alpha_ul = alpha_ul * tau
            if alpha_ul <= 10**-8:
                alpha_ul = 0 
                y_tilde = y;
                break  
            
            y_tilde = self.update_llp_true_funct_constr(x - alpha_ul * d, y, LL_iters_true_funct=10) 
    
        return alpha_ul, y_tilde


    def backtracking_armijo_ll_constr(self, x, y, eta = 10**-4, tau = 0.5):
        initial_rate = 1
        
        alpha = initial_rate
        iterLS = 0
        
        grad_f_l_y = self.func.grad_ll_pen_func_ll_vars(x, y)
        
        while (self.func.ll_pen_func_ll_vars(x, y - alpha * grad_f_l_y) > self.func.ll_pen_func_ll_vars(x,y) - eta * alpha * np.dot(grad_f_l_y.T, grad_f_l_y)):                                    
            iterLS = iterLS + 1
            alpha = alpha * tau
            if alpha <= 10**-8:
                alpha = 0 
                break 
          
        return alpha
    

    def darts_ulp_grad(self, ul_vars, ll_vars, ll_orig_vars): 
        """
        Updates the UL variables based on the DARTS step for the UL problem
        """      
        
        # Gradient of the UL objective function wrt LL variables
        grad_ulp_ll_vars = self.func.grad_fu_ll_vars(ul_vars, ll_vars)
        grad_norm = np.linalg.norm(grad_ulp_ll_vars)
        # Gradient of the UL objective function wrt UL variables
        grad_ulp_ul_vars = self.func.grad_fu_ul_vars(ul_vars, ll_vars)
        
        ep = 0.01 / grad_norm
        # Define y+ and y-
        ll_plus = ll_orig_vars + ep * grad_ulp_ll_vars
        ll_minus = ll_orig_vars - ep * grad_ulp_ll_vars
        
        # Gradient of the LL objective function wrt UL variables at the point (ul_vars, ll_plus)
        grad_plus = self.func.grad_fl_ul_vars(ul_vars, ll_plus)
        
        # Gradient of the LL objective function wrt UL variables at the point (ul_vars, ll_minus)
        grad_minus = self.func.grad_fl_ul_vars(ul_vars, ll_minus)
    
        grad_approx = grad_ulp_ul_vars - self.ll_lr * ((grad_plus - grad_minus) / (2 * ep))
        
        # Normalize the direction
        if self.normalize:
            grad_approx = grad_approx / np.linalg.norm(grad_approx, np.inf)
        
        # Update the UL variables
        ul_vars = ul_vars - self.ul_lr * grad_approx
        return ul_vars

    
    def calc_dx_ulp(self, ul_vars, ll_vars): 
        """
        Updates the UL variables based on the BSG step for the UL problem 
        """       

        # Gradient of the UL objective function wrt LL variables
        grad_ulp_ll_vars = self.func.grad_fu_ll_vars(ul_vars, ll_vars)
        # Gradient of the UL problem wrt UL variables
        grad_ulp_ul_vars = self.func.grad_fu_ul_vars(ul_vars, ll_vars)
        
        # Gradient of the LL objective function wrt LL variables
        grad_llp_ll_vars = self.func.grad_fl_ll_vars(ul_vars, ll_vars)
        # Grad of the LL problem wrt UL vars
        grad_LLP_ul_vars = self.func.grad_fl_ul_vars(ul_vars, ll_vars)
        
        # Compute the bsg direction
        if self.hess == True:
            # BSG-H
            # Hessian of the LL problem wrt UL and LL variables
            hess_LLP_ul_vars_ll_vars = self.func.hess_fl_ul_vars_ll_vars(ul_vars, ll_vars)
            # Hessian of the LL problem wrt LL variables
            hess_LLP_ll_vars_ll_vars = self.func.hess_fl_ll_vars_ll_vars(ul_vars, ll_vars)
            
            lambda_adj, exit_code = cg(hess_LLP_ll_vars_ll_vars,grad_ulp_ll_vars, x0=None, tol=1e-4, maxiter=3)
            lambda_adj = np.reshape(lambda_adj, (-1,1))
            
            # lambda_adj = np.matmul(np.linalg.inv(hess_LLP_ll_vars_ll_vars),grad_ulp_ll_vars)
            
            bsg = grad_ulp_ul_vars - np.matmul(hess_LLP_ul_vars_ll_vars,lambda_adj)
            
        elif self.hess == False:
            # BSG-1
            bsg = grad_ulp_ul_vars - ((np.dot(grad_llp_ll_vars.T, grad_ulp_ll_vars)) / np.linalg.norm(grad_llp_ll_vars)**2) * grad_LLP_ul_vars

        elif self.hess == 'CG-FD':            
            # # BSG-N-FD
            
            def mv(v):
                ## Finite difference approximation
                v_norm = np.linalg.norm(v) 
                ep = 1e-1 #0.1/np.maximum(1,v_norm) #1e-1
                # Define y+ and y-
                ll_plus = ll_vars + ep * v.reshape(-1,1)
                ll_minus = ll_vars - ep * v.reshape(-1,1)                
                # Gradient of the LL objective function wrt LL variables at the point (ul_vars, ll_plus)
                grad_plus = self.func.grad_fl_ll_vars(ul_vars, ll_plus)                
                # Gradient of the LL objective function wrt LL variables at the point (ul_vars, ll_minus)
                grad_minus = self.func.grad_fl_ll_vars(ul_vars, ll_minus) 
                return (grad_plus - grad_minus) / (2 * ep)  
            
            self.hess_LLP_ll_vars_ll_vars_lin = LinearOperator((self.func.y_dim,self.func.y_dim), matvec=mv)
            
            lambda_adj, exit_code = cg(self.hess_LLP_ll_vars_ll_vars_lin,grad_ulp_ll_vars,x0=None, tol=1e-4, maxiter=3) #maxiter=100
            lambda_adj = np.reshape(lambda_adj, (-1,1))
            
            ## Finite difference approximation
            grad_norm = np.linalg.norm(lambda_adj)            
            ep = 1e-1 #0.01 / grad_norm
            # Define y+ and y-
            ll_plus = ll_vars + ep * lambda_adj
            ll_minus = ll_vars - ep * lambda_adj
            
            # Gradient of the LL objective function wrt UL variables at the point (ul_vars, ll_plus)
            grad_plus = self.func.grad_fl_ul_vars(ul_vars, ll_plus)
            
            # Gradient of the LL objective function wrt UL variables at the point (ul_vars, ll_minus)
            grad_minus = self.func.grad_fl_ul_vars(ul_vars, ll_minus)
        
            bsg = grad_ulp_ul_vars - ((grad_plus - grad_minus) / (2 * ep))

        else:
            print('There is something wrong with self.hess')
        
        # Normalize the direction
        if self.normalize:
            bsg = bsg / np.linalg.norm(bsg, np.inf)
        
        # Update the UL variables
        # self.ul_lr, _ = self.backtracking_armijo_ul(ul_vars, ll_vars, bsg)
        ul_vars = ul_vars - self.ul_lr * bsg
        
        return ul_vars


    def calc_dx_ulp_constr(self, ul_vars, ll_vars, jacob_x, jacob_y, inconstr_vec, lam0, lagr_mul):
        """
        Updates the UL variables based on the BSG step for the UL problem in the LL constrained case
        """       
        
        # Gradient of the UL objective function 
        ulp_grad_x = self.func.grad_fu_ul_vars(ul_vars, ll_vars)
        ulp_grad_y = self.func.grad_fu_ll_vars(ul_vars, ll_vars)
        
        rhs = np.concatenate((ulp_grad_y,np.zeros((lagr_mul.shape[0],1))), axis=0)
        if self.it == 0:
            lam0 = None
            # lam0 = np.random.rand(self.func.y_dim + lagr_mul.shape[0],1)
               
        # Compute the bsg direction
        if self.hess == True:
            # BSG-H
            # Hessian of the LL problem wrt UL and LL variables
            # hess_LLP_ul_vars_ll_vars = self.func.hess_fl_ul_vars_ll_vars(ul_vars, ll_vars)
            # Jacobian of the lower-level KKT system wrt lower-level variables
            jacob_ll_kkt_ul_vars = self.func.jacob_ll_kkt_ul_vars(ul_vars, ll_vars, lagr_mul)

            def mv(v):
                # jacob_ll_kkt_ll_vars = self.func.jacob_ll_kkt_ll_vars(ul_vars, ll_vars, lagr_mul)
                # return np.matmul(jacob_ll_kkt_ll_vars,v)
                
                aa = v[:self.func.y_dim].reshape(-1,1)
                bb = v[self.func.y_dim:].reshape(-1,1)            
                
                aux_1 = np.matmul(self.func.hess_ll_lagr_func_ll_vars_ll_vars(ul_vars, ll_vars, lagr_mul),aa) + np.matmul(lagr_mul.T*jacob_y,bb)               
                aux_2 = np.matmul(jacob_y.T,aa) + np.multiply(inconstr_vec,bb) 
                
                return  np.concatenate((aux_1,aux_2),axis=0)    
                 
            self.ll_kkt_lin = LinearOperator((self.func.y_dim+self.func.num_constr, self.func.y_dim+self.func.num_constr), matvec=mv)                 
            lam, exit_code = gmres(self.ll_kkt_lin, rhs, x0=lam0, tol=1e-4, maxiter=3)
            lam = lam.reshape(-1,1)
            print('exit_code',exit_code)
            # time.sleep(1)
            
            ###########
            # jacob_ll_kkt_ll_vars = self.func.jacob_ll_kkt_ll_vars(ul_vars, ll_vars, lagr_mul)
            # lam = np.matmul(np.linalg.inv(jacob_ll_kkt_ll_vars),rhs)
            ###########
            
            bsg = ulp_grad_x - np.matmul(jacob_ll_kkt_ul_vars,lam)
            
        elif self.hess == False:
            # Gradient of the Lagrangian wrt y       
            llp_lagr_grad_x = self.func.grad_ll_lagr_func_ul_vars(ul_vars, ll_vars, lagr_mul)
            llp_lagr_grad_y = self.func.grad_ll_lagr_func_ll_vars(ul_vars, ll_vars, lagr_mul) 
            
            # BSG-1
            def mv(v):
               aa = v[0:self.func.y_dim].reshape(-1,1)
               bb = v[self.func.y_dim:].reshape(-1,1)         
               aux_1 = np.array(np.matmul(llp_lagr_grad_y, np.matmul(llp_lagr_grad_y.T,aa)) + np.matmul(lagr_mul.T*jacob_y,bb)) 
               aux_2 = np.array(np.matmul(jacob_y.T,aa) + np.multiply(inconstr_vec,bb))              
               return np.concatenate((aux_1,aux_2),axis=0) 
                   
            self.ll_kkt_lin = LinearOperator((self.func.y_dim+self.func.num_constr, self.func.y_dim+self.func.num_constr), matvec=mv)             
            lam, _ = gmres(self.ll_kkt_lin, rhs, x0=lam0, tol=1e-4, maxiter=3) #maxiter=50
            lam = lam.reshape(-1,1)
            bsg = ulp_grad_x - np.matmul(llp_lagr_grad_x,np.matmul(llp_lagr_grad_y.T,lam[0:llp_lagr_grad_y.shape[0]])) - np.matmul(lagr_mul.T*jacob_x,lam[llp_lagr_grad_y.shape[0]:])
        
        elif self.hess == 'CG-FD':            
            # # BSG-N-FD
            
            def mv(v):
                aa = v[:self.func.y_dim].reshape(-1,1)
                bb = v[self.func.y_dim:].reshape(-1,1)

                ## Finite difference approximation
                v_norm = np.linalg.norm(v) 
                ep = 1e-1 #/ v_norm
                # Define y+ and y-
                ll_plus = ll_vars + ep * aa
                ll_minus = ll_vars - ep * aa                
                # Gradient of the LL objective function wrt LL variables at the point (ul_vars, ll_plus)
                grad_plus = self.func.grad_ll_lagr_func_ll_vars(ul_vars, ll_plus, lagr_mul)                
                # Gradient of the LL objective function wrt LL variables at the point (ul_vars, ll_minus)
                grad_minus = self.func.grad_ll_lagr_func_ll_vars(ul_vars, ll_minus, lagr_mul) 
                
                aux_1 = (grad_plus - grad_minus)/(2*ep) + np.matmul(lagr_mul.T*jacob_y,bb)               
                aux_2 = np.matmul(jacob_y.T,aa) + np.multiply(inconstr_vec,bb)                
                return  np.concatenate((aux_1,aux_2),axis=0)       
            
            self.ll_kkt_lin = LinearOperator((self.func.y_dim+self.func.num_constr, self.func.y_dim+self.func.num_constr), matvec=mv) 
            
            lam, _ = gmres(self.ll_kkt_lin, rhs, x0=lam0, tol=1e-4, maxiter=3)
            lam = lam.reshape(-1,1)
            
            ## Finite difference approximation
            grad_norm = np.linalg.norm(lam[:self.func.y_dim])      
            ep = 1e-1 #/ grad_norm
            # Define y+ and y-
            ll_plus = ll_vars + ep * lam[:self.func.y_dim]
            ll_minus = ll_vars - ep * lam[:self.func.y_dim]      
            # Gradient of the LL objective function wrt UL variables at the point (ul_vars, ll_plus)
            grad_plus = self.func.grad_ll_lagr_func_ul_vars(ul_vars, ll_plus, lagr_mul)
            # Gradient of the LL objective function wrt UL variables at the point (ul_vars, ll_minus)
            grad_minus = self.func.grad_ll_lagr_func_ul_vars(ul_vars, ll_minus, lagr_mul)
            
            aux_1 = (grad_plus - grad_minus) / (2 * ep)
            aux_2 = np.matmul(lagr_mul.T*jacob_x,lam[self.func.y_dim:])
            
            bsg = ulp_grad_x - (aux_1 + aux_2)
             
        else:
            print('There is something wrong with self.hess')
        
        # Normalize the direction
        if self.normalize:
            bsg = bsg / np.linalg.norm(bsg, np.inf)
        
        # Update the UL variables
        # self.ul_lr, _ = self.backtracking_armijo_ul_constr(ul_vars, ll_vars, bsg)
        #print(bsg)
        ul_vars = ul_vars - self.ul_lr * bsg
        
        return ul_vars, lam
    

    def stocbio_ulp(self, ul_vars, ll_vars): 
        """
        StocBiO algorithm. See https://github.com/JunjieYang97/stocBiO/blob/master/Hyperparameter-optimization/experimental/stocBiO.py
        """
        # Gradient of the UL objective function wrt LL variables
        grad_ulp_ll_vars = self.func.grad_fu_ll_vars(ul_vars, ll_vars)
        # Gradient of the UL problem wrt UL variables
        grad_ulp_ul_vars = self.func.grad_fu_ul_vars(ul_vars, ll_vars)
        
        v_0 = torch.from_numpy(grad_ulp_ll_vars).detach()   
        
        eta = 0.05#0.01#0.05
        hessian_q = 2
        
        ll_vars = torch.tensor(ll_vars, dtype=torch.float64)
        ll_vars.requires_grad_()      
        ul_vars = torch.tensor(ul_vars, dtype=torch.float64)
        
        # Gradient of the LL objective function wrt LL variables
        grad_llp_ll_vars = self.func.grad_fl_ll_vars_torch(ul_vars, ll_vars)
        
        G_gradient = torch.reshape(ll_vars, [-1]) - eta * torch.reshape(grad_llp_ll_vars, [-1]) 
        
        # Hessian
        z_list = []
        
        for _ in range(hessian_q):
            Jacobian = torch.matmul(G_gradient.double(), v_0.double())
            v_new = torch.autograd.grad(Jacobian, ll_vars, create_graph=True)[0]
            v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
            z_list.append(v_0) 
        v_Q = eta*v_0+torch.sum(torch.stack(z_list), dim=0)
    
        ul_vars.requires_grad_()    
        ll_vars = ll_vars.detach()

        # Gradient of the LL objective function wrt LL variables
        grad_llp_ll_vars = self.func.grad_fl_ll_vars_torch(ul_vars, ll_vars)
        
        # Gyx_gradient
        Gy_gradient = torch.reshape(grad_llp_ll_vars, [-1])
        
        Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient.double(), v_Q.detach().double()), ul_vars, create_graph=True)[0]
        
        bsg = grad_ulp_ul_vars - Gyx_gradient.detach().numpy()
        
        # Update the UL variables
        ul_vars = ul_vars.detach().numpy() - self.ul_lr * bsg
        
        return ul_vars 
    
    
    def sigd_ulp_constr(self, ul_vars, ll_vars): 
        """
        (S)SIGD algorithm ULP update for bilevel problems with linear LL constraints. See https://openreview.net/pdf?id=VwxjC1tm5o
        """      
        # Hessian of the LL problem wrt UL and LL variables
        hess_LLP_ul_vars_ll_vars = self.func.hess_fl_ul_vars_ll_vars(ul_vars, ll_vars)
        # Hessian of the LL problem wrt LL variables
        hess_LLP_ll_vars_ll_vars = self.func.hess_fl_ll_vars_ll_vars(ul_vars, ll_vars)
        # The A matrix that corresponds with the y-component of the constraints
        A_ = self.func.A
        # Compute the inverse of hess_LLP_ll_vars_ll_vars
        inv_hess_LLP_ll_vars_ll_vars = np.linalg.inv(hess_LLP_ll_vars_ll_vars)
        # Mat-Mat product A_*inv_hess_LLP_ll_vars_ll_vars
        A_inv_prod = np.matmul(A_, inv_hess_LLP_ll_vars_ll_vars)
        # Compute the jacobian of the lagrange multipliers \nabla\lambda(x)
        jac_lambda = - np.matmul( np.linalg.inv( np.matmul(A_inv_prod, np.transpose(A_))), np.matmul(A_inv_prod, hess_LLP_ul_vars_ll_vars) )
        
        # Compute the jacobian of the LL variables \nabla y(x)
        jac_y = np.matmul(inv_hess_LLP_ll_vars_ll_vars, (-hess_LLP_ul_vars_ll_vars - np.matmul(np.transpose(A_), jac_lambda) ) )
        
        
        # Gradient of the UL objective function wrt LL variables
        grad_ulp_ll_vars = self.func.grad_fu_ll_vars(ul_vars, ll_vars)
        # Gradient of the UL problem wrt UL variables
        grad_ulp_ul_vars = self.func.grad_fu_ul_vars(ul_vars, ll_vars)
        # Compute the gradient of the ULP
        bsg = grad_ulp_ul_vars + np.dot( np.transpose(jac_y), grad_ulp_ll_vars )
        
        # Update the UL variables
        ul_vars = ul_vars - self.ul_lr * bsg
        
        return ul_vars

    
    def update_llp_true_funct(self, ul_vars, ll_vars, LL_iters_true_funct): 
        """
        Updates the LL problem when considering the true objective function of the bilevel problem
        """        
        for i in range(LL_iters_true_funct):
            # Obtain gradient of the LL objective function wrt LL variables
            grad_llp_ll_vars = self.func.grad_fl_ll_vars(ul_vars, ll_vars) 
            
            # Update the variables
            if self.opt_stepsize: 
                def obj_funct_2(stepsize):
                    aux_var = ll_vars - stepsize * grad_llp_ll_vars
                    return self.func.f_l(ul_vars, aux_var)
                res = minimize_scalar(obj_funct_2, bounds=(0, 1), method='bounded')
                ll_lr = res.x
            else:
                ll_lr = self.ll_lr_init
                # ll_lr = 1
            ll_vars = ll_vars - ll_lr * grad_llp_ll_vars
            # End if gradient is small enough
            if np.linalg.norm(grad_llp_ll_vars) <= 1e-4:
                return ll_vars
        
        return ll_vars


    def update_llp_true_funct_constr(self, ul_vars, ll_vars, LL_iters_true_funct): 
        """
        Updates the LL problem when considering the true objective function of the bilevel problem
        """        
        for i in range(LL_iters_true_funct):
            # Obtain gradient of the LL objective function wrt LL variables
            grad_llp_ll_vars = self.func.grad_ll_pen_func_ll_vars(ul_vars, ll_vars)
            
            # Update the variables
            if self.opt_stepsize: 
                def obj_funct_2(stepsize):
                    aux_var = ll_vars - stepsize * grad_llp_ll_vars
                    return self.func.ll_pen_func_ll_vars(ul_vars, aux_var)
                res = minimize_scalar(obj_funct_2, bounds=(0, 1), method='bounded')
                ll_lr = res.x
            else:
                ll_lr = self.ll_lr_init
                # ll_lr = 1
            ll_vars = ll_vars - ll_lr * grad_llp_ll_vars
            # End if gradient is small enough
            if np.linalg.norm(grad_llp_ll_vars) <= 1e-4:
                return ll_vars
        
        return ll_vars
    
    
    def true_function_value(self, ul_vars, ll_vars, true_func_list): 
        """
        Computes the true function value
        """ 
        true_func_list.append(self.func.f(ul_vars)) 
        return true_func_list


    def true_function_value_constr(self, ul_vars, ll_vars, lagr_mul, true_func_list): 
        """
        Computes the true function value
        """ 
        true_func_list.append(self.func.f_constr(ul_vars, ll_vars, lagr_mul)) 
        return true_func_list
    
    
    def main_algorithm(self): 
        """
        Main body of a bilevel stochastic algorithm
        """
        cur_time = time.time()    
        
        # Initialize lists
        func_val_list = []
        true_func_list = []
        time_list = [] 
        
        # Initialize the variables
        ul_vars = np.random.uniform(0, 10, (self.func.x_dim, 1)) #np.random.uniform(0, 10, (self.func.x_dim, 1))
        ll_vars = np.random.uniform(0, 10, (self.func.y_dim, 1)) #np.random.uniform(0, 10, (self.func.x_dim, 1))
        
        self.ul_lr = self.ul_lr_init
        self.ll_lr = self.ll_lr_init
        self.llp_iters = self.llp_iters_init
        
        j = 1            
        for it in range(self.max_iter):
            self.it = it
            
            # Check if we stop the algorithm based on time
            if not(self.use_stopping_iter) and (time.time() - cur_time >= self.stopping_time): 
                break
            
            pre_obj_val = self.func.f_u(ul_vars, ll_vars) 
            
            # Update the LL variables with a single step
            ll_orig_vars = ll_vars
            if self.constrained:
                ll_vars = self.update_llp_constr(ul_vars, ll_vars)
            else:
                ll_vars = self.update_llp(ul_vars, ll_vars)
            # print('*****111*****ll_vars:', ll_vars)
            # time.sleep(1)
            # LL constrained case
            if self.constrained:
              jacob_x, jacob_y = self.func.inequality_constraint_jacob(ul_vars, ll_vars)
              inconstr_vec = self.func.inequality_constraint(ul_vars, ll_vars)

              # Gradient of LL objective function wrt y without penalty
              grad_llp_ll_vars = self.func.grad_fl_ll_vars(ul_vars, ll_vars) 
	
              if (it == 0):
              	lagr_mul = None

              def mv(v):
                  out = np.matmul(jacob_y.T, np.matmul(jacob_y, v.reshape(-1,1))) + np.multiply(inconstr_vec**2, v.reshape(-1,1)) 
                  return out  
                
              G = LinearOperator((self.func.num_constr,self.func.num_constr), matvec=mv)
                
              lagr_mul, exit_code = cg(G,-np.matmul(jacob_y.T,grad_llp_ll_vars),x0=lagr_mul, tol=1e-4, maxiter=3) #maxiter=100
              lagr_mul = np.reshape(lagr_mul, (-1,1))
            # print('*****222*****lagr_mul:', lagr_mul)
            # time.sleep(1)            
            # Estimate the gradient of the UL objective function
            if self.algo == 'darts':
                ul_vars = self.darts_ulp_grad(ul_vars, ll_vars, ll_orig_vars) 
            elif self.algo == 'bsg':
                if self.constrained:
                    if self.it == 0:
                       lam = []
                    ul_vars, lam = self.calc_dx_ulp_constr(ul_vars, ll_vars, jacob_x, jacob_y, inconstr_vec, lam, lagr_mul) 
                else:
                    ul_vars = self.calc_dx_ulp(ul_vars, ll_vars) 
            elif self.algo == 'stocbio':
                ul_vars = self.stocbio_ulp(ul_vars, ll_vars) 
            elif self.algo == 'sigd':
                ul_vars = self.sigd_ulp_constr(ul_vars, ll_vars)
            # print('*****333*****ul_vars:', ul_vars)
            # time.sleep(1)              
            j += 1  
            func_val_list.append(self.func.f_u(ul_vars, ll_vars))
            
            end_time =  time.time() - cur_time
            time_list.append(end_time)

            # Compute the true function (only for plotting purposes)                
            if self.true_func == True: 
                if self.constrained:
                    true_func_list = self.true_function_value_constr(ul_vars, ll_vars, lagr_mul, true_func_list)
                else:                
                    true_func_list = self.true_function_value(ul_vars, ll_vars, true_func_list) 

            # Increasing accuracy strategy            
            if self.inc_acc == True:
             	if self.llp_iters >= 30:
              		self.llp_iters = 30
             	else:
                    post_obj_val = self.func.f_u(ul_vars, ll_vars) 
                    obj_val_diff = abs(post_obj_val - pre_obj_val)
                    if obj_val_diff/abs(pre_obj_val) <= 3e-3:  #1e-4 
                    # if obj_val_diff <= 1e-1:  #1e-4              
                        self.llp_iters += 1
                
            # Update the learning rates
            self.ul_lr = self.ul_lr_init #/j  
            self.ll_lr = self.ll_lr_init 

            if self.iprint >= 2:
                if self.true_func:
                    print("Algorithm: ",self.algo,' iter: ',it,' f_u: ',func_val_list[len(func_val_list)-1],' f: ',true_func_list[len(true_func_list)-1],' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)    
                else:
                    print("Algorithm: ",self.algo,' iter: ',it,' f_u: ',func_val_list[len(func_val_list)-1],' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)   
            
        if self.iprint >= 1:
            if self.true_func:
                print("Algorithm: ",self.algo,' f_u: ',func_val_list[len(func_val_list)-1],' f: ',true_func_list[len(true_func_list)-1],' time: ',time.time() - cur_time,' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)    
            else:
                print("Algorithm: ",self.algo,' f_u: ',func_val_list[len(func_val_list)-1],' time: ',time.time() - cur_time,' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)    
    
        return [func_val_list, true_func_list, time_list]
    
    
    def piecewise_func(self, x, boundaries, func_val):
        """
        Computes the value of a piecewise constant function defined by boundaries and func_val at x
        """
        for i in range(len(boundaries)):
            if x <= boundaries[i]:
              return func_val[i]
        return func_val[len(boundaries)-1]
    
    
    def main_algorithm_avg_ci(self, num_rep=1):
        """
        Returns arrays with averages and 95% confidence interval half-widths for function values or true function values at each iteration obtained over multiple runs
        """
        self.set_seed(self.seed) 
        # Solve the problem for the first time
        sol = self.main_algorithm()
        values = sol[0]
        true_func_values = sol[1]
        times = sol[2]
        values_rep = np.zeros((len(values),num_rep))
        values_rep[:,0] = np.asarray(values) 
        if self.true_func:
            true_func_values_rep = np.zeros((len(true_func_values),num_rep))
            true_func_values_rep[:,0] = np.asarray(true_func_values)
        # Solve the problem num_rep-1 times
        for i in range(num_rep-1):
          self.set_seed(self.seed+1+i) 
          sol = self.main_algorithm()
          if self.use_stopping_iter:
              values_rep[:,i+1] = np.asarray(sol[0])
              if self.true_func:
                  true_func_values_rep[:,i+1] = np.asarray(sol[1])
          else:
              values_rep[:,i+1] = list(map(lambda x: self.piecewise_func(x,sol[2],sol[0]),times))
              if self.true_func:
                  true_func_values_rep[:,i+1] = list(map(lambda x: self.piecewise_func(x,sol[2],sol[1]),times))
        values_avg = np.mean(values_rep, axis=1)
        values_ci = t.ppf(0.975,num_rep-1)*(np.sqrt(np.var(values_rep, ddof=1, axis=1))/np.sqrt(num_rep))
        if self.true_func:
            true_func_values_avg = np.mean(true_func_values_rep, axis=1)
            true_func_values_ci = t.ppf(0.975,num_rep-1)*(np.sqrt(np.var(true_func_values_rep, ddof=1, axis=1))/np.sqrt(num_rep))
        else:
            true_func_values_avg = []
            true_func_values_ci = []
            
        return values_avg, values_ci, true_func_values_avg, true_func_values_ci, times












class BilevelSolverCL:
    """
    Class used to implement bilevel stochastic algorithms for continual learning

    Attributes
        ul_lr (real):                       Current upper-level stepsize (default 1)
        ll_lr (real):                       Current lowel-level stepsize (default 1)
        llp_iters (int):                    Current number of lower-level steps (default 1)
        func (obj):                         Object used to define the bilevel problem to solve 
        algo (str):                         Name of the algorithm to run ('bsg' or 'darts')
        seed (int, optional):               The seed used for the experiments (default 0)
        ul_lr_init (real, optional):        Initial upper-level stepsize (default 5)
        ll_lr_init (real, optional):        Initial lower-level stepsize (default 0.1)
        max_epochs (int, optional):         Maximum number of epochs (default 1)
        normalize (bool, optional):         A flag used to normalize the direction used in the update of the upper-level variables (default False)
        llp_iters_init (real, optional):    Initial number of lower-level steps (default 1)
        inc_acc (bool, optional):           A flag to use an increasing accuracy strategy for the lower-level problem (default False)
        true_func (bool, optional):         A flag to compute the true objective function of the bilevel problem (default False)
        constrained (bool, optional):       A flag to consider lower-level constraints in the bilevel problem 
        pen_param (bool, optional):         Penalty parameter used in the lower-level penalty function when constrained is True (default 10**(-1))
        use_stopping_iter (bool, optional): A flag to use the maximum number of epochs as a stopping criterion; if False, the running time is used as a stopping criterion (default True)
        stopping_times (real, optional):    List of times used when use_stopping_iter is False to determine when a new task must be added to the problem (default [20, 40, 60, 80, 100])
        iprint (int):                       Used to print info on the optimization (default 1): 0 --> no printing; 1 --> at the end of every task; 2 --> at each iteration 
        training_loaders:                   List of 5 increasing training datasets (01, 0123, 012345, etc.)
        testing_loaders:                    List of 5 increasing testing datasets (01, 0123, 012345, etc.) 
        training_task_loaders:              List of 5 training datasets, each associated with a task
    """    
    
    ul_lr = 1
    ll_lr = 1
    llp_iters = 1
    
    
    def __init__(self, func, algo, seed=0, ul_lr=5, ll_lr=0.1, max_epochs=1, normalize=False,\
                 hess=False, llp_iters=1, inc_acc=False, true_func=False, constrained=False, \
                 pen_param = 10**(-1), use_stopping_iter=True, stopping_times=[20, 40, 60, 80, 100],\
                 iprint = 1):
        self.func = func
        self.algo = algo

        self.seed = seed
        self.ul_lr_init = ul_lr
        self.ll_lr_init = ll_lr
        self.max_epochs = max_epochs
        self.normalize = normalize
        self.hess = hess
        self.llp_iters_init = llp_iters
        self.inc_acc = inc_acc
        self.true_func = true_func
        self.constrained = constrained
        self.pen_param = pen_param        
        self.use_stopping_iter = use_stopping_iter
        self.stopping_times = stopping_times
        self.iprint = iprint
        
        self.set_seed(self.seed)
        self.model = self.func.generate_model() 
        self.x_dim = sum(p.numel() for idx_layer, p in enumerate(self.model.parameters()) if (0 <= idx_layer <= 3))
        self.y_dim = sum(p.numel() for idx_layer, p in enumerate(self.model.parameters()) if (4 <= idx_layer <= 5))
        self.training_loaders = self.func.training_loaders
        self.testing_loaders = self.func.testing_loaders
        self.training_task_loaders = self.func.training_task_loaders


    def set_seed(self, seed):
        """
        Sets the seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed) 


    def minibatch(self, loader, stream):
        """
        Returns a mini-batch of data (features and labels) and the updated stream of mini-batches
        """
        if len(stream) == 0:
            stream = self.create_stream(loader)
        minibatch_feat, minibatch_label = self.rand_samp_no_replace(stream)
        return minibatch_feat, minibatch_label, stream


    class flatten(nn.Module):
        """ 
        Defines a flattened layer of the DNN 
        """
        def forward(self, x):
            return x.view(x.shape[0], -1)
    
    
    def yield_2d_tensor(self, model, grad, layer, used):
        """
        Repacks a 2-d tensor for a layer in the network
        """
        tensor = []
        for i in range(model[layer].weight.shape[0]):
            new_used = used + model[layer].weight.shape[1]
            sub_list = grad[used : new_used]
            used = new_used
            tensor.append(sub_list)
        tensor = torch.from_numpy(np.asarray(tensor)).requires_grad_(True)
        return tensor, used
    
    
    def yield_4d_tensor(self, model, grad, layer, used):
        """
        Repacks a 4-d tensor for a layer in the network
        """        
        tensor = []
        for i in range(model[layer].weight.shape[0]):
            sub_tensor = []
            for j in range(model[layer].weight.shape[1]):
                new_used = used + model[layer].weight.shape[2] * model[layer].weight.shape[3]
                sub_list = grad[used : new_used]
                used = new_used
                sub_tensor.append(sub_list.reshape(model[layer].weight.shape[2], model[layer].weight.shape[3]))
            tensor.append(sub_tensor)
        tensor = torch.from_numpy(np.asarray(tensor)).requires_grad_(True)
        return tensor, used
    
    
    def repack_weights(self, model, grad):
        """
        Takes a single vector of weights and restructures it into a tensor of correct dimensions for the network
        """
        repacked_list = []
        used = 0
        i = 0
        for param in model.parameters():
            try:
                model[i].weight
                if len(model[i].weight.shape) == 4:
                    tensor, used = self.yield_4d_tensor(model, grad, i, used)
                    repacked_list.append(tensor.requires_grad_(False).float())
                elif len(model[i].weight.shape) == 2:
                    tensor, used = self.yield_2d_tensor(model, grad, i, used)
                    repacked_list.append(tensor.requires_grad_(False).float())
                elif len(model[i].weight.shape) == 1:
                    new_used = used + param.shape[0]
                    tensor = torch.from_numpy(np.asarray(grad[used : new_used]))
                    repacked_list.append(tensor.requires_grad_(False).float())
                    used = new_used
            except:
                if len(param.shape) == 2:
                    tensor = []
                    for i in range(param.shape[0]):
                        new_used = used + param.shape[1]
                        sub_list = grad[used : new_used]
                        used = new_used
                        tensor.append(sub_list)
                    tensor = torch.from_numpy(np.asarray(tensor)).requires_grad_(True)
                    repacked_list.append(tensor.requires_grad_(False).float())
                elif len(param.shape) == 1:
                    new_used = used + param.shape[0]
                    tensor = torch.from_numpy(np.asarray(grad[used : new_used]))
                    repacked_list.append(tensor.requires_grad_(False).float())
                    used = new_used
            i += 1
        # Returns a list of tensors
        return repacked_list 
    
    
    def unpack_weights(self, model, grad):
        """
        Unpacks a gradient wrt the UL variables, thus yielding a very large single vector of weights
        """
        unpacked_list = []
        i=0
        for i in range(len(grad)):
            unpacked_list.append(grad[i].flatten().detach().cpu().numpy())
        weights_list = np.hstack(np.array(unpacked_list))
        # Returns a numpy array
        return weights_list 
    
    
    def create_stream(self, loader):
        """
        Creates a stream of mini-batches of data
        """
        stream = random.sample(loader,len(loader)) 
        random.shuffle(stream) 
        return stream
    
    
    def rand_samp_no_replace(self, stream):
        """
        Randomly samples (without replacement) from a stream of mini-batches data
        """
        aux = [item for list_aux in stream[-1:] for item in list_aux]
        C, u = aux[0], aux[1]
        del stream[-1:]
        return [C, u]
    
    
    def repack_y_weights(self, y_weights):
        """
        Repacks the delta weight vector into a tensor of correct dimensions
        """
        repacked_list = []
        used = 0
        # value = 64*14*14 #MNIST
        value = 64*16*16 #This needs to match the number of nodes in the last hidden layer of the DNN
        for i in range(int(len(y_weights)/value)):
            new_used = used + value
            sub_list = y_weights[used : new_used]
            used = new_used
            repacked_list.append(sub_list)
        tensor = torch.stack(repacked_list)
        return tensor
 
    
    def update_model_wrt_x(self, model, ulp_grad):
        """
        Updates the hidden layer weights and biases
        """
        g=0
        for param in model.parameters():
            if g == 4:
                break
            param.data = param.data - self.ul_lr * ulp_grad[g].to(self.func.device) 
            g+=1
        return model.to(self.func.device)   
 
    
    def update_model_wrt_y(self, model, y_weights, y_biases):
        """
        Updates the output layer weights and biases
        """
        f = 0
        for param in model.parameters():
            if f == 4:
                param.data = self.repack_y_weights(y_weights).float()
            elif f == 5:
                param.data = y_biases.float()
            f+=1
        return model.to(self.func.device)
    
    
    def generate_y(self, model, randomize):
        """
        Creates a vector of LL variables 
        """
        # Initial model parameters
        params = [param.data for param in model.parameters()] 
        y_weights = params[4].flatten()
        y_biases = params[5]
        if randomize:
            y_init_weights = np.random.uniform(-0.02, 0.02, y_weights.shape[0])
            y_init_biases = np.zeros(y_biases.shape[0]) 
        else:
            y_init_weights = np.zeros(y_weights.shape[0])
            y_init_biases = np.zeros(y_biases.shape[0])
        # Update the model before returning
        model = self.update_model_wrt_y(model, torch.from_numpy(y_init_weights), torch.from_numpy(y_init_biases))
        return [y_init_weights, y_init_biases], model.to(self.func.device)
    
    
    def inequality_constraint_task(self, id_task, train_task_loader, model, oldmodel, stream_task_training_list):
        """
        Computes the value of the inequality constraint associated with a specific task
        """
        if len(stream_task_training_list[id_task]) == 0:
          stream_task_training_list[id_task] = self.create_stream(train_task_loader[id_task])
        task_train_X, task_train_y = self.rand_samp_no_replace(stream_task_training_list[id_task])
        C, u = task_train_X.to(self.func.device), task_train_y.to(self.func.device)
        u = torch.flatten(u)
        return self.func.loss_func(oldmodel(C), u) - self.func.loss_func(model(C), u)
    
    
    def inequality_constraint(self, num_constr, train_task_loader, model, oldmodel, stream_task_training_list):
        """
        Computes the array of the inequality constraints associated with each task
        """
        vec = np.zeros((num_constr,1))
        for j in range(num_constr):
          inconstr_value = self.inequality_constraint_task(j, train_task_loader, model, oldmodel, stream_task_training_list)
          vec[j][0] = inconstr_value
        return vec
    
    
    def inequality_constraint_task_grad(self, id_task, train_task_loader, model, oldmodel, stream_task_training_list):
        """
        Computes the gradient of the inequality constraint associated with a specific task
        """
        output = self.inequality_constraint_task(id_task, train_task_loader, model, oldmodel, stream_task_training_list)
        # print('constraint value',output)
        model.zero_grad()
        #Calculate the gradient
        output.backward() 
        grad_model = [param.grad for param in model.parameters()]
        # Pull out the y variables that we need for the LL problem
        y_weights_grad = grad_model[4].to(self.func.device)
        y_biases_grad = grad_model[5].to(self.func.device)
        grad_y_weights = y_weights_grad.flatten()
        grad_y_biases = y_biases_grad
        grad_y = np.concatenate((grad_y_weights.cpu().numpy(), grad_y_biases.cpu().numpy()))
        # Gradient of the LL objective function wrt x
        grad_model.pop(5); grad_model.pop(4)
        grad_x = self.unpack_weights(model, grad_model)
        return grad_x, grad_y
    
    
    def inequality_constraint_jacob(self, num_constr, train_task_loader, model, oldmodel, stream_task_training_list):
        """
        Computes the Jacobian matrices of the inequality constraints
        """
        for j in range(num_constr):
            grad_x, grad_y = self.inequality_constraint_task_grad(j, train_task_loader, model, oldmodel, stream_task_training_list)
            grad_x = np.reshape(grad_x, (1,grad_x.shape[0]))
            grad_y = np.reshape(grad_y, (1,grad_y.shape[0]))
            if j == 0:
              jacob_x = grad_x
              jacob_y = grad_y
            else:
              jacob_x = np.concatenate((jacob_x, grad_x), axis=0)
              jacob_y = np.concatenate((jacob_y, grad_y), axis=0)
    
        return jacob_x, jacob_y
    
    
    def update_llp_constr(self, train_loader, train_task_loader, model, oldmodel, num_constr, y, LL_iters, stream_training, stream_task_training_list): 
        """
        Updates the LL variables when the LL problem is constrained
        """        
        def add_penalty_term(output,model,oldmodel):
          pen_term = 0
          for j in range(num_constr):
            inconstr_value = self.inequality_constraint_task(j, train_task_loader, model, oldmodel, stream_task_training_list)
            pen_term = pen_term + torch.max(torch.Tensor([0]).to(self.func.device),-(inconstr_value))      
          output = output + (1/self.pen_param)*pen_term  
          return output
        
        y_weights = torch.from_numpy(y[0]).to(self.func.device)
        y_biases = torch.from_numpy(y[1]).to(self.func.device)
        for i in range(LL_iters):
            # Sample a new training batch
            if len(stream_training) == 0:
              stream_training = self.create_stream(train_loader)
            train_X, train_y = self.rand_samp_no_replace(stream_training)
            C, u = train_X.to(self.func.device), train_y.to(self.func.device)            
            # Obtain gradient of the LL objective function wrt y at point (x,y)
            u = torch.flatten(u) 
            output = self.func.loss_func(model(C), u)
            output = add_penalty_term(output,model,oldmodel) 
            model.zero_grad()
            # Calculate the gradient
            output.backward() 
            grad_model = [param.grad for param in model.parameters()]
            # Pull out the y variables that we need for the LLP
            y_weights_grad = grad_model[4].to(self.func.device)
            y_biases_grad = grad_model[5].to(self.func.device)
            # Update the variables
            y_weights = y_weights - self.ll_lr * y_weights_grad.flatten()
            y_biases = y_biases - self.ll_lr * y_biases_grad.flatten()
            # Update the model before returning
            model = self.update_model_wrt_y(model, y_weights, y_biases)
        return [y_weights, y_biases], model.to(self.func.device)
    
    
    def update_llp(self, train_loader, model, y, LL_iters, stream_training):
        """
        Updates the LL variables by taking a gradient descent step for the LL problem
        """   
        y_weights = torch.from_numpy(y[0]).to(self.func.device)
        y_biases = torch.from_numpy(y[1]).to(self.func.device)
        for i in range(LL_iters):
            train_X, train_y, stream_training = self.minibatch(train_loader,stream_training)
            C, u = train_X.to(self.func.device), train_y.to(self.func.device)
            # Obtain gradient of the LL objective function wrt y at point (x,y)
            u = torch.flatten(u) 
            output = self.func.loss_func(model(C), u)
            model.zero_grad()
            # Calculate the gradient
            output.backward() 
            grad_model = [param.grad for param in model.parameters()]
            # Pull out the y variables that we need for the LLP
            y_weights_grad = grad_model[4].to(self.func.device)
            y_biases_grad = grad_model[5].to(self.func.device)
            y_weights = y_weights - self.ll_lr * y_weights_grad.flatten()
            y_biases = y_biases - self.ll_lr * y_biases_grad.flatten()
            # Update the model before returning
            model = self.update_model_wrt_y(model, y_weights, y_biases)
        return [y_weights, y_biases], model.to(self.func.device)
    
    
    def darts_ulp_grad(self, train_loader, test_loader, model, y, y_orig, stream_training, stream_validation):
        """
        Updates the UL variables based on the DARTS step for the UL problem
        """
        y_orig = np.concatenate((torch.from_numpy(y_orig[0]).to(self.func.device).cpu(), torch.from_numpy(y_orig[1]).to(self.func.device).cpu()))
        train_X, train_y, stream_training = self.minibatch(train_loader,stream_training)
        valid_X, valid_y, stream_validation = self.minibatch(test_loader,stream_validation)
        C, u = train_X.to(self.func.device), train_y.to(self.func.device)
        c_valid, u_valid = valid_X.to(self.func.device), valid_y.to(self.func.device)
        
        # Gradient of the UL objective function wrt y
        u_valid = torch.flatten(u_valid) 
        output = self.func.loss_func(model(c_valid), u_valid)
        model.zero_grad()
        output.backward()
        ulp_grad_model = [param.grad for param in model.parameters()]
        grad_y_prime_weights = ulp_grad_model[4].flatten()
        grad_y_prime_biases = ulp_grad_model[5]
        grad_y_prime = np.concatenate((grad_y_prime_weights.cpu().numpy(), grad_y_prime_biases.cpu().numpy()))
        grad_norm = np.linalg.norm(grad_y_prime)
        # Remove the delta variables
        ulp_grad_model.pop(5); ulp_grad_model.pop(4) 
        grad_x_prime = self.unpack_weights(model, ulp_grad_model)
        
        ep = 0.01 / grad_norm
        # Define y+ and y-
        y_plus = y_orig + ep * grad_y_prime
        y_plus_biases = y_plus[len(y_plus)-10:]; y_plus_weights = y_plus[:len(y_plus)-10]
        y_minus = y_orig - ep * grad_y_prime
        y_minus_biases = y_minus[len(y_minus)-10:]; y_minus_weights = y_minus[:len(y_minus)-10]
        
        # Gradient of the LL objective function wrt x at the point (x,y+)
        model = self.update_model_wrt_y(model, torch.from_numpy(y_plus_weights), torch.from_numpy(y_plus_biases)) ######
        u = torch.flatten(u) 
        out_plus = self.func.loss_func(model(C), u)
        model.zero_grad()
        out_plus.backward()
        grad_x_plus = [plus_grad.grad for plus_grad in model.parameters()]
        # Remove the y variables
        grad_x_plus.pop(5); grad_x_plus.pop(4) 
        grad_x_plus = self.unpack_weights(model, grad_x_plus)
        
        # Gradient of the LL objective function wrt x at the point (x,y-)
        model = self.update_model_wrt_y(model, torch.from_numpy(y_minus_weights), torch.from_numpy(y_minus_biases))
        u = torch.flatten(u) 
        out_minus = self.func.loss_func(model(C), u)
        model.zero_grad()
        out_minus.backward()
        grad_x_minus = [minus_grad.grad for minus_grad in model.parameters()]
        # Remove the y variables
        grad_x_minus.pop(5); grad_x_minus.pop(4) 
        grad_x_minus = self.unpack_weights(model, grad_x_minus)
    
        grad_approx = grad_x_prime - self.ll_lr * ((grad_x_plus - grad_x_minus) / (2 * ep))
        # Normalize the direction
        if self.normalize:
            grad_approx = grad_approx / np.linalg.norm(grad_approx, np.inf)
        # Repack the weights
        ulp_grad = self.repack_weights(model, grad_approx) 
        ulp_grad.pop(5); ulp_grad.pop(4)
        
        # Update the UL variables
        model = self.update_model_wrt_x(model, ulp_grad)
        model = self.update_model_wrt_y(model, y[0], y[1])
        return model.to(self.func.device)
    
    
    def calc_dx_ulp(self, train_loader, test_loader, y, model, stream_training, stream_validation):
        """
        Updates the UL variables based on the BSG step for the UL problem
        """ 
        y_orig = np.concatenate((y[0].detach().cpu().numpy(), y[1].detach().cpu().numpy()))
        # y_orig = np.concatenate((torch.from_numpy(y[0]).to(self.func.device).cpu(), torch.from_numpy(y[1]).to(self.func.device).cpu()))
        
        train_X, train_y, stream_training = self.minibatch(train_loader,stream_training)
        valid_X, valid_y, stream_validation = self.minibatch(test_loader,stream_validation)
        C, u = train_X.to(self.func.device), train_y.to(self.func.device)
        c_valid, u_valid = valid_X.to(self.func.device), valid_y.to(self.func.device)
        
        # Gradient of the UL objective function wrt x and y
        u_valid = torch.flatten(u_valid) 
        output = self.func.loss_func(model(c_valid), u_valid)
        model.zero_grad()
        output.backward()
        ulp_grad_model = [param.grad for param in model.parameters()]
        grad_y_weights = ulp_grad_model[4].flatten()
        grad_y_biases = ulp_grad_model[5]
        ulp_grad_y = np.concatenate((grad_y_weights.cpu().numpy(), grad_y_biases.cpu().numpy()))
        # Gradient of the UL problem wrt x
        ulp_grad_model.pop(5); ulp_grad_model.pop(4)
        ulp_grad_x = self.unpack_weights(model, ulp_grad_model)
       
        if self.hess == False: 
            # # BSG-1
            
            # Gradient of the LL objective function wrt y
            u = torch.flatten(u)
            out_prime = self.func.loss_func(model(C), u)
            model.zero_grad()
            out_prime.backward()
            llp_grad_model = [param.grad for param in model.parameters()]
            LL_grad_y_weights = llp_grad_model[4].flatten()
            ll_grad_y_biases = llp_grad_model[5]
            llp_grad_y = np.concatenate((LL_grad_y_weights.cpu().numpy(), ll_grad_y_biases.cpu().numpy()))
            # Gradient of the LL problem wrt x
            llp_grad_model.pop(5); llp_grad_model.pop(4)
            llp_grad_x = self.unpack_weights(model, llp_grad_model)
            
            # Compute the bsg-1 direction
            bsg = ulp_grad_x - ((np.dot(llp_grad_y, ulp_grad_y)) / np.linalg.norm(llp_grad_y)**2) * llp_grad_x

        elif self.hess == 'CG-FD':            
            # # BSG-QN

            oldmodel = copy.deepcopy(model)
            
            ulp_grad_y = np.reshape(ulp_grad_y, (-1, 1))
    
            def mv(v):
                nonlocal u
                nonlocal model

                ## Finite difference approximation
                v_norm = np.linalg.norm(v)                 
                ep = 1e-1 #/ v_norm
                # Define y+ and y-
                y_plus = y_orig + ep * v
                y_plus_biases = y_plus[len(y_plus)-10:]; y_plus_weights = y_plus[:len(y_plus)-10]
                y_minus = y_orig - ep * v
                y_minus_biases = y_minus[len(y_minus)-10:]; y_minus_weights = y_minus[:len(y_minus)-10]
                
                # Gradient of the LL objective function wrt y at the point (x,y+)
                model = self.update_model_wrt_y(model, torch.from_numpy(y_plus_weights), torch.from_numpy(y_plus_biases)) ######
                u = torch.flatten(u) 
                out_plus = self.func.loss_func(model(C), u)
                model.zero_grad()
                out_plus.backward()
                llp_grad_model = [plus_grad.grad for plus_grad in model.parameters()]
                grad_y_prime_weights = llp_grad_model[4].flatten()
                grad_y_prime_biases = llp_grad_model[5]
                grad_y_plus = np.concatenate((grad_y_prime_weights.cpu().numpy(), grad_y_prime_biases.cpu().numpy()))
                
                # Gradient of the LL objective function wrt y at the point (x,y-)
                model = self.update_model_wrt_y(model, torch.from_numpy(y_minus_weights), torch.from_numpy(y_minus_biases))
                u = torch.flatten(u) 
                out_minus = self.func.loss_func(model(C), u)
                model.zero_grad()
                out_minus.backward()
                llp_grad_model = [minus_grad.grad for minus_grad in model.parameters()]
                grad_y_prime_weights = llp_grad_model[4].flatten()
                grad_y_prime_biases = llp_grad_model[5]
                grad_y_minus = np.concatenate((grad_y_prime_weights.cpu().numpy(), grad_y_prime_biases.cpu().numpy()))
                return (grad_y_plus - grad_y_minus) / (2 * ep)  
            
            self.hess_LLP_ll_vars_ll_vars_lin = LinearOperator((self.y_dim,self.y_dim), matvec=mv)
            
            lambda_adj, exit_code = cg(self.hess_LLP_ll_vars_ll_vars_lin,ulp_grad_y,x0=None, tol=1e-4, maxiter=3) #maxiter=100
            # lambda_adj = np.reshape(lambda_adj, (-1,1))
            ################################
            
            ## Finite difference approximation
            grad_norm = np.linalg.norm(lambda_adj)  
            ep = 0.01 / grad_norm
            # Define y+ and y-
            y_plus = y_orig + ep * lambda_adj
            y_plus_biases = y_plus[len(y_plus)-10:]; y_plus_weights = y_plus[:len(y_plus)-10]
            y_minus = y_orig - ep * lambda_adj
            y_minus_biases = y_minus[len(y_minus)-10:]; y_minus_weights = y_minus[:len(y_minus)-10]
            
            # Gradient of the LL objective function wrt x at the point (x,y+)
            model = self.update_model_wrt_y(model, torch.from_numpy(y_plus_weights), torch.from_numpy(y_plus_biases)) ######
            u = torch.flatten(u) 
            out_plus = self.func.loss_func(model(C), u)
            model.zero_grad()
            out_plus.backward()
            grad_x_plus = [plus_grad.grad for plus_grad in model.parameters()]
            # Remove the y variables
            grad_x_plus.pop(5); grad_x_plus.pop(4) 
            grad_x_plus = self.unpack_weights(model, grad_x_plus)
            
            # Gradient of the LL objective function wrt x at the point (x,y-)
            model = self.update_model_wrt_y(model, torch.from_numpy(y_minus_weights), torch.from_numpy(y_minus_biases))
            u = torch.flatten(u) 
            out_minus = self.func.loss_func(model(C), u)
            model.zero_grad()
            out_minus.backward()
            grad_x_minus = [minus_grad.grad for minus_grad in model.parameters()]
            # Remove the y variables
            grad_x_minus.pop(5); grad_x_minus.pop(4) 
            grad_x_minus = self.unpack_weights(model, grad_x_minus)

            bsg = ulp_grad_x - ((grad_x_plus - grad_x_minus) / (2 * ep))

            model = oldmodel

        else:
            print('There is something wrong with self.hess')
        
        # Normalize the direction
        if self.normalize:
            bsg = bsg / np.linalg.norm(bsg, np.inf)
            
        # Repack the weights
        ulp_grad = self.repack_weights(model, bsg) 
        ulp_grad.pop(5); ulp_grad.pop(4)
        
        # Update the UL variables
        model = self.update_model_wrt_x(model, ulp_grad)
        return model.to(self.func.device)
       
    
    def calc_dx_ulp_constr(self, train_loader, train_task_loader, test_loader, jacob_x, jacob_y, inconstr_vec, lam0, num_constr, lagr_mul,  y, model, oldmodel, batch_num, stream_training, stream_task_training_list, stream_validation):
        """
        Updates the UL variables based on the BSG step for the UL problem in the LL constrained case
        """         
        def add_Lagrangian_term(output,model,oldmodel,lagr_mul):
          lagr_term = 0
          for j in range(num_constr):
            inconstr_value = self.inequality_constraint_task(j, train_task_loader, model, oldmodel, stream_task_training_list)
            lagr_mul = torch.tensor(lagr_mul).to(self.func.device)
            lagr_term = lagr_term + lagr_mul[j]*inconstr_value      
          return lagr_term
    
        # Sample new batches
        if len(stream_training) == 0:
          stream_training = self.create_stream(train_loader)
        if len(stream_validation) == 0:
          stream_validation = self.create_stream(test_loader)
        train_X, train_y = self.rand_samp_no_replace(stream_training)
        valid_X, valid_y = self.rand_samp_no_replace(stream_validation)
        C, u = train_X.to(self.func.device), train_y.to(self.func.device)
        c_valid, u_valid = valid_X.to(self.func.device), valid_y.to(self.func.device)
        
        # Gradient of the UL objective function wrt x and y
        u_valid = torch.flatten(u_valid) 
        output = self.func.loss_func(model(c_valid), u_valid)    
        model.zero_grad()
        output.backward()
        ulp_grad_model = [param.grad for param in model.parameters()]
        grad_y_weights = ulp_grad_model[4].flatten()
        grad_y_biases = ulp_grad_model[5]
        ulp_grad_y = np.concatenate((grad_y_weights.cpu().numpy(), grad_y_biases.cpu().numpy()))
        # Gradient of the UL objective function wrt x
        ulp_grad_model.pop(5); ulp_grad_model.pop(4)
        ulp_grad_x = self.unpack_weights(model, ulp_grad_model)
    
        ulp_grad_x = np.reshape(ulp_grad_x, (ulp_grad_x.shape[0],1)) 
        ulp_grad_y = np.reshape(ulp_grad_y, (ulp_grad_y.shape[0],1))

        if self.hess == False: 
            # # BSG-1
           
            # Gradient of the Lagrangian wrt x and y
            u = torch.flatten(u) 
            out_prime = self.func.loss_func(model(C), u)
            out_prime = add_Lagrangian_term(out_prime,model,oldmodel,lagr_mul)  
            model.zero_grad()
            out_prime.backward()
            llp_lagr_grad_model = [param.grad for param in model.parameters()]
            ll_lagr_grad_y_weights = llp_lagr_grad_model[4].flatten()
            ll_lagr_grad_y_biases = llp_lagr_grad_model[5]
            llp_lagr_grad_y = np.concatenate((ll_lagr_grad_y_weights.cpu().numpy(), ll_lagr_grad_y_biases.cpu().numpy()))
            # Gradient of the LL objective function wrt x
            llp_lagr_grad_model.pop(5); llp_lagr_grad_model.pop(4)
            llp_lagr_grad_x = self.unpack_weights(model, llp_lagr_grad_model)
        
            llp_lagr_grad_x = np.reshape(llp_lagr_grad_x, (llp_lagr_grad_x.shape[0],1)) 
            llp_lagr_grad_y = np.reshape(llp_lagr_grad_y, (llp_lagr_grad_y.shape[0],1))     
        
            rhs = np.concatenate((ulp_grad_y,np.zeros((lagr_mul.shape[0],1))), axis=0)
            if batch_num == 0:
              lam0 = np.random.rand(llp_lagr_grad_y.shape[0] + lagr_mul.shape[0],1)
            # lam = self.gmres(llp_lagr_grad_y, jacob_y, inconstr_vec, rhs, lagr_mul, lam0, max_it=50)
            
            def mv(v):
                aa = v[:self.y_dim].reshape(-1,1)
                bb = v[self.y_dim:].reshape(-1,1)            
                   
                aux_1 = np.array(np.matmul(llp_lagr_grad_y, np.matmul(llp_lagr_grad_y.T,aa)) + np.matmul(np.multiply(lagr_mul,jacob_y).T,bb)) 
                aux_2 = np.array(np.matmul(jacob_y,aa) + np.multiply(inconstr_vec,bb))              
                return np.concatenate((aux_1,aux_2),axis=0) 
                 
            self.ll_kkt_lin = LinearOperator((self.y_dim + num_constr, self.y_dim + num_constr), matvec=mv)                 
            lam, _ = gmres(self.ll_kkt_lin, rhs, x0=lam0, tol=1e-4, maxiter=50)  #tol=1e-4, maxiter=5
            # lam = self.gmres(llp_lagr_grad_y, jacob_y, inconstr_vec, rhs, lagr_mul, lam0, max_it=50)
            lam = lam.reshape(-1,1)
        
            # Compute the bsg direction
            bsg = ulp_grad_x - np.matmul(llp_lagr_grad_x,np.matmul(llp_lagr_grad_y.T,lam[0:llp_lagr_grad_y.shape[0]])) - np.matmul(np.multiply(lagr_mul,jacob_x).T,lam[llp_lagr_grad_y.shape[0]:])
            

        elif self.hess == 'CG-FD':            
            # # BSG-N-FD
           
           y_orig = np.concatenate((y[0].detach().cpu().numpy(), y[1].detach().cpu().numpy())) 
           
           oldmodel = copy.deepcopy(model)          
           ulp_grad_y = np.reshape(ulp_grad_y, (-1, 1))
           
           rhs = np.concatenate((ulp_grad_y,np.zeros((lagr_mul.shape[0],1))), axis=0)
           if batch_num == 0:
             lam0 = np.random.rand(self.y_dim + lagr_mul.shape[0],1)
   
           def mv(v):
               nonlocal u
               nonlocal model

               aa = v[:self.y_dim].reshape(-1,1)
               bb = v[self.y_dim:].reshape(-1,1)  

               ## Finite difference approximation
               v_norm = np.linalg.norm(v[:self.y_dim])                 
               ep = 1e-1 #/ v_norm
               # Define y+ and y-
               y_plus = y_orig + ep * v[:self.y_dim]
               y_plus_biases = y_plus[len(y_plus)-10:]; y_plus_weights = y_plus[:len(y_plus)-10]
               y_minus = y_orig - ep * v[:self.y_dim]
               y_minus_biases = y_minus[len(y_minus)-10:]; y_minus_weights = y_minus[:len(y_minus)-10]
               
               # Gradient of the LL Lagrangian wrt y at the point (x,y+)
               model = self.update_model_wrt_y(model, torch.from_numpy(y_plus_weights), torch.from_numpy(y_plus_biases)) ######
               u = torch.flatten(u) 
               out_plus = self.func.loss_func(model(C), u)
               out_plus = add_Lagrangian_term(out_plus,model,oldmodel,lagr_mul)
               model.zero_grad()
               out_plus.backward()
               llp_grad_model = [plus_grad.grad for plus_grad in model.parameters()]
               grad_y_prime_weights = llp_grad_model[4].flatten()
               grad_y_prime_biases = llp_grad_model[5]
               grad_y_plus = np.concatenate((grad_y_prime_weights.cpu().numpy(), grad_y_prime_biases.cpu().numpy()))
               
               # Gradient of the LL Lagrangian wrt y at the point (x,y-)
               model = self.update_model_wrt_y(model, torch.from_numpy(y_minus_weights), torch.from_numpy(y_minus_biases))
               u = torch.flatten(u) 
               out_minus = self.func.loss_func(model(C), u)
               out_minus = add_Lagrangian_term(out_minus,model,oldmodel,lagr_mul)
               model.zero_grad()
               out_minus.backward()
               llp_grad_model = [minus_grad.grad for minus_grad in model.parameters()]
               grad_y_prime_weights = llp_grad_model[4].flatten()
               grad_y_prime_biases = llp_grad_model[5]
               grad_y_minus = np.concatenate((grad_y_prime_weights.cpu().numpy(), grad_y_prime_biases.cpu().numpy()))
              
               aux_1 = ((grad_y_plus - grad_y_minus)/(2*ep)).reshape(-1,1) + np.matmul(np.multiply(lagr_mul,jacob_y).T,bb)             
               aux_2 = np.array(np.matmul(jacob_y,aa) + np.multiply(inconstr_vec,bb))               
               return  np.concatenate((aux_1,aux_2),axis=0) 
           
           # self.hess_LLP_ll_vars_ll_vars_lin = LinearOperator((self.y_dim,self.y_dim), matvec=mv)
           
           # lambda_adj, exit_code = cg(self.hess_LLP_ll_vars_ll_vars_lin,ulp_grad_y,x0=None, tol=1e-4, maxiter=3) #maxiter=100
           # lambda_adj = np.reshape(lambda_adj, (-1,1))

           self.ll_kkt_lin = LinearOperator((self.y_dim + num_constr, self.y_dim + num_constr), matvec=mv)                 
           lam, _ = gmres(self.ll_kkt_lin, rhs, x0=lam0, tol=1e-4, maxiter=3)  

           ################################
           
           ## Finite difference approximation
           grad_norm = np.linalg.norm(lam)  
           ep = 0.01 / grad_norm
           # Define y+ and y-
           y_plus = y_orig + ep * lam[:self.y_dim]
           y_plus_biases = y_plus[len(y_plus)-10:]; y_plus_weights = y_plus[:len(y_plus)-10]
           y_minus = y_orig - ep * lam[:self.y_dim]
           y_minus_biases = y_minus[len(y_minus)-10:]; y_minus_weights = y_minus[:len(y_minus)-10]
           
           # Gradient of the LL Lagrangian wrt x at the point (x,y+)
           model = self.update_model_wrt_y(model, torch.from_numpy(y_plus_weights), torch.from_numpy(y_plus_biases)) ######
           u = torch.flatten(u) 
           out_plus = self.func.loss_func(model(C), u)
           out_plus = add_Lagrangian_term(out_plus,model,oldmodel,lagr_mul)
           model.zero_grad()
           out_plus.backward()
           grad_x_plus = [plus_grad.grad for plus_grad in model.parameters()]
           # Remove the y variables
           grad_x_plus.pop(5); grad_x_plus.pop(4) 
           grad_x_plus = self.unpack_weights(model, grad_x_plus)
           
           # Gradient of the LL Lagrangian wrt x at the point (x,y-)
           model = self.update_model_wrt_y(model, torch.from_numpy(y_minus_weights), torch.from_numpy(y_minus_biases))
           u = torch.flatten(u) 
           out_minus = self.func.loss_func(model(C), u)
           out_minus = add_Lagrangian_term(out_minus,model,oldmodel,lagr_mul)  
           model.zero_grad()
           out_minus.backward()
           grad_x_minus = [minus_grad.grad for minus_grad in model.parameters()]
           # Remove the y variables
           grad_x_minus.pop(5); grad_x_minus.pop(4) 
           grad_x_minus = self.unpack_weights(model, grad_x_minus)
           
           aux_1 = ((grad_x_plus - grad_x_minus) / (2 * ep)).reshape(-1,1)
           aux_2 = np.matmul(np.multiply(lagr_mul,jacob_x).T,lam[self.y_dim:].reshape(-1,1))
           
           bsg = ulp_grad_x - (aux_1 + aux_2)
                    
           model = oldmodel

        else:
           print('There is something wrong with self.hess')           
           
        bsg = bsg.flatten()
        # Normalize the direction
        if self.normalize:
            bsg = bsg / np.linalg.norm(bsg, np.inf)
            
        #Repack the weights
        ulp_grad = self.repack_weights(model, bsg) 
        ulp_grad.pop(5); ulp_grad.pop(4)
        
        # Update the UL variables
        model = self.update_model_wrt_x(model, ulp_grad)
        return lam, model.to(self.func.device)
    

    def stocbio_ulp(self, train_loader, test_loader, y, model, stream_training, stream_validation): 
        """
        StocBiO algorithm. See https://github.com/JunjieYang97/stocBiO/blob/master/Hyperparameter-optimization/experimental/stocBiO.py
        """      

        train_X, train_y, stream_training = self.minibatch(train_loader,stream_training)
        valid_X, valid_y, stream_validation = self.minibatch(test_loader,stream_validation)
        C, u = train_X.to(self.func.device), train_y.to(self.func.device)
        c_valid, u_valid = valid_X.to(self.func.device), valid_y.to(self.func.device)

        # Gradient of the UL objective function wrt x and y
        u_valid = torch.flatten(u_valid) 
        output = self.func.loss_func(model(c_valid), u_valid)
        model.zero_grad()
        output.backward()
        ulp_grad_model = [param.grad for param in model.parameters()]
        grad_y_weights = ulp_grad_model[4].flatten()
        grad_y_biases = ulp_grad_model[5]
        ulp_grad_y = np.concatenate((grad_y_weights.cpu().numpy(), grad_y_biases.cpu().numpy()))
        # Gradient of the UL problem wrt x
        ulp_grad_model.pop(5); ulp_grad_model.pop(4)
        ulp_grad_x = self.unpack_weights(model, ulp_grad_model)
        
        v_0 = torch.from_numpy(ulp_grad_y).detach().to(self.func.device) 
        
        eta = 0.05
        hessian_q = 2
        
        # Obtain gradient of the LL objective function wrt y at point (x,y)
        u = torch.flatten(u)
        output = self.func.loss_func(model(C), u)
        model.zero_grad()
        # Calculate the gradient
        output.backward(create_graph=True,retain_graph=True) 
        for idx, param in enumerate(model.parameters()):
            # print('FFFF',param.grad)
            if idx == 4:
                G_gradient_y_weights = param - eta * param.grad
            elif idx == 5:
                G_gradient_y_biases = param - eta * param.grad
        
        G_gradient = torch.cat((G_gradient_y_weights.flatten(), G_gradient_y_biases.flatten()), axis=0)
        
        # Hessian
        z_list = []

        for _ in range(hessian_q):
            Jacobian = torch.dot(G_gradient.double(), torch.squeeze(v_0).double())
            Jacobian.backward(retain_graph=True) 
            for idx, param in enumerate(model.parameters()):
                if idx == 4:
                    v_new_weights = param.grad
                elif idx == 5:
                    v_new_biases = param.grad
            # v_new = torch.autograd.grad(Jacobian, torch.cat((v_new_weights.flatten(), v_new_biases.flatten()), axis=0), create_graph=True)[0]
            v_new = torch.cat((v_new_weights.flatten(), v_new_biases.flatten()), axis=0)
            v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
            z_list.append(v_0) 
        v_Q = eta*v_0+torch.sum(torch.stack(z_list), dim=0).detach().to(self.func.device) 
        # v_Q = eta*v_0+eta*torch.sum(torch.stack(z_list), dim=0) #This would be the correct version, but StocBiO uses the line above
 
        # Gradient of the LL objective function wrt y
        u = torch.flatten(u)
        out_prime = self.func.loss_func(model(C), u)
        model.zero_grad()
        out_prime.backward(create_graph=True,retain_graph=True)
      
        # # Gyx_gradient
        # Gy_gradient = torch.reshape(torch.from_numpy(llp_grad_y), [-1])
        for idx, param in enumerate(model.parameters()):
            if idx == 4:
                Gy_gradient_weights = param.grad
            elif idx == 5:
                Gy_gradient_biases = param.grad
        
        Gy_gradient = torch.cat((Gy_gradient_weights.flatten(), Gy_gradient_biases.flatten()), axis=0)
        
        output = torch.dot(Gy_gradient.double(), torch.squeeze(v_Q).double())
        output.backward(retain_graph=True) 
        # Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient.double(), v_Q.detach().double()), ul_vars_list, create_graph=True)[0]
        Gyx_gradient_list = []
        for idx, param in enumerate(model.parameters()):
            if 0 <= idx <=3:
                Gyx_gradient_list.append(param.grad.flatten())
        
        Gyx_gradient = torch.cat(Gyx_gradient_list,dim=0)

        bsg = ulp_grad_x - Gyx_gradient.detach().cpu().numpy()
        
        # Repack the weights
        ulp_grad = self.repack_weights(model, bsg) 
        ulp_grad.pop(5); ulp_grad.pop(4)
        
        # Update the UL variables
        model = self.update_model_wrt_x(model, ulp_grad)
        return model.to(self.func.device)
    
    
    def true_function_value_constr(self, train_loader, training_task_loaders, test_loader, y, model, oldmodel, num_constr, batch, LLP_steps, true_func_list, stream_training, stream_task_training_list, stream_validation): 
        """
        Computes the true objective function of the bilevel problem in the LL constrained case
        """   
        y_weights_orig = torch.from_numpy(y[0]); y_biases_orig = torch.from_numpy(y[1])
        # Reinitialize the y variables
        y, model = self.generate_y(model, randomize = True) 
        y, model = self.update_llp_constr(train_loader, training_task_loaders, model, oldmodel, num_constr, y, LLP_steps, stream_training, stream_task_training_list) 
        
        if len(stream_validation) == 0:
            stream_validation = self.create_stream(test_loader)
        valid_X, valid_y = self.rand_samp_no_replace(stream_validation)
        c_valid = valid_X.to(self.func.device); u_valid = valid_y.to(self.func.device)
        
        u_valid = torch.flatten(u_valid) 
        true_func_list.append(float(self.func.loss_func(model(c_valid), u_valid)))
        model = self.update_model_wrt_y(model, y_weights_orig, y_biases_orig)
        return true_func_list
    
    
    def true_function_value(self, train_loader, test_loader, y, model, llp_steps_true_funct, true_func_list, stream_training, stream_validation):
        """
        Computes the true objective function of the bilevel problem 
        """
        y_weights_orig = torch.from_numpy(y[0]); y_biases_orig = torch.from_numpy(y[1])
        # Reinitialize the y variables
        y, model = self.generate_y(model, randomize = True) 
        y, model = self.update_llp(train_loader, model, y, 0.0007, llp_steps_true_funct, stream_training)
        
        valid_X, valid_y, stream_validation = self.minibatch(test_loader,stream_validation)
        c_valid = valid_X.to(self.func.device); u_valid = valid_y.to(self.func.device)
        
        u_valid = torch.flatten(u_valid) 
        true_func_list.append(float(self.func.loss_func(model(c_valid), u_valid)))
        model = self.update_model_wrt_y(model, y_weights_orig, y_biases_orig)
        return true_func_list


    def main_algorithm(self):
    	"""
    	Main body of a bilevel stochastic algorithm
    	"""     
    	# Initialize lists
    	func_val_list = []
    	true_func_list = []
    	time_list = []
    	
    	# Timing data
    	cur_time = time.time()
    	time_tru_func_tot = 0   
    
    	# LL constrained case
    	if self.constrained:
    		stream_task_training_list = [] 
    
    	self.ul_lr = self.ul_lr_init
    	self.ll_lr = self.ll_lr_init
    	self.llp_iters = self.llp_iters_init
    	
    	model = self.model #self.func.generate_model() #self.model
    	
    	# Loop through the five tasks
    	for i in range(len(self.training_loaders)):
        	# if i <= 2:
        		istop = 0 
        		
        		orig_ul_lr = self.ul_lr
        		orig_ll_lr = self.ll_lr
        		
        		stream_training = self.create_stream(self.training_loaders[i])
        		stream_validation = self.create_stream(self.testing_loaders[i])
        		# First batch
        		valid_X, valid_y, stream_validation = self.minibatch(self.testing_loaders[i],stream_validation)
        		C = valid_X.to(self.func.device); u = valid_y.to(self.func.device)
        		
        		# LL constrained case
        		if self.constrained and i > 0:
        		  stream_task_training = self.create_stream(self.training_task_loaders[i])
        		  stream_task_training_list.append(stream_task_training)
        		  oldmodel = copy.deepcopy(model) 
        		
        		# Initialize the y variables (output weights of the network)
        		y, model = self.generate_y(model, randomize = True)
        		
        		for cur_epoch in range(self.max_epochs):
        			if istop:
        			  break
        			if  cur_epoch == 0:
        				u = torch.flatten(u) 
        				func_val_list.append(float(self.func.loss_func(model(C), u)))
        				time_list.append(time.time() - cur_time - time_tru_func_tot) 
        				if self.true_func == True: 
        						cur_time_aux = time.time() 
        
        						if (self.constrained and i > 0):
        							num_constr = i
        							true_func_list = self.true_function_value_constr(self.training_loaders[i], self.training_task_loaders, self.testing_loaders[i], y, model, oldmodel, num_constr, [], 50000, true_func_list, stream_training, stream_task_training_list, stream_validation)
        						else:
        							true_func_list = self.true_function_value(self.training_loaders[i], self.testing_loaders[i], y, model, 50000, true_func_list, stream_training, stream_validation)
        						
        						time_tru_func = time.time() - cur_time_aux 
        						time_tru_func_tot = time_tru_func_tot + time_tru_func 
        												
        			j = 1
        			total_err = 0
        			for batch in range(len(self.training_loaders[i])):
        				# Check if a new task must be added
        				if not(self.use_stopping_iter) and (time.time() - cur_time - time_tru_func_tot) >= self.stopping_times[i]: 
        					istop = 1
        					break
        				
        				valid_X, valid_y, stream_validation = self.minibatch(self.testing_loaders[i],stream_validation)
        				c_valid = valid_X.to(self.func.device); u_valid = valid_y.to(self.func.device)
        				u_valid = torch.flatten(u_valid) 
        				pre_obj_val = float(self.func.loss_func(model(c_valid), u_valid).detach().cpu().numpy())
        				
        				# Update the LL variables (delta) with a single step
        				y_orig = y
        				if (self.constrained and i > 0):
        				  num_constr = i
        				  y, model = self.update_llp_constr(self.training_loaders[i], self.training_task_loaders, model, oldmodel, num_constr, y, self.llp_iters, stream_training, stream_task_training_list) 
        				else:
        				  y, model = self.update_llp(self.training_loaders[i], model, y, self.llp_iters, stream_training) 
        				
        				# LL constrained case
        				if (self.constrained and i > 0):
        				  jacob_x, jacob_y = self.inequality_constraint_jacob(i, self.training_task_loaders, model, oldmodel, stream_task_training_list)
        				  inconstr_vec = self.inequality_constraint(i, self.training_task_loaders, model, oldmodel, stream_task_training_list)
        	
        				  if len(stream_training) == 0:
        				    stream_training = self.create_stream(self.training_loaders[i])
        				  train_X, train_y = self.rand_samp_no_replace(stream_training)
        				  C, u = train_X.to(self.func.device), train_y.to(self.func.device)
        	
        				  # Gradient of LL objective function wrt y without penalty
        				  u = torch.flatten(u) 
        				  out = self.func.loss_func(model(C), u)  
        				  model.zero_grad()
        				  out.backward()
        				  llp_grad_model = [param.grad for param in model.parameters()]
        				  LL_grad_y_weights = llp_grad_model[4].flatten()
        				  ll_grad_y_biases = llp_grad_model[5]
        				  llp_grad_y = np.concatenate((LL_grad_y_weights.cpu().numpy(), ll_grad_y_biases.cpu().numpy()))
        				  llp_grad_y = np.reshape(llp_grad_y, (llp_grad_y.shape[0],1))
        	
        				  if (batch == 0):
        				    lagr_mul = -np.matmul(jacob_y,llp_grad_y)  
           
      				  # lagr_mul = self.linear_cg_kkt(jacob_y, inconstr_vec, -np.matmul(jacob_y,llp_grad_y), 100, 10**-4, lagr_mul)

        				  def mv(v):				    
        				      out = np.matmul(jacob_y,np.matmul(jacob_y.T,v.reshape(-1,1))) + np.multiply(inconstr_vec**2,v.reshape(-1,1)) 
        				      return out  
                            
        				  G = LinearOperator((num_constr,num_constr), matvec=mv)
                            
        				  lagr_mul, exit_code = cg(G,-np.matmul(jacob_y,llp_grad_y),x0=lagr_mul, tol=1e-4, maxiter=3) #maxiter=100
        				  lagr_mul = np.reshape(lagr_mul, (-1,1))

        				# Compute the gradient of the UL objective function wrt x (weights of the hidden layers except the last)
        				if self.algo == 'darts':
        					model = self.darts_ulp_grad(self.training_loaders[i], self.testing_loaders[i], model, y, y_orig, stream_training, stream_validation)
        				elif self.algo == 'bsg':
        				  if self.constrained and i > 0:
        				  	if batch == 0:
        				  	  lam = []
        				  	lam, model = self.calc_dx_ulp_constr(self.training_loaders[i], self.training_task_loaders, self.testing_loaders[i], jacob_x, jacob_y, inconstr_vec, lam, i, lagr_mul, y, model, oldmodel, batch, stream_training, stream_task_training_list, stream_validation) 
        				  else:
        				  	model = self.calc_dx_ulp(self.training_loaders[i], self.testing_loaders[i], y, model, stream_training, stream_validation)	              
        				elif self.algo == 'stocbio':
        				  	model = self.stocbio_ulp(self.training_loaders[i], self.testing_loaders[i], y, model, stream_training, stream_validation)	                                          
                            	
        				j += 1
        				# Convert y back to numpy arrays
        				y = [y[0].cpu().numpy(), y[1].cpu().numpy()]
        				valid_X, valid_y, stream_validation = self.minibatch(self.testing_loaders[i],stream_validation)
        				c_valid = valid_X.to(self.func.device); u_valid = valid_y.to(self.func.device)
        				
        				u_pred = model(c_valid).to(self.func.device)
        				total_err += (u_pred.max(dim=1)[1] != u_valid).sum().item()
        				u_valid = torch.flatten(u_valid) 
        				func_val_list.append(float(self.func.loss_func(u_pred, u_valid).detach().cpu().numpy()))
        				time_list.append(time.time() - cur_time) 
        								
        				if self.true_func == True: 
        					time_list.append(time.time() - cur_time  - time_tru_func_tot)
        
        					cur_time_aux = time.time() 
        					
        					if (self.constrained and i > 0):
        					  num_constr = i
        					  true_func_list = self.true_function_value_constr(self.training_loaders[i], self.training_task_loaders, self.testing_loaders[i], y, model, oldmodel, num_constr, batch, 50000, true_func_list, stream_training, stream_task_training_list, stream_validation)
        					else:
        					  true_func_list = self.true_function_value(self.training_loaders[i], self.testing_loaders[i], y, model, 50000, true_func_list, stream_training, stream_validation) 
        
        					time_tru_func = time.time() - cur_time_aux 
        					time_tru_func_tot = time_tru_func_tot + time_tru_func                        
        						
        				if self.inc_acc == True:
        					if self.llp_iters > 30:
        						self.llp_iters = 30
        					else:
        						u_valid = torch.flatten(u_valid) 
        						post_obj_val = float(self.func.loss_func(u_pred, u_valid).detach().cpu().numpy())
        						obj_val_diff = abs(post_obj_val - pre_obj_val)
        						if obj_val_diff <= 1e-2:
        							self.llp_iters += 1
        				
        				# Update the learning rates
        				if self.algo == 'bsg':
        					self.ul_lr = orig_ul_lr/j
        					self.ll_lr = orig_ll_lr
        				elif self.algo == 'darts':
        					self.ul_lr = orig_ul_lr/j
        					self.ll_lr = orig_ll_lr
                        
        				if self.iprint >= 2:
        				        print("Algo: ",self.algo," task: ",i,' batch: ',batch,' f_u: ',func_val_list[len(func_val_list)-1],' time: ',time.time() - cur_time - time_tru_func_tot,' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)    				
        		
        		if self.iprint >= 1:
        			print("Algo: ",self.algo," task: ",i,' f_u: ',func_val_list[len(func_val_list)-1],' time: ',time.time() - cur_time - time_tru_func_tot,' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)
    		# string_name = Algo + '_task' + str(i) + '_values.csv'
    		# pd.DataFrame(func_val_list).to_csv(string_name, index=False)
    		# string_name = Algo + '_task' + str(i) + '_true_func_vals.csv'
    		# pd.DataFrame(true_func_list).to_csv(string_name, index=False)
    
    	return [func_val_list, true_func_list, time_list]
    
    
    def piecewise_func(self,x,boundaries,func_val):
      """
      Computes the value of a piecewise constant function at x
      """
      for i in range(len(boundaries)):
        if x <= boundaries[i]:
          return func_val[i]
      return func_val[len(boundaries)-1]
    
    
    def main_algorithm_avg_ci(self, num_rep=1):
        """
        Returns arrays with averages and 95% confidence interval half-widths for function values or true function values at each iteration obtained over multiple runs
        """ 
        self.set_seed(self.seed)
        # Solve the problem for the first time
        sol = self.main_algorithm() 
        values = sol[0]
        true_func_values = sol[1]
        times = sol[2]
        values_rep = np.zeros((len(values),num_rep))
        values_rep[:,0] = np.asarray(values)
        if self.true_func:
            true_func_values_rep = np.zeros((len(true_func_values),num_rep))
            true_func_values_rep[:,0] = np.asarray(true_func_values)
        # Solve the problem num_rep-1 times
        for i in range(num_rep-1):
          self.set_seed(self.seed+1+i)
          sol = self.main_algorithm() 
          if self.use_stopping_iter:
              values_rep[:,i+1] = np.asarray(sol[0])
              if self.true_func:
                  true_func_values_rep[:,i+1] = np.asarray(sol[1])
          else:
              values_rep[:,i+1] = list(map(lambda x: self.piecewise_func(x,sol[2],sol[0]),times))
              if self.true_func:
                  true_func_values_rep[:,i+1] = list(map(lambda x: self.piecewise_func(x,sol[2],sol[1]),times))
        values_avg = np.mean(values_rep, axis=1)
        values_ci = t.ppf(0.975,num_rep-1)*(np.sqrt(np.var(values_rep, ddof=1, axis=1))/np.sqrt(num_rep))
        if self.true_func:
            true_func_values_avg = np.mean(true_func_values_rep, axis=1)
            true_func_values_ci = t.ppf(0.975,num_rep-1)*(np.sqrt(np.var(true_func_values_rep, ddof=1, axis=1))/np.sqrt(num_rep))
        else:
            true_func_values_avg = []
            true_func_values_ci = []
             
        return values_avg, values_ci, true_func_values_avg, true_func_values_ci, times







