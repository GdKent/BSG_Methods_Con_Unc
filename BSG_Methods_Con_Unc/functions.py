import os
import numpy as np
import random
import math
import torch
import torch.nn as nn
import time


from torchvision import datasets, transforms

from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.datasets import make_spd_matrix
import tarfile









class SyntheticProblem:
    """
    Class used to define the following synthetic quadratic bilevel problem:
        min_{x} f_u(x,y) =  a'x + b'y + 0.5*x'H1 y + 0.5 x'H2 x
        s.t. y = argmin_{y} f_l(x,y) = 0.5 y'H3 y - y'H4 x

    In the constrained LL case, the LL problem includes the following constraints: A1x + A2y \le rhs

    Attributes
        x_dim:                        Dimension of the upper-level problem
        y_dim:                        Dimension of the lower-level problem 
        std_dev:                      Standard deviation of the upper-level stochastic gradient estimates 
        ll_std_dev:                   Standard deviation of the lower-level stochastic gradient estimates
        hess_std_dev:                 Standard deviation of the lower-level stochastic Hessian estimates
        seed (int, optional):         The seed used for the experiments (default 42)
    """
    
    # def __init__(self, x_dim, y_dim, std_dev, batch, hess_batch, seed=42):
    def __init__(self, x_dim, y_dim, std_dev, ll_std_dev, hess_std_dev, num_constr, constr_type, pen_param = 1e-1, seed=42):
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.std_dev = std_dev
        self.ll_std_dev = ll_std_dev
        self.hess_std_dev = hess_std_dev
        # self.batch = batch
        # self.hess_batch = hess_batch
        self.num_constr = num_constr
        self.constr_type = constr_type
        self.seed = seed
        
        self.set_seed(self.seed)
        
        self.a = np.random.uniform(0,10,(self.x_dim,1)) #np.random.uniform(0,10,(self.x_dim,1))
        self.b = np.random.uniform(0,10,(self.y_dim,1)) #np.random.uniform(0,10,(self.x_dim,1))
        
        self.H1 = np.eye(self.x_dim,self.y_dim) 
        self.H2 = make_spd_matrix(self.x_dim, random_state=self.seed) 
        self.H3 = make_spd_matrix(self.y_dim, random_state=self.seed) 
        self.H4 = np.eye(self.y_dim,self.x_dim) #ll_H_yx
        
        # Constrained LL case
        if self.constr_type == 'Linear_y':
            # Constraint of the form: Ay <= rhs
            self.rhs = np.random.uniform(0,10,(self.num_constr,1))
            self.A  = np.random.uniform(0,1,(self.num_constr,self.y_dim))
            self.pen_param = pen_param
        elif self.constr_type == 'Linear_xy':
            # Constraint of the form: A1*x + A2*y <= rhs
            self.rhs = np.random.uniform(0,10,(self.num_constr,1))
            self.A1  = np.random.uniform(0,1,(self.num_constr,self.x_dim))
            self.A2  = np.random.uniform(0,1,(self.num_constr,self.y_dim))
            self.pen_param = pen_param
        elif self.constr_type == 'Quadratic':
            # Constraint of the form: y^T*Q1*y + x^T*Q2*y + q1*y + q2*x <= rhs
            self.rhs = np.random.uniform(0,10,(self.num_constr,1))
            self.q1 = np.random.uniform(0,10,(self.num_constr,self.y_dim))
            self.q2 = np.random.uniform(0,10,(self.num_constr,self.x_dim))
            self.Q1_matrices = []
            self.Q2_matrices = []
            for i in range(num_constr):
                self.Q1_matrices.append(0.01*make_spd_matrix(self.y_dim, random_state=self.seed+i))
                self.Q2_matrices.append(np.random.uniform(0,1,(self.x_dim, self.y_dim)))
            self.pen_param = pen_param
      
        
      
        
        
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


    def f(self, x):
        """
        The true objective function of the bilevel problem
    	"""
        y_opt = np.dot(np.linalg.inv(self.H3),np.dot(self.H4,x))
        return self.f_u(x, y_opt)
        
    
    def f_opt(self):
        """
        The optimal value of the bilevel problem
    	"""
        invH3 = np.linalg.inv(self.H3)
        aux = np.dot(self.H1,np.dot(invH3,self.H4)) + 2*self.H2 + np.dot(self.H4,np.dot(invH3.T,self.H1.T))
        x_opt = -2*np.dot(np.linalg.inv(aux),self.a + np.dot(self.H4,np.dot(invH3.T,self.b)))
        return self.f(x_opt)
    
    
    def f_u(self, x, y):
        """
        The upper-level objective function
    	"""
        out = np.dot(self.a.T,x) + np.dot(self.b.T,y) + 0.5*np.dot(x.T,np.dot(self.H1,y)) + 0.5*np.dot(x.T,np.dot(self.H2,x))
        return np.squeeze(out)
    

    def f_l(self, x, y):
        """
        The lower-level objective function
    	"""
        out = 0.5*np.dot(y.T,np.dot(self.H3,y)) - np.dot(y.T,np.dot(self.H4,x)) 
        return np.squeeze(out)


    def grad_fu_ul_vars(self, x, y):
        """
        The gradient of the upper-level objective function wrt the upper-level variables
    	"""
        out = self.a + 0.5*np.dot(self.H1,y) + np.dot(self.H2,x) 
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def grad_fu_ll_vars(self, x, y):
        """
        The gradient of the upper-level objective function wrt the lower-level variables
    	"""
        out = self.b + 0.5*np.dot(self.H1.T,x)
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def grad_fl_ul_vars(self, x, y):
        """
        The gradient of the lower-level objective function wrt the upper-level variables
    	"""
        out = -np.dot(self.H4.T,y)
        out = out + self.ll_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def grad_fl_ll_vars(self, x, y):
        """
        The gradient of the lower-level objective function wrt the lower-level variables
    	"""
        out = np.dot(self.H3,y) - np.dot(self.H4,x)
        out = out + self.ll_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def grad_fl_ll_vars_torch(self, x, y):
        """
        The gradient of the lower-level objective function wrt the lower-level variables
    	"""
        out = torch.matmul(torch.tensor(self.H3, dtype=torch.float64),y) - torch.matmul(torch.tensor(self.H4, dtype=torch.float64),x)
        out = out + self.ll_std_dev*torch.randn(out.shape[0],out.shape[1])
        return out
    

    def hess_fl_ll_vars_ll_vars(self, x, y):
        """
        The Hessian of the lower-level objective function wrt the lower-level variables
    	"""
        out = self.H3.T
        out = out + self.hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def hess_fl_ll_vars_ul_vars(self, x, y):
        """
        The Hessian of the lower-level objective function wrt the lower and upper level variables
    	"""
        out = -self.H4.T
        out = out + self.hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    

    def hess_fl_ul_vars_ll_vars(self, x, y):
        """
        The Hessian of the lower-level objective function wrt the upper and lower level variables
    	"""
        out = -self.H4
        out = out + self.hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    # Constrained LL case
    def inequality_constraint(self, x, y):
        """
        Computes the array of the inequality constraints
        """ 
        if self.constr_type == 'Linear_y':
            return np.matmul(self.A,y) - self.rhs
        elif self.constr_type == 'Linear_xy':
            return np.matmul(self.A1,x) + np.matmul(self.A2,y) - self.rhs
        elif self.constr_type == 'Quadratic':
            constr_val_list = []
            for i in range(self.num_constr):
                val_i = 0.5*np.dot(y.T,np.dot(self.Q1_matrices[i],y)) + np.dot(x.T,np.dot(self.Q2_matrices[i],y)) + np.dot(self.q1[i],y) + np.dot(self.q2[i],x) - self.rhs[i]
                constr_val_list.append(val_i)
            return np.reshape( np.asarray(constr_val_list), (self.num_constr,1))
    
    
    def inequality_constraint_jacob(self, x, y):
        """
        Computes the Jacobian matrices of the inequality constraints
        """
        if self.constr_type == 'Linear_y':
            jacob_x = np.zeros((self.x_dim,self.num_constr))
            jacob_y = self.A.T
            jacob_y = jacob_y + self.ll_std_dev*np.random.randn(jacob_y.shape[0],jacob_y.shape[1])
            return jacob_x, jacob_y 
        elif self.constr_type == 'Linear_xy':
            jacob_x = self.A1.T
            jacob_y = self.A2.T 
            jacob_x = jacob_x + self.ll_std_dev*np.random.randn(jacob_x.shape[0],jacob_x.shape[1])
            jacob_y = jacob_y + self.ll_std_dev*np.random.randn(jacob_y.shape[0],jacob_y.shape[1])
            return jacob_x, jacob_y 
        elif self.constr_type == 'Quadratic':
            jacob_x_list = []
            jacob_y_list = []
            for i in range(self.num_constr):
                jacob_x_list.append(np.dot(self.Q2_matrices[0],y) + np.reshape(self.q2[0], (self.x_dim,1)))
                jacob_y_list.append(np.dot(self.Q1_matrices[i],y) + np.dot(x.T,self.Q2_matrices[i]).T + np.reshape(self.q1[i], (self.y_dim,1)))
            jacob_x = np.reshape( np.asarray(jacob_x_list), (self.x_dim,self.num_constr))
            jacob_y = np.reshape( np.asarray(jacob_y_list).T, (self.y_dim,self.num_constr))
            jacob_x = jacob_x + self.ll_std_dev*np.random.randn(jacob_x.shape[0],jacob_x.shape[1])
            jacob_y = jacob_y + self.ll_std_dev*np.random.randn(jacob_y.shape[0],jacob_y.shape[1])
            return jacob_x, jacob_y
        
        
    def inequality_constraint_hess(self, x, y):
        """
        Computes the Hessian matrices of the inequality constraints wrt x and y
        """
        if self.constr_type == 'Linear_y':
            hess_x = np.zeros((self.x_dim,self.y_dim))
            hess_y = np.zeros((self.y_dim,self.y_dim))
            return hess_x, hess_y
        elif self.constr_type == 'Linear_xy':
            hess_x = np.zeros((self.x_dim,self.y_dim))
            hess_y = np.zeros((self.y_dim,self.y_dim))
            return hess_x, hess_y
        elif self.constr_type == 'Quadratic':
            hess_x = np.zeros((self.x_dim,self.y_dim))
            hess_y_list = []
            noise = self.hess_std_dev*np.random.randn(self.y_dim,self.y_dim)
            for i in range(self.num_constr):
                hess_y_list.append(self.Q1_matrices[i] + (1/self.num_constr)*noise)
            hess_y_list = np.asarray(hess_y_list)
            return hess_x, hess_y_list
    

    def grad_ll_pen_func_ll_vars(self, x, y):
        """
        Subgradient of the lower-level penalty function
        """
        ineq_constr = self.inequality_constraint(x,y)
        _, jacob_y = self.inequality_constraint_jacob(x, y)
        out = self.grad_fl_ll_vars(x, y) + 1/self.pen_param*np.matmul(jacob_y,np.maximum(0,ineq_constr))
        return out


    def ll_pen_func_ll_vars(self, x, y):
        """
        Lower-level penalty function
        """
        ineq_constr = self.inequality_constraint(x,y)
        pen_func = self.f_l(x, y) + 1/self.pen_param*(np.matmul(np.ones((self.num_constr,1)).T,np.maximum(0,ineq_constr)))
        return pen_func
    

    def grad_ll_lagr_func_ll_vars(self, x, y, lagr_mul):
        """
        Gradient of the lower-level Lagrangian function
        """
        grad_llp_ll_vars = self.grad_fl_ll_vars(x, y)
        _, jacob_y = self.inequality_constraint_jacob(x, y)
        return grad_llp_ll_vars + np.matmul(jacob_y,lagr_mul)


    def grad_ll_lagr_func_ul_vars(self, x, y, lagr_mul):
        """
        Gradient of the lower-level Lagrangian function
        """
        grad_llp_ul_vars = self.grad_fl_ul_vars(x, y)
        jacob_x, _ = self.inequality_constraint_jacob(x, y)
        return grad_llp_ul_vars + np.matmul(jacob_x,lagr_mul)


    def hess_ll_lagr_func_ll_vars_ll_vars(self, x, y, lagr_mul):
        """
        Hessian of the lower-level Lagrangian function
        """
        hess_llp_ll_vars = self.hess_fl_ll_vars_ll_vars(x, y)
        _, hess_y_list = self.inequality_constraint_hess(x,y)
        if self.constr_type == 'Linear_y':
            return hess_llp_ll_vars + np.zeros((self.y_dim,self.y_dim))
        elif self.constr_type == 'Linear_xy':
            return hess_llp_ll_vars + np.zeros((self.y_dim,self.y_dim))
        elif self.constr_type == 'Quadratic':
            hess_y = np.zeros((self.y_dim,self.y_dim))
            for i in range(self.num_constr):
                hess_y = hess_y + hess_y_list[i]*lagr_mul[i]
            return hess_llp_ll_vars + hess_y


    def hess_ll_lagr_func_ll_vars_ul_vars(self, x, y, lagr_mul):
        """
        Hessian of the lower-level Lagrangian function
        """
        hess_llp_ul_vars = self.hess_fl_ll_vars_ul_vars(x, y)
        return hess_llp_ul_vars + np.zeros((self.x_dim,self.y_dim))


    def jacob_ll_kkt_ll_vars(self, x, y, lagr_mul):
        """
        Jacobian of the lower-level KKT system wrt lower-level variables
        """
        _, jacob_y = self.inequality_constraint_jacob(x, y)
        out_1 = np.concatenate((self.hess_ll_lagr_func_ll_vars_ll_vars(x, y, lagr_mul), lagr_mul.T*jacob_y), axis=1)
        out_2 = np.concatenate((jacob_y.T,np.diag(np.squeeze(self.inequality_constraint(x, y)))), axis=1)
        out = np.concatenate((out_1,out_2), axis=0)   
        return out


    def jacob_ll_kkt_ul_vars(self, x, y, lagr_mul):
        """
        Jacobian of the lower-level KKT system wrt upper-level variables
        """
        jacob_x, __ = self.inequality_constraint_jacob(x, y)
        out = np.concatenate((self.hess_ll_lagr_func_ll_vars_ul_vars(x, y, lagr_mul).T, lagr_mul.T*jacob_x), axis=1) 
        return out


    def f_constr(self, x, y, lagr_mul):
        """
        The true objective function of the bilevel problem in the LL constrained case
     	"""
        if self.constr_type == 'Linear_y':
            y_opt = np.dot(np.linalg.inv(self.H3),(np.dot(self.H4,x) - np.dot(self.A.T,lagr_mul)))
            return self.f_u(x, y_opt)
        elif self.constr_type == 'Linear_xy':
            y_opt = np.dot(np.linalg.inv(self.H3),(np.dot(self.H4,x) - np.dot(self.A2.T,lagr_mul)))
            return self.f_u(x, y_opt)
        elif self.constr_type == 'Quadratic':
            temp_ll_vars = y
            def temp_f_l(y):
                out = 0.5*np.dot(y.T,np.dot(self.H3,y)) - np.dot(y.T,np.dot(self.H4,x)) 
                return np.squeeze(out)
            def fun(y):
                constr_val_list = []
                for i in range(self.num_constr):
                    val_i = 0.5*np.dot(y.T,np.dot(self.Q1_matrices[i],y)) + np.dot(x.T,np.dot(self.Q2_matrices[i],y)) + np.dot(self.q1[i],y) + np.dot(self.q2[i],x) - self.rhs[i]
                    constr_val_list.append(val_i)
                return np.reshape( np.asarray(constr_val_list), (self.num_constr,))
            cons = NonlinearConstraint(fun, lb = -np.inf, ub = 0)
            sol = minimize(temp_f_l, np.squeeze(temp_ll_vars), method='SLSQP', constraints=cons)
            y_opt = np.asarray([sol.x]).T
            return self.f_u(x, y_opt)
        
    
    def f_opt_constr(self, y, lagr_mul):
        """
        The optimal value of the bilevel problem given the vector of Lagrange multipliers
     	"""
        invH3 = np.linalg.inv(self.H3)
        aux = 0.5*( np.dot(self.H1,np.dot(invH3,self.H4)) + 2*self.H2 + np.dot(self.H4.T,np.dot(invH3.T,self.H1.T)) )
        x_opt = -np.dot(np.linalg.inv(aux),self.a + np.dot(self.H4.T,np.dot(invH3.T,self.b)) - 0.5*np.dot(self.H1,np.dot(invH3.T,np.dot(self.A2.T,lagr_mul))))
        return self.f_constr(x_opt, y, lagr_mul)








class ContinualLearning:
    """
    Class used to define a continual learning problem

    Attributes
        device (str):               Name of the device used, whether GPU or CPU (if GPU is not available)
        file_path (str):            Path to the dataset file 
        download_cifar_bool (bool): A flag to download the dataset CIFAR-10
        seed (int, optional):       The seed used for the experiments (default 42)
        val_pct (real):             Percentage of data used for validation (default 0.2)
        train_batch_size (real):    Minibatch size for training (default 0.01)
        batch_size (real):          Minibatch size for validation (default 0.05)
        training_loaders:           List of 5 increasing training datasets (01, 0123, 012345, etc.)
        testing_loaders:            List of 5 increasing testing datasets (01, 0123, 012345, etc.) 
        training_task_loaders:      List of 5 training datasets, each associated with a task
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def __init__(self, file_path, download_cifar_bool, seed=42, val_pct = 0.2 , train_batch_size = 0.01, batch_size = 0.05):       
        self.file_path = file_path
        self.download_cifar_bool = download_cifar_bool
        self.seed = seed
        self.val_pct = val_pct
        self.train_batch_size = train_batch_size
        self.batch_size = batch_size
        
        data_out = self.load_CIFAR()
        self.training_loaders = data_out[0]  #training_loaders
        self.testing_loaders = data_out[1] #testing_loaders
        self.training_task_loaders = data_out[2] #training_task_loaders
        
        
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
                 
    def download_CIFAR(self):    
        """
        Downloads CIFAR and create folder 'cifar10' within file_path
    	"""
        dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
        download_url(dataset_url, root='.')
		#Extract from archive into the 'data' directory
        with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
            tar.extractall(path=self.file_path)        

    def load_CIFAR(self):
    	"""
        Loads the CIFAR-10 dataset and generates the data loaders    
    	"""
    	self.set_seed(self.seed)

        # Download CIFAR-10
    	if self.download_cifar_bool:
            self.download_CIFAR()          

      	# Load the CIFAR-10 dataset
    	data_dir = self.file_path + '/cifar10' 
    	classes = os.listdir(data_dir + "/train")
    	# Load the data as PyTorch tensors
    	dataset = ImageFolder(data_dir + '/train', transform=ToTensor())
    	
    	# Generate training and validation indices
    	def split_indices(n, val_pct, seed):
    	 		# Determine the size of the validation set
    	 		n_val = int(val_pct*n)
    	         # Set the random seed
    	 		np.random.seed(seed)
    	         # Create random permutation of 0 to n-1
    	 		idxs = np.random.permutation(n)
    	         # Pick first n_val indices for validation
    	 		return idxs[n_val:], idxs[:n_val] 

    	# Generate the set of indices to sample from each set
    	train_indices, val_indices = split_indices(len(dataset), self.val_pct, self.seed) 
    	   	
    	# Load the full train and validation datasets
    	train_sampler = SubsetRandomSampler(train_indices)
    	train_full_loader = DataLoader(dataset, sampler=train_sampler) #50,000 images
    	test_sampler = SubsetRandomSampler(val_indices)
    	test_full_loader = DataLoader(dataset, sampler=test_sampler) #9,999 images (one of the images had an issue)  	


        # if use_MNIST:
    	if False:            
        	#Load the MNIST dataset
        	mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
        	mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
        	# train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
        	# test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=True)
        	#The full train and test datasets
        	train_full_loader = DataLoader(mnist_train, shuffle=True) #60,000 images
        	test_full_loader = DataLoader(mnist_test, shuffle=True) #10,000 images
        
        	#Set up the five training data loaders that contain the respective images: (0,1), (2,3), (4,5), (6,7), (8,9)
        	train_loader_01 = [] #Will add 12,665 images. Total 12,665.
        	train_loader_0123 = [] #Will add 12,089 images. Total 24,754.
        	train_loader_012345 = [] #Will add 11,263 images. Total 36,017.
        	train_loader_01234567 = [] #Will add 12,183 images. Total 48,200.
        	train_loader_0123456789 = [] #Will add 11,800 images. Total 60,000.


    	
    	# Set up the five training data loaders
    	train_loader_01 = [] #Will add 7,958 images. Total 7,958. (airplanes, automobiles)
    	train_loader_0123 = [] #Will add 8,007 images. Total 15,965. (airplanes, automobiles, birds, cats)
    	train_loader_012345 = [] #Will add 7,984 images. Total 23,949. (airplanes, automobiles, birds, cats, deer, dogs)
    	train_loader_01234567 = [] #Will add 8,018 images. Total 31,967. (airplanes, automobiles, birds, cats, deer, dogs, forgs, horses)
    	train_loader_0123456789 = [] #Will add 8,033 images. Total 40,000. (airplanes, automobiles, birds, cats, deer, dogs, forgs, horses, ships, trucks)

    	train_task_loader_01 = [] 
    	train_task_loader_23 = [] 
    	train_task_loader_45 = [] 
    	train_task_loader_67 = [] 
    	train_task_loader_89 = [] 

    	for X,y in train_full_loader:
            if y == 0 or y == 1:
                train_loader_01.append([X,y])
            if y == 0 or y == 1 or y == 2 or y == 3:
                train_loader_0123.append([X,y])
            if y == 0 or y == 1 or y == 2 or y == 3 or y == 4 or y == 5:
                train_loader_012345.append([X,y])
            if y == 0 or y == 1 or y == 2 or y == 3 or y == 4 or y == 5 or y == 6 or y == 7:
                train_loader_01234567.append([X,y])
            train_loader_0123456789.append([X,y])

    	for X,y in train_full_loader:
            if y == 0 or y == 1:
                train_task_loader_01.append([X,y]) 
            if y == 2 or y == 3:
                train_task_loader_23.append([X,y]) 
            if y == 4 or y == 5:
                train_task_loader_45.append([X,y])
            if y == 0 or y == 1 or y == 2 or y == 3 or y == 4 or y == 5 or y == 6 or y == 7:
                train_task_loader_67.append([X,y])
            train_task_loader_89.append([X,y])
    
        # Batch the training loaders
    	loaders = [train_loader_01, train_loader_0123, train_loader_012345, train_loader_01234567, train_loader_0123456789]
    	train_loader_1 = []; train_loader_2 = []; train_loader_3 = []; train_loader_4 = []; train_loader_5 = []
    	for i in range(len(loaders)):
            X = []; y = []
            counter = 0
            for X_y in loaders[i]:
                X.append(X_y[0][0].numpy())
                y.append(X_y[1].numpy()) 
                if len(X) == math.trunc(len(loaders[i]) * self.train_batch_size):
                    if i == 0:
                        train_loader_1.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 1:
                        train_loader_2.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 2:
                        train_loader_3.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 3:
                        train_loader_4.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 4:
                        train_loader_5.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    counter = 0
                    X = []; y = []
                else:
                    counter += 1
    	
       	# Set up the five testing data loaders
    	test_loader_01 = [] #Will add a total of 2,042 images. Total 2,042.
    	test_loader_0123 = [] #Will add a total of 1,993 images. Total 4,035.
    	test_loader_012345 = [] #Will add a total of 2,016 images. Total 6,051.
    	test_loader_01234567 = [] #Will add a total of 1,982 images. Total 8,033.
    	test_loader_0123456789 = [] #Will add a total of 1,966 images. Total 9,999.
        
    	for X,y in test_full_loader:
            if y == 0 or y == 1:
                test_loader_01.append([X,y])
            if y == 0 or y == 1 or y == 2 or y == 3:
                test_loader_0123.append([X,y])
            if y == 0 or y == 1 or y == 2 or y == 3 or y == 4 or y == 5:
                test_loader_012345.append([X,y])
            if y == 0 or y == 1 or y == 2 or y == 3 or y == 4 or y == 5 or y == 6 or y == 7:
                test_loader_01234567.append([X,y])
            test_loader_0123456789.append([X,y])
        
        # Batch the validation data
    	test_loaders = [test_loader_01, test_loader_0123, test_loader_012345, test_loader_01234567, test_loader_0123456789]
    	test_loader_1 = []; test_loader_2 = []; test_loader_3 = []; test_loader_4 = []; test_loader_5 = []
    	for i in range(len(test_loaders)):
            X = []; y = []
            counter = 0
            for X_y in test_loaders[i]:
                X.append(X_y[0][0].numpy())
                y.append(X_y[1].numpy())
                if len(X) == math.trunc(len(test_loaders[i]) * self.batch_size):
                    if i == 0:
                        test_loader_1.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 1:
                        test_loader_2.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 2:
                        test_loader_3.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 3:
                        test_loader_4.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 4:
                        test_loader_5.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    counter = 0
                    X = []; y = []
                else:
                    counter += 1

        # Set up a training data loader for each task  
    	train_task_loader_01 = [] 
    	train_task_loader_23 = [] 
    	train_task_loader_45 = [] 
    	train_task_loader_67 = [] 
    	train_task_loader_89 = []
    
    	for X,y in train_full_loader:
            if y == 0 or y == 1:
                train_task_loader_01.append([X,y]) 
            if y == 2 or y == 3:
                train_task_loader_23.append([X,y]) 
            if y == 4 or y == 5:
                train_task_loader_45.append([X,y])
            if y == 0 or y == 1 or y == 2 or y == 3 or y == 4 or y == 5 or y == 6 or y == 7:
                train_task_loader_67.append([X,y])
            train_task_loader_89.append([X,y])

        # Batch the training data for each task               
    	task_loaders = [train_task_loader_01, train_task_loader_23, train_task_loader_45, train_task_loader_67, train_task_loader_89]
    	train_task_loader_1 = []; train_task_loader_2 = []; train_task_loader_3 = []; train_task_loader_4 = []; train_task_loader_5 = []
    	for i in range(len(task_loaders)):
            X = []; y = []
            counter = 0
            for X_y in task_loaders[i]:
                X.append(X_y[0][0].numpy())
                y.append(X_y[1].numpy()) 
                if len(X) == math.trunc(len(task_loaders[i]) * self.train_batch_size):
                    if i == 0:
                        train_task_loader_1.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 1:
                        train_task_loader_2.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 2:
                        train_task_loader_3.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 3:
                        train_task_loader_4.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 4:
                        train_task_loader_5.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    counter = 0
                    X = []; y = []
                else:
                    counter += 1 

    	training_loaders = [train_loader_1, train_loader_2, train_loader_3, train_loader_4, train_loader_5]
    	testing_loaders = [test_loader_1, test_loader_2, test_loader_3, test_loader_4, test_loader_5]
    	training_task_loaders = [train_task_loader_1, train_task_loader_2, train_task_loader_3, train_task_loader_4, train_task_loader_5]
    	
    	return training_loaders, testing_loaders, training_task_loaders

    class flatten(nn.Module):
        """
        Flattens a Convolutional Deep Neural Network
    	"""
        def forward(self, x):
            return x.view(x.shape[0], -1)

    def generate_model(self):
        """
        Creates a Convolutional Deep Neural Network
    	"""
   	    # Convolutional Deep Neural Network for CIFAR
        model_cnn_robust = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), #This turns the [3,32,32] image into a [32,32,32] image
    	                                     nn.ReLU(),
    	                                     nn.Conv2d(32, 64, kernel_size=3, padding=1), #This turns the [32,32,32] image into a [64,32,32] image
    	                                     nn.ReLU(),
    	                                     nn.MaxPool2d(2, stride=2), #Turns the [64,32,32] image into a [64,16,16] image
    	                                     self.flatten(),
    	                                     nn.Linear(64*16*16, 10)).to(self.device)
        return model_cnn_robust


        # model_cnn_robust = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),
        #                                  nn.ReLU(),
        #                                  nn.Conv2d(32, 64, 3, padding=1),
        #                                  nn.ReLU(),
        #                                  nn.MaxPool2d(2, stride=2), 
        #                                  self.flatten(),
        #                                  nn.Linear(64*14*14, 10)).to(self.device)
        # return model_cnn_robust


    def loss_func(self, in_1, in_2):
     	"""
        Loss function
    	"""
     	func = nn.CrossEntropyLoss().to(self.device)
     	return func(in_1, in_2)










