import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import functions as func
import bilevel_solver as bls
import time






#--------------------------------------------------#
#-------------- Auxiliary Functions  --------------#
#--------------------------------------------------#

def run_experiment(exp_param_dict, num_rep_value=10):
    """
    Auxiliary function to run the experiments
    
    Args:
        exp_param_dict (dictionary):   Dictionary having some of the attributes of the class SyntheticProblem in functions.py as keys 
        num_rep_value (int, optional): Number of runs for each algorithm (default 10)    
    """
    run = bls.BilevelSolverSyntheticProb(prob, algo=exp_param_dict['algo'], ul_lr=exp_param_dict['ul_lr'], \
                                  ll_lr=exp_param_dict['ll_lr'], use_stopping_iter=exp_param_dict['use_stopping_iter'], \
                                  max_iter=exp_param_dict['max_iter'], stopping_time=exp_param_dict['stopping_time'], \
                                  inc_acc=exp_param_dict['inc_acc'], hess=exp_param_dict['hess'], normalize=exp_param_dict['normalize'], constrained=exp_param_dict['constrained'], \
                                  iprint=exp_param_dict['iprint'])
    run_out = run.main_algorithm_avg_ci(num_rep=num_rep_value)
    values_avg = run_out[0]
    values_ci = run_out[1]
    true_func_values_avg = run_out[2]
    true_func_values_ci = run_out[3]
    times = run_out[4]

    # pd.DataFrame(values_avg).to_csv(exp_param_dict['algo_full_name'] + '_values_avg.csv', index=False)
    # pd.DataFrame(values_ci).to_csv(exp_param_dict['algo_full_name'] + '_values_ci.csv', index=False)
    # pd.DataFrame(true_func_values_avg).to_csv(exp_param_dict['algo_full_name'] + '_true_func_values_avg.csv', index=False)
    # pd.DataFrame(true_func_values_ci).to_csv(exp_param_dict['algo_full_name'] + '_true_func_values_ci.csv', index=False)
    # pd.DataFrame(times).to_csv(exp_param_dict['algo_full_name'] + '_times.csv', index=False)       
    
    return run, values_avg, values_ci, true_func_values_avg, true_func_values_ci, times


def get_nparray(file_name):
    """
    Auxiliary function to obtain numpy arrays from csv files
    """
    values_avg = pd.read_csv(file_name)
    values_avg = [item for item_2 in values_avg.values.tolist() for item in item_2]
    values_avg = np.array(values_avg)
    return values_avg








#------------------------------------------------#
#-------------- Define the problem --------------#
#------------------------------------------------#

# Dimension of the upper-level problem
x_dim_val = 50 #50
# Dimension of the lower-level problem 
y_dim_val = 50 #50
# Standard deviation of the stochastic gradient and Hessian estimates
std_dev_val = 0.5                     #0.2                #0.05               #0.1 #0.01 #BSG-H 0.1
ll_std_dev_val = 0.5                  #0.2                #0.001              #0.1  #0.001 
hess_std_dev_val = 0.005                #0.05                  #0.0001             #BSG-H 0.001
# Number of lower-level constraints
num_constr_val = 5 #5
constr_type = 'Quadratic'



prob = func.SyntheticProblem(x_dim=x_dim_val, y_dim=y_dim_val, std_dev=std_dev_val, ll_std_dev=ll_std_dev_val, hess_std_dev=hess_std_dev_val, num_constr=num_constr_val, constr_type=constr_type)


# batch_val = 5 # if std_dev_val > 0, batch_val should be >= 1
# hess_batch_val = 20 # if std_dev_val > 0, hess_batch_val should be >= 0
# prob = func.SyntheticProblem(x_dim=x_dim_val, y_dim=y_dim_val, std_dev=std_dev_val, batch=batch_val, hess_batch=hess_batch_val)





#----------------------------------------------------------------------#
#-------------- Parameters common to all the algorithms  --------------#
#----------------------------------------------------------------------#

# A flag to use the total number of iterations as a stopping criterion
use_stopping_iter = True
# Maximum number of iterations
max_iter = 500    #2000iter if use_stopping_iter == True #5000iter if use_stopping_iter == False
# Maximum running time (in sec) used when use_stopping_iter is False
stopping_time = 200   #500dim: 40 #2000dim: 900 #400
# Number of runs for each algorithm
num_rep_value = 10
# Used to print info on the optimization (default 1): 0 --> no printing; 1 --> at the end of the optimization; 2 --> at each iteration
iprint = 2
# List of colors for the algorithms in the plots
plot_color_list = ['#1f77b4','#bcbd22']
# List of names for the algorithms in the legends of the plots
plot_legend_list = ['BSG-N-FD (inc. acc.)','BSG-H (inc. acc.)']  




#--------------------------------------------------------------------#
#-------------- Parameters specific for each algorithm --------------#
#--------------------------------------------------------------------#

# Create a dictionary with parameters for each experiment
exp_param_dict = {}

# # BSG-N-FD (1 step)
# # exp_param_dict[10] = {'algo': 'bsg', 'algo_full_name': 'bsghincacc', 'ul_lr': 0.001, 'll_lr': 0.004, 'use_stopping_iter': use_stopping_iter, \
# #                       'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': False, 'hess': 'CG-FD', 'normalize': False, 'constrained': False, 'iprint': iprint} 
# #                     ##  Best stepsizes for each value of standard deviation
# #                     ##  dim 'ul_lr': , 'll_lr': , 
    
# # BSG-N-FD (inc. acc.)
# exp_param_dict[5] = {'algo': 'bsg', 'algo_full_name': 'bsghincacc', 'ul_lr': 0.0009, 'll_lr': 0.001, 'use_stopping_iter': use_stopping_iter, \
#                       'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': 'CG-FD', 'normalize': False, 'constrained': False, 'iprint': iprint}  
#                     ##  Best stepsizes for each value of standard deviation
#                     ##  500dim: 'ul_lr': 0.001, 'll_lr': 0.001, 2000, 0-0-005 - 'ul_lr': 0.001, 'll_lr': 0.001, 2000, 1-1-005
#                     ##  2000dim: 'ul_lr': 0.0009, 'll_lr': 0.001, 2000, 1-1-005
    
# # BSG-1 (1 step)
# # exp_param_dict[10] = {'algo': 'bsg', 'algo_full_name': 'bsg', 'ul_lr': 0.001, 'll_lr': 0.002, 'use_stopping_iter': use_stopping_iter, \
# #                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': False, 'hess': False, 'normalize': False, 'constrained': False, 'iprint': iprint}
# #                     ##  Best stepsizes for each value of standard deviation
# #                     ##  dim 'ul_lr': , 'll_lr': , 

# # BSG-1 (inc. acc.)
# exp_param_dict[10] = {'algo': 'bsg', 'algo_full_name': 'bsgincacc', 'ul_lr': 0.0005, 'll_lr': 0.0004, 'use_stopping_iter': use_stopping_iter, \
#                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': False, 'normalize': False, 'constrained': False, 'iprint': iprint}                                                                                                                                                     
#                     ##  Best stepsizes for each value of standard deviation                
#                     ##  500dim: 'ul_lr': 0.0005, 'll_lr': 0.0009, 2000, 0-0-005 - 'ul_lr': 0.0005, 'll_lr': 0.0004, 2000, 1-1-005

# # BSG-H (1 step)
# # exp_param_dict[10] = {'algo': 'bsg', 'algo_full_name': 'bsgh', 'ul_lr': 0.001, 'll_lr': 0.004, 'use_stopping_iter': use_stopping_iter, \
# #                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': False, 'hess': True, 'normalize': False, 'constrained': False, 'iprint': iprint}
# #                     ##  Best stepsizes for each value of standard deviation
# #                     ##  dim 'ul_lr': , 'll_lr': , 
                    
# # BSG-H (inc. acc.)
# exp_param_dict[10] = {'algo': 'bsg', 'algo_full_name': 'bsghincacc', 'ul_lr': 0.001, 'll_lr': 0.005, 'use_stopping_iter': use_stopping_iter, \
#                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': True, 'normalize': False, 'constrained': False, 'iprint': iprint} 
#                     ##  Best stepsizes for each value of standard deviation
#                     ##  500dim: 'ul_lr': 0.001, 'll_lr': 0.001, 2000iter, 0-0-005  - 'ul_lr': 0.0005, 'll_lr': 0.001, 2000iter, 05-05-001       OLD:'ul_lr': 0.001, 'll_lr': 0.005, 2000iter, 1-1-005:OLD             
                    
# # DARTS
# exp_param_dict[10] = {'algo': 'darts', 'algo_full_name': 'darts', 'ul_lr': 0.0009, 'll_lr': 0.0008, 'use_stopping_iter': use_stopping_iter, \
#                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': False, 'hess': False, 'normalize': False, 'constrained': False, 'iprint': iprint} 
#                     ##  Best stepsizes for each value of standard deviation
#                     ##  500dim: 'ul_lr': 0.0009, 'll_lr': 0.003, 2000, 0-0-005  - 'ul_lr': 0.0009, 'll_lr': 0.0008, 2000, 1-1-005                                                                                                                   
    
# # StocBiO (inc. acc.)
# exp_param_dict[5] = {'algo': 'stocbio', 'algo_full_name': 'stocbio', 'ul_lr': 8e-4, 'll_lr': 0.0001, 'use_stopping_iter': use_stopping_iter, \
#                       'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': False, 'normalize': False, 'constrained': False, 'iprint': iprint}                 
#                     ##  Best stepsizes for each value of standard deviation
#                     ##  500dim: 'ul_lr': 0.0009, 'll_lr': 0.001, 2000, 0-0-005 - 'ul_lr': 0.0009, 'll_lr': 0.0005, 2000, 1-1-005
#                     ##  2000dim: 'ul_lr': 0.0008, 'll_lr': 0.0001, 2000, 1-1-005




# ############## Constrained LL case
    
# BSG-N-FD (inc. acc.)
exp_param_dict[0] = {'algo': 'bsg', 'algo_full_name': 'bsgnfd', 'ul_lr': 0.001, 'll_lr': 0.0000001, 'use_stopping_iter': use_stopping_iter, \
                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': 'CG-FD', 'normalize': False, 'constrained': True, 'iprint': iprint}  
                    ## Best stepsizes for each value of standard deviation
                    ## 300dim 30constr: 'ul_lr': 0.001, 'll_lr': 0.0001, 2000iter 0-0-0 - 'ul_lr': 0.000001, 'll_lr': 0.00000005, 2000, 05-05-001

# # BSG-1 (inc. acc.)
# exp_param_dict[0] = {'algo': 'bsg', 'algo_full_name': 'bsghincacc', 'ul_lr': 0.0001, 'll_lr': 0.0000001, 'use_stopping_iter': use_stopping_iter, \
#                       'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': False, 'normalize': False, 'constrained': True, 'iprint': iprint}  
#                     ## Best stepsizes for each value of standard deviation
#                     ## 300dim 30constr: 'ul_lr': 0.00001, 'll_lr': 0.001 2000iter, 0-0-0 - 'ul_lr': , 'll_lr': , 2000, 1-1-005

# BSG-H (inc. acc.)
exp_param_dict[1] = {'algo': 'bsg', 'algo_full_name': 'bsghincacc', 'ul_lr': 0.001, 'll_lr': 0.0000001, 'use_stopping_iter': use_stopping_iter, \
                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': True, 'normalize': False, 'constrained': True, 'iprint': iprint}  
                    ## Best stepsizes for each value of standard deviation
                    ## 300dim 30constr: 'ul_lr': 0.001, 'll_lr': 0.0001, 2000iter 0-0-0  - 'ul_lr': 0.0001, 'll_lr': 0.00001, 2000iter, 05-05-001 

# # SIGD (inc. acc.)
# exp_param_dict[3] = {'algo': 'sigd', 'algo_full_name': 'sigd', 'ul_lr': 0.0000001, 'll_lr': 0.00001, 'use_stopping_iter': use_stopping_iter, \
#                       'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': True, 'normalize': False, 'constrained': True, 'iprint': iprint}  






#--------------------------------------------------------------------#
#-------------- Run the experiments and make the plots --------------#
#--------------------------------------------------------------------#

# Create a dictionary collecting the output for each experiment
exp_out_dict = {}

for i in range(len(exp_param_dict)):
    run, values_avg, values_ci, true_func_values_avg, true_func_values_ci, times = run_experiment(exp_param_dict[i], num_rep_value=num_rep_value)
    exp_out_dict[i] = {'run': run, 'values_avg': values_avg, 'values_ci': values_ci, 'true_func_values_avg': true_func_values_avg, 'true_func_values_ci': true_func_values_ci, 'times': times}


# Make the plots
for i in range(len(exp_out_dict)):
    if exp_out_dict[i]['run'].use_stopping_iter:
        if exp_out_dict[i]['run'].true_func:
            val_x_axis = [i for i in range(len(exp_out_dict[i]['true_func_values_avg']))]
        else:
            val_x_axis = [i for i in range(len(exp_out_dict[i]['values_avg']))]
    else:
        val_x_axis = exp_out_dict[i]['times']
        val_x_axis = [item*10**3 for item in val_x_axis]
    if exp_out_dict[i]['run'].true_func:
        val_y_axis_avg = exp_out_dict[i]['true_func_values_avg'] 
        val_y_axis_ci = exp_out_dict[i]['true_func_values_ci'] 
    else:
        val_y_axis_avg = exp_out_dict[i]['values_avg'] 
        val_y_axis_ci = exp_out_dict[i]['values_ci']        
    string_legend = r'{0} $\alpha^u = {1}$, $\alpha^\ell = {2}$'.format(plot_legend_list[i],exp_param_dict[i]['ul_lr'],exp_param_dict[i]['ll_lr'])
    sns.lineplot(x=val_x_axis, y=val_y_axis_avg, linewidth = 1, label = string_legend, color = plot_color_list[i])
    plt.fill_between(val_x_axis, (val_y_axis_avg-val_y_axis_ci), (val_y_axis_avg+val_y_axis_ci), alpha=.4, linewidth = 0.5, color = plot_color_list[i])   

plt.gca().set_ylim([-1000,5000]) # -1000,5000    #-20000,10000 #50dim -2100,5000 #500dim -50000,5000 #2000dim -200000,200000
# The optimal value of the bilevel problem
# plt.hlines(exp_out_dict[0]['run'].func.f_opt(), 0, val_x_axis[len(val_x_axis)-1], color='red', linestyle='dotted') 

if exp_out_dict[0]['run'].use_stopping_iter:
    plt.xlabel("Iterations", fontsize = 13)
else:
    plt.xlabel("Time (ms)", fontsize = 13)
plt.ylabel("f", fontsize = 13)
plt.tick_params(axis='both', labelsize=11)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

plt.legend(frameon=False) # No borders in the legend
# string = ' standard deviation = ' + str(exp_out_dict[0]['run'].func.std_dev)
string = ' UL grad std dev = ' + str(exp_out_dict[0]['run'].func.std_dev) + ', LL grad std dev = ' + str(exp_out_dict[0]['run'].func.ll_std_dev) + ', \n Hess std dev = ' + str(exp_out_dict[0]['run'].func.hess_std_dev)
plt.title(string)

fig = plt.gcf()
fig.set_size_inches(7, 5.5)  
fig.tight_layout(pad=4.5)

# Uncomment the next line to save the plot
#string = 'Synthetic_problems.png'
#fig.savefig(string, dpi = 100, format='pdf')

