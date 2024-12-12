import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import functions as func
import bilevel_solver as bls
import pickle # November 2024 Update
import copy # November 2024 Update



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
    run = bls.BilevelSolverSyntheticProb(prob, algo=exp_param_dict['algo'], algo_full_name=exp_param_dict['algo_full_name'], ul_lr=exp_param_dict['ul_lr'], \
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
x_dim_val = 300 
# Dimension of the lower-level problem 
y_dim_val = 300 
# Standard deviation of the stochastic gradient and Hessian estimates
std_dev_val = 0                     
ll_std_dev_val = 0                  
hess_std_dev_val = 0                
# Number of lower-level constraints
num_constr_val = 50 
constr_type = 'Linear_y'



prob = func.SyntheticProblem(x_dim=x_dim_val, y_dim=y_dim_val, std_dev=std_dev_val, ll_std_dev=ll_std_dev_val, hess_std_dev=hess_std_dev_val, num_constr=num_constr_val, constr_type=constr_type)



#----------------------------------------------------------------------#
#-------------- Parameters common to all the algorithms  --------------#
#----------------------------------------------------------------------#

# A flag to use the total number of iterations as a stopping criterion
use_stopping_iter = False
# Maximum number of iterations
max_iter = 5000 #1000 iter if use_stopping_iter == True #5000iter if use_stopping_iter == False
# Maximum running time (in sec) used when use_stopping_iter is False
stopping_time = 100 #100   
# Number of runs for each algorithm
num_rep_value = 10
# Used to print info on the optimization (default 1): 0 --> no printing; 1 --> at the end of the optimization; 2 --> at each iteration
iprint = 2
# List of colors for the algorithms in the plots
plot_color_list = ['#1f77b4','#bcbd22','#9467bd','#ff7f0e']
# List of names for the algorithms in the legends of the plots
plot_legend_list = ['BSG-N-FD (inc. acc.)','BSG-H (inc. acc.)','SIGD (inc. acc.)']  



#--------------------------------------------------------------------#
#-------------- Parameters specific for each algorithm --------------#
#--------------------------------------------------------------------#

# Create a dictionary with parameters for each experiment
exp_param_dict = {}
    
# BSG-N-FD (inc. acc.)
exp_param_dict[0] = {'algo': 'bsg', 'algo_full_name': 'bsgnfd', 'ul_lr': 0.001, 'll_lr': 0.0001, 'use_stopping_iter': use_stopping_iter, \
                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': 'CG-FD', 'normalize': False, 'constrained': True, 'iprint': iprint}  

# BSG-H (inc. acc.)
exp_param_dict[1] = {'algo': 'bsg', 'algo_full_name': 'bsghincacc', 'ul_lr': 0.001, 'll_lr': 0.0001, 'use_stopping_iter': use_stopping_iter, \
                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': True, 'normalize': False, 'constrained': True, 'iprint': iprint}  

# SIGD (inc. acc.)
exp_param_dict[2] = {'algo': 'sigd', 'algo_full_name': 'sigd', 'ul_lr': 0.001, 'll_lr': 0.001, 'use_stopping_iter': use_stopping_iter, \
                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': True, 'normalize': False, 'constrained': True, 'iprint': iprint}  


    
#--------------------------------------------------------------------#
#-------------- Run the experiments and make the plots --------------#
#--------------------------------------------------------------------#

# Create a dictionary collecting the output for each experiment
exp_out_dict = {}

for i in range(len(exp_param_dict)):
    run, values_avg, values_ci, true_func_values_avg, true_func_values_ci, times = run_experiment(exp_param_dict[i], num_rep_value=num_rep_value)
    exp_out_dict[i] = {'run': run, 'values_avg': values_avg, 'values_ci': values_ci, 'true_func_values_avg': true_func_values_avg, 'true_func_values_ci': true_func_values_ci, 'times': times}



## Save results in a pickle file ## November 2024 Update
## Define the filename for the pickle file
filename = 'exp_out_dict_linear_y_deterministic_time.pkl'

write_read = 1 ## 1 to write results, 0 to read results

if write_read:
    ## Create a deep copy of exp_out_dict
    exp_out_dict_copy = copy.deepcopy(exp_out_dict)
    
    ## Remove the 'run' key from each sub-dictionary in the copy
    for i in exp_out_dict_copy:
        if 'run' in exp_out_dict_copy[i]:
            del exp_out_dict_copy[i]['run']
    
    ## Save exp_out_dict to a pickle file
    with open(filename, 'wb') as file:
        pickle.dump(exp_out_dict_copy, file)

else:
    ## Load exp_out_dict from the pickle file
    with open(filename, 'rb') as file:
        exp_out_dict = pickle.load(file)
    
    ## Add 'run' back with a placeholder value or recomputed value
    for i in exp_out_dict:
        exp_out_dict[i]['run'] = bls.BilevelSolverSyntheticProb(prob, algo=exp_param_dict[i]['algo'], algo_full_name=exp_param_dict[i]['algo_full_name'], ul_lr=exp_param_dict[i]['ul_lr'], \
                                      ll_lr=exp_param_dict[i]['ll_lr'], use_stopping_iter=exp_param_dict[i]['use_stopping_iter'], \
                                      max_iter=exp_param_dict[i]['max_iter'], stopping_time=exp_param_dict[i]['stopping_time'], \
                                      inc_acc=exp_param_dict[i]['inc_acc'], hess=exp_param_dict[i]['hess'], normalize=exp_param_dict[i]['normalize'], constrained=exp_param_dict[i]['constrained'], \
                                      iprint=exp_param_dict[i]['iprint'])



plt.figure()  # Start a new figure


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

# plt.gca().set_ylim([-40000,40000]) 
plt.gca().set_ylim([-15000,40000])

if exp_out_dict[0]['run'].use_stopping_iter:
    plt.xlabel("UL Iterations", fontsize = 13) # November 2024 Update
else:
    plt.xlabel("Time (ms)", fontsize = 13)
plt.ylabel("f", fontsize = 13)
plt.tick_params(axis='both', labelsize=11)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

plt.legend(frameon=False) # No borders in the legend

string = ' UL grad std dev = ' + str(exp_out_dict[0]['run'].func.std_dev) + ', LL grad std dev = ' + str(exp_out_dict[0]['run'].func.ll_std_dev) + ',\n Hess std dev = ' + str(exp_out_dict[0]['run'].func.hess_std_dev)
plt.title(string)


fig = plt.gcf()
fig.set_size_inches(7, 5.5)  
fig.tight_layout(pad=4.5)

## Uncomment the next line to save the plot
string = 'linear_y_deterministic_time.pdf'
fig.savefig(string, dpi = 100, format='pdf')

