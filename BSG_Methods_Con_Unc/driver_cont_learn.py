import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import functions as func
import bilevel_solver as bls





#--------------------------------------------------#
#-------------- Auxiliary Functions  --------------#
#--------------------------------------------------#

def run_experiment(exp_param_dict,num_rep_value=1):
    """
    Auxiliary function to run the experiments
    
    Args:
        exp_param_dict (dictionary):   Dictionary having some of the attributes of the class ContinualLearning in functions.py as keys 
        num_rep_value (int, optional): Number of runs for each algorithm (default 10)      
    """
    run = bls.BilevelSolverCL(prob, algo=exp_param_dict['algo'], ul_lr=exp_param_dict['ul_lr'], \
                                  ll_lr=exp_param_dict['ll_lr'], use_stopping_iter=exp_param_dict['use_stopping_iter'], \
                                  max_epochs=exp_param_dict['max_epochs'], stopping_times=exp_param_dict['stopping_times'], \
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

# Path to the dataset file
file_path_val = "./data"
# A flag to download the dataset CIFAR-10 (Only need to do this once)
download_cifar_bool_val = False

# The next line can be commented if the problem has already been defined
prob = func.ContinualLearning(file_path=file_path_val, download_cifar_bool=download_cifar_bool_val)





#----------------------------------------------------------------------#
#-------------- Parameters common to all the algorithms  --------------#
#----------------------------------------------------------------------#

# A flag to use the total number of iterations as a stopping criterion
use_stopping_iter = True
# Maximum number of epochs
max_epochs = 1 #This should be set to 1000 when running the plots in terms of time. Otherwise 1
# List of times (in seconds) used when use_stopping_iter is False to determine when a new task must be added to the problem
stopping_times = [20, 40, 60, 80, 100] #[200, 400, 600, 800, 1000] constrained #[20, 40, 60, 80, 100] unconstrained
# Number of runs for each algorithm
num_rep_value = 10
# Used to print info on the optimization (default 1): 0 --> no printing; 1 --> at the end of every task; 2 --> at each iteration 
iprint = 2 

# List of colors for the algorithms in the plots
plot_color_list = ['#ff7f0e','#CD5C5C','#2ca02c','#1f77b4']
# List of names for the algorithms in the legends of the plots
plot_legend_list = ['StocBiO (inc. acc.)','DARTS','BSG-1 (inc. acc.)','BSG-N-FD (inc. acc.)']  





#--------------------------------------------------------------------#
#-------------- Parameters specific for each algorithm --------------#
#--------------------------------------------------------------------#

# Create a dictionary with parameters for each experiment
exp_param_dict = {}

# bsg-CG (inc. acc.)
exp_param_dict[3] = {'algo': 'bsg', 'algo_full_name': 'bsgincacc', 'ul_lr': 0.00000001, 'll_lr': 0.05, 'use_stopping_iter': use_stopping_iter, \
                      'max_epochs': max_epochs, 'stopping_times': stopping_times, 'inc_acc': True, 'hess': 'CG-FD', 'normalize': False, 'constrained': False, 'iprint': iprint}
# 'ul_lr': 0.001, 'll_lr': 0.005 unconstrained (best)
# 'ul_lr': 0.0001, 'll_lr': 0.005 unconstrained

# bsg-1 (inc. acc.)
exp_param_dict[2] = {'algo': 'bsg', 'algo_full_name': 'bsgincacc', 'ul_lr': 0.001, 'll_lr': 0.05, 'use_stopping_iter': use_stopping_iter, \
                      'max_epochs': max_epochs, 'stopping_times': stopping_times, 'inc_acc': True, 'hess': False, 'normalize': False, 'constrained': False, 'iprint': iprint}
# 'ul_lr': 0.001, 'll_lr': 0.05 unconstrained

# darts
exp_param_dict[1] = {'algo': 'darts', 'algo_full_name': 'darts', 'ul_lr': 0.001, 'll_lr': 0.5, 'use_stopping_iter': use_stopping_iter, \
                      'max_epochs': max_epochs, 'stopping_times': stopping_times, 'inc_acc': False, 'hess': False, 'normalize': False, 'constrained': False, 'iprint': iprint}
# 'ul_lr': 0.001, 'll_lr': 1 unconstrained

# StocBiO (inc. acc.)
exp_param_dict[0] = {'algo': 'stocbio', 'algo_full_name': 'bsgincacc', 'ul_lr': 0.00000001, 'll_lr': 0.05, 'use_stopping_iter': use_stopping_iter, \
                      'max_epochs': max_epochs, 'stopping_times': stopping_times, 'inc_acc': True, 'hess': False, 'normalize': False, 'constrained': False, 'iprint': iprint}
# 'ul_lr': 0.00001, 'll_lr': 0.005 unconstrained














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
    if exp_out_dict[i]['run'].true_func:
        val_y_axis_avg = exp_out_dict[i]['true_func_values_avg'] #get_nparray(exp_param_dict['algo_full_name'] + '_true_func_values_avg.csv')
        val_y_axis_ci = exp_out_dict[i]['true_func_values_ci'] #get_nparray(exp_param_dict['algo_full_name'] + '_true_func_values_ci.csv')
    else:
        val_y_axis_avg = exp_out_dict[i]['values_avg'] #get_nparray(exp_param_dict['algo_full_name'] + '_values_avg.csv')
        val_y_axis_ci = exp_out_dict[i]['values_ci'] #get_nparray(exp_param_dict['algo_full_name'] + '_values_ci.csv')        
    string_legend = r'{0} $\alpha_k^u = {1}/k$, $\alpha^\ell = {2}$'.format(plot_legend_list[i],exp_param_dict[i]['ul_lr'],exp_param_dict[i]['ll_lr'])
    sns.lineplot(x=val_x_axis, y=val_y_axis_avg, linewidth = 1, label = string_legend, color = plot_color_list[i])
    if num_rep_value > 1:
        plt.fill_between(val_x_axis, (val_y_axis_avg-val_y_axis_ci), (val_y_axis_avg+val_y_axis_ci), alpha=.4, linewidth = 0.5, color = plot_color_list[i])

plt.gca().set_ylim([0.25,2.5])

if exp_out_dict[0]['run'].use_stopping_iter:
    plt.xlabel("Iterations", fontsize = 13)
else:
    plt.xlabel("Time (s)", fontsize = 13)
plt.ylabel("Validation Error", fontsize = 13)
plt.tick_params(axis='both', labelsize=11)

plt.legend(frameon=False) #no borders in the legend

# fig = plt.gcf()
# fig.set_size_inches(7, 5.5)  
# fig.tight_layout(pad=4.5)
fig = plt.gcf()
fig.set_size_inches(11, 5.5)  
fig.tight_layout(pad=4.5)

# Save the plot
# string = 'ContinualLearning.png'
# fig.savefig(string, dpi = 100, format='pdf')




