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
    accuracy_values_avg = run_out[5] # November 2024 Update
    accuracy_values_ci = run_out[6] # November 2024 Update
    
    return run, values_avg, values_ci, true_func_values_avg, true_func_values_ci, times, accuracy_values_avg, accuracy_values_ci # November 2024 Update


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
use_stopping_iter = False
# Maximum number of epochs
max_epochs = 1000 #This should be set to 5 when running the plots in terms of time
# List of times (in seconds) used when use_stopping_iter is False to determine when a new task must be added to the problem
stopping_times = [50, 100, 150, 200, 250] #[400, 800, 1200, 1600, 2000] #[50, 100, 150, 200, 250] 
# Number of runs for each algorithm
num_rep_value = 10
# Used to print info on the optimization (default 1): 0 --> no printing; 1 --> at the end of every task; 2 --> at each iteration 
iprint = 2 
# List of colors for the algorithms in the plots
plot_color_list = ['#1f77b4','#2ca02c','#9467bd','#ff7f0e']
# plot_color_list = ['#2ca02c']
# plot_color_list = ['#1f77b4']
# List of names for the algorithms in the legends of the plots
plot_legend_list = ['BSG-N-FD (inc. acc.)','BSG-1 (inc. acc.)','DARTS','StocBiO']  
# plot_legend_list = ['BSG-1 (inc. acc.)']  
# plot_legend_list = ['BSG-N-FD (inc. acc.)']


#--------------------------------------------------------------------#
#-------------- Parameters specific for each algorithm --------------#
#--------------------------------------------------------------------#

# # Create a dictionary with parameters for each experiment
exp_param_dict = {}

## bsg-CG (inc. acc.)
exp_param_dict[0] = {'algo': 'bsg', 'algo_full_name': 'bsgincacc', 'ul_lr': 0.0001, 'll_lr': 0.001, 'use_stopping_iter': use_stopping_iter, \
                      'max_epochs': max_epochs, 'stopping_times': stopping_times, 'inc_acc': True, 'hess': 'CG-FD', 'normalize': False, 'constrained': True, 'iprint': iprint}

## bsg-1 (inc. acc.)
exp_param_dict[1] = {'algo': 'bsg', 'algo_full_name': 'bsgincacc', 'ul_lr': 0.0001, 'll_lr': 0.0005, 'use_stopping_iter': use_stopping_iter, \
                      'max_epochs': max_epochs, 'stopping_times': stopping_times, 'inc_acc': True, 'hess': False, 'normalize': False, 'constrained': True, 'iprint': iprint}



#--------------------------------------------------------------------#
#-------------- Run the experiments and make the plots --------------#
#--------------------------------------------------------------------#

# Create a dictionary collecting the output for each experiment
exp_out_dict = {}

# for i in range(len(exp_param_dict)):
#      if i <= 1:
#          run, values_avg, values_ci, true_func_values_avg, true_func_values_ci, times, accuracy_values_avg, accuracy_values_ci = run_experiment(exp_param_dict[i], num_rep_value=num_rep_value) # November 2024 Update
#          exp_out_dict[i] = {'run': run, 'values_avg': values_avg, 'values_ci': values_ci, 'true_func_values_avg': true_func_values_avg, 'true_func_values_ci': true_func_values_ci, 'times': times, 'accuracy_values_avg': accuracy_values_avg, 'accuracy_values_ci': accuracy_values_ci} # November 2024 Update




## Save results in a pickle file ## November 2024 Update
## Define the filename for the pickle file
filename = 'exp_out_dict_CL_constrained_time_large.pkl'

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
        exp_out_dict[i]['run'] = bls.BilevelSolverCL(prob, algo=exp_param_dict[i]['algo'], ul_lr=exp_param_dict[i]['ul_lr'], \
                                      ll_lr=exp_param_dict[i]['ll_lr'], use_stopping_iter=exp_param_dict[i]['use_stopping_iter'], \
                                      max_epochs=exp_param_dict[i]['max_epochs'], stopping_times=exp_param_dict[i]['stopping_times'], \
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
    if exp_out_dict[i]['run'].true_func:
        val_y_axis_avg = exp_out_dict[i]['true_func_values_avg'] 
        val_y_axis_ci = exp_out_dict[i]['true_func_values_ci']      
    else:
        val_y_axis_avg = exp_out_dict[i]['values_avg'] 
        val_y_axis_ci = exp_out_dict[i]['values_ci']      
    string_legend = r'{0} $\alpha_k^u = {1}/k$, $\alpha^\ell = {2}$'.format(plot_legend_list[i],exp_param_dict[i]['ul_lr'],exp_param_dict[i]['ll_lr'])
    sns.lineplot(x=val_x_axis, y=val_y_axis_avg, linewidth = 1, label = string_legend, color = plot_color_list[i])
    if num_rep_value > 1:
        plt.fill_between(val_x_axis, (val_y_axis_avg-val_y_axis_ci), (val_y_axis_avg+val_y_axis_ci), alpha=.4, linewidth = 0.5, color = plot_color_list[i])

plt.gca().set_ylim([0.2,4.75])

if exp_out_dict[0]['run'].use_stopping_iter:
    plt.xlabel("UL Iterations", fontsize = 13) # November 2024 Update
else:
    plt.xlabel("Time (s)", fontsize = 13)
plt.ylabel("Validation Loss", fontsize = 13) # November 2024 Update
plt.tick_params(axis='both', labelsize=11)

plt.legend(frameon=False) #no borders in the legend


fig = plt.gcf()
fig.set_size_inches(11, 5.5)  
fig.tight_layout(pad=4.5)

## Uncomment the next line to save the plot
string = 'CL_constrained_time_large.pdf'
fig.savefig(string, dpi = 100, format='pdf')




plt.figure()  # Start a new figure

# Make the plots for accuracy # November 2024 Update
for i in range(len(exp_out_dict)):
    if exp_out_dict[i]['run'].use_stopping_iter:
        val_x_axis = [i for i in range(len(exp_out_dict[i]['values_avg']))]
    else:
        val_x_axis = exp_out_dict[i]['times']
    val_y_axis_avg = exp_out_dict[i]['accuracy_values_avg'] 
    val_y_axis_ci = exp_out_dict[i]['accuracy_values_ci']     
    string_legend = r'{0} $\alpha_k^u = {1}/k$, $\alpha^\ell = {2}$'.format(plot_legend_list[i],exp_param_dict[i]['ul_lr'],exp_param_dict[i]['ll_lr'])
    sns.lineplot(x=val_x_axis, y=val_y_axis_avg, linewidth = 1, label = string_legend, color = plot_color_list[i])
    if num_rep_value > 1:
        plt.fill_between(val_x_axis, (val_y_axis_avg-val_y_axis_ci), (val_y_axis_avg+val_y_axis_ci), alpha=.4, linewidth = 0.5, color = plot_color_list[i])

plt.gca().set_ylim([0.,1])

if exp_out_dict[0]['run'].use_stopping_iter:
    plt.xlabel("UL Iterations", fontsize = 13) 
else:
    plt.xlabel("Time (s)", fontsize = 13)
plt.ylabel("Validation Accuracy", fontsize = 13) 
plt.tick_params(axis='both', labelsize=11)

plt.legend(frameon=False) #no borders in the legend


fig = plt.gcf()
fig.set_size_inches(11, 5.5)  
fig.tight_layout(pad=4.5)

## Uncomment the next line to save the plot
string = 'CL_constrained_time_large_accuracy.pdf'
fig.savefig(string, dpi = 100, format='pdf')

