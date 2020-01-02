# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 09:03:28 2019

@author: yimen
"""
#%%
import numpy as np
#import scipy.sparse as sps
import pandas as pd
import ipopt

#%% Globals 
dir_proj = 'C:/Git/taxdata_weighting_python/' # (change as needed)

includes_dir   = dir_proj + 'code/'
input_data_dir = dir_proj + 'input_files/'
log_dir        = dir_proj + 'logfiles/'   # for ipopt log files
interim_results_dir = dir_proj + 'interim_results/'
outfiles_dir   = dir_proj + 'output_files/'


#%% getting data

targets_state = pd.read_csv(input_data_dir + 'targets_state.csv')
targets_state # note that target_num uniquely identifies each target - unique combination of stabbr, constraint_name, constraint_type

# get national puf using Boyd 2017 grow factors, with selected 2017 Tax-Calculator output variables, and wtus_2017 weight
pufbase_all = pd.read_csv(input_data_dir + 'puf2017_weighted.csv')
pufbase_all 

ratio_adjusted_weights_state = pd.read_csv(input_data_dir + 
                                           'ratio_adjusted_weights_state.csv')
ratio_adjusted_weights_state


#%% Prepare PUF base file - with initial state weight and essential variables 

vars_select = ['RECID', 'MARS'] + targets_state.puf_link.unique().tolist()
vars_select = [elem for elem in vars_select if elem not in ['MARS4', 'MARS1', 'MARS3', 'MARS2']]
pufbase_state = pufbase_all.loc[:, vars_select]
pufbase_state = pufbase_state.merge(ratio_adjusted_weights_state.loc[:,['RECID', 'AGI_STUB', 'stabbr', 'weight_state']], 
                                    how = 'left', on  = 'RECID')

pufbase_state['weight_initial'] = pufbase_state['weight_state'] # we need to keep a copy of the state weight

# create a column for each MARS category that has the initial state weight if the record is of that category
#  # so that we have columns that match up against targets for number of returns by MARS category
pufbase_state['MARS1'] = np.where(pufbase_state['MARS'] == 1, pufbase_state['weight_state'], 0)
pufbase_state['MARS2'] = np.where(pufbase_state['MARS'] == 2, pufbase_state['weight_state'], 0)
pufbase_state['MARS3'] = np.where(pufbase_state['MARS'] == 3, pufbase_state['weight_state'], 0)
pufbase_state['MARS4'] = np.where(pufbase_state['MARS'] == 4, pufbase_state['weight_state'], 0)
pufbase_state = pufbase_state.drop(columns = ['MARS', 'weight_state'])
# better way to implement pivot_wider in R?

pufbase_state.columns

pufbase_state.to_csv(interim_results_dir + "pufbase_state.csv") # TODO: save the object instead
# pufbase_state = pd.read_csv(interim_results_dir + "pufbase_state.csv")


#%% Prepare nonzero constraint coefficients



# make a long sparse file that has the initial weight and the variable value for each numeric variable
# we will make constraint coefficients from this
long = pd.melt(pufbase_state, id_vars = ['RECID', 'stabbr', 'AGI_STUB', 'weight_initial'], 
               var_name = 'puf_link', value_name = 'value')
long = long[long.value != 0]
long

long.AGI_STUB.value_counts()
long.puf_link.value_counts()


# full join against the targets file allowing multiple targets (typically 2) per variable -- # of nonzeros, and amount
long_full = long.merge(targets_state, how = 'outer', on = ['stabbr', 'AGI_STUB', 'puf_link'])
long_full = long_full[long_full.RECID.notnull()]
long_full
# example:
#   we won't have any returns where variable c05800 tax liability is nonzero and the AGI_STUB==1 (AGI < $1)
#   even though the historical table 2 data have about 910 such returns, with total taxbc of ~$31m

# keep track of the good constraints - those that are found in long_full
good_con = long_full.target_num.unique()
good_con

targets_state[targets_state.target_num.isin(good_con)]
targets_state[~targets_state.target_num.isin(good_con)]
# not surprisingly, we lost a few constraints in the lowest AGI ranges, where no PUF records have nonzero values for these

targets_state_good = targets_state[targets_state.target_num.isin(good_con)]
# this is what we will use for targeting and for constraint coefficients

targets_state_good.constraint_type.value_counts() # the types of constraint coefficients we will need

# now we can create a data frame of nonzero constraint coefficients

nzcc = targets_state_good.merge(long_full.drop(columns = ['puf_link', 'year', 'stabbr', 'AGI_STUB', 'table_desc', 'target', 
                                                          'constraint_name', 'constraint_type', 'constraint_base_name']), 
                                how = 'left', on = 'target_num')

nzcc['nzcc'] = np.where(nzcc['constraint_type'] == 'amount',    nzcc['value'] * nzcc['weight_initial'], 0)
nzcc['nzcc'] = np.where(nzcc['constraint_type'] == 'n_nonzero', nzcc['weight_initial'], nzcc['nzcc'])
nzcc['nzcc'] = np.where(nzcc['constraint_type'] == 'n_exempt',  nzcc['value'] * nzcc['weight_initial'], nzcc['nzcc'])  # number of exemptions times record weight
nzcc['nzcc'] = np.where(nzcc['constraint_type'] == 'n_returns', nzcc['weight_initial'], nzcc['nzcc'])

(nzcc.nzcc == 0).sum()
nzcc.columns # we've kept a lot of unnecessary variables but some are nice to have


#%% set tolerances so that we can calculate constraint bounds ####

# compute the starting point (the value on the file using weight_initial) for each target and use it to set constraint bounds

starting_point = nzcc.groupby(['AGI_STUB', 'target_num']).agg(
        constraint_name = ('constraint_name', 'first'),
        constraint_type = ('constraint_type', 'first'),
        table_desc      = ('table_desc',      'first'),
        target = ('target', 'first'),
        file   = ('nzcc',   'sum')
        )

starting_point['diff'] = starting_point['file'] - starting_point['target']
starting_point['pdiff'] = 100 * starting_point['diff'] / starting_point['target']

starting_point.reset_index(level = ['AGI_STUB', 'target_num'], inplace = True)
# https://stackoverflow.com/questions/20461165/how-to-convert-index-of-a-pandas-dataframe-into-a-column

starting_point
starting_point[starting_point.AGI_STUB == 2]


# create a few priority levels
priority1 = ["A00100", "A00200", "A05800", "A09600", "A18500", "N2", "MARS1", "MARS2", "MARS4"]
priority2 = ["N00100", "N00200", "N05800", "N09600", "N18500"]

tolerances = starting_point
tolerances['tol_default'] = np.where(tolerances.constraint_name.isin(priority1 + priority2), 0.005, abs(tolerances.pdiff/100) * 0.1)

tolerances
tolerances[tolerances.AGI_STUB == 2]




#%% Define the class for ipopt problem objects

class puf_obj(object):
    def __init__(self, inputs):
        self.inputs = inputs
 
        
    def objective(self, x):
        # objective function - evaluates to a single number
        
        # ipoptr requires that ALL functions receive the same arguments, so the inputs list is passed to ALL functions
        # here are the objective function, the 1st deriv, and the 2nd deriv
        # http://www.derivative-calculator.net/
        # w{x^p + x^(-p) - 2}                                 objective function
        # w{px^(p-1) - px^(-p-1)}                             first deriv
        # p*w*x^(-p-2)*((p-1)*x^(2*p)+p+1)                    second deriv
  
        # make it easier to read:
        p = self.inputs['p']
        w = self.inputs['wt'] / self.inputs['objscale']
        # x = x.astype(float)
        
        return ((x.astype(float)**p + x.astype(float)**(-p) - 2) * w).sum()


    def gradient(self, x):
        # gradient of objective function - a vector length x 
        # giving the partial derivatives of obj wrt each x[i]
    
        # http://www.derivative-calculator.net/
        # w{x^p + x^(-p) - 2}                                 objective function
        # w{px^(p-1) - px^(-p-1)}                             first deriv
        # p*w*x^(-p-2)*((p-1)*x^(2*p)+p+1)                    second deriv

        # make it easier to read:
        p = self.inputs['p']
        w = self.inputs['wt'] / self.inputs['objscale']
        # x = x.astype(float)
        
        return w * (p * x.astype(float)**(p-1) - p * x.astype(float)**(-p-1))
    
    
    def hessian(self, x, lagrange, obj_factor):
        # The Hessian matrix has many zero elements and so we set it up as a sparse matrix
        # We only keep the (potentially) non-zero values that run along the diagonal.
  
        # http://www.derivative-calculator.net/
        # w{x^p + x^(-p) - 2}                                 objective function
        # w{px^(p-1) - px^(-p-1)}                             first deriv
        # p*w*x^(-p-2)*((p-1)*x^(2*p)+p+1)                    second deriv
  
        # make it easier to read:

        p = self.inputs['p']
        w = self.inputs['wt'] / self.inputs['objscale']
        # x = x.astype(float)
        
        return obj_factor * (p*w*x.astype(float)**(-p-2) * ((p-1)*x.astype(float)**(2*p)+p+1))


    def hessianstructure(self):
         # The structure of the Hessian
         # In this problem, only the diagnal elements have non-zero values
         
        hs_col = np.arange(self.inputs['n_variables'])
        hs_row = np.arange(self.inputs['n_variables'])
         
        # return (hs_col, hs_row)
        return (hs_row, hs_col)
    
    
    def constraints(self, x):
        # constraints that must hold in the solution - just give the LHS of the expression
        # return a vector where each element evaluates a constraint (i.e., sum of (x * a ccmat column), for each column)
        
        df_eval_c = self.inputs['constraint_coefficients_sparse'].loc[:, ['nzcc', 'i', 'j']]
        df_eval_c['x_eval'] = np.take(x, self.inputs['constraint_coefficients_sparse'].j)
        df_eval_c['nzcc_x'] = df_eval_c['nzcc'] * df_eval_c['x_eval']
        
        c_vals = df_eval_c.groupby('i')['nzcc_x'].agg('sum')
        
        return c_vals
        
    
    def jacobian(self, x):    
        # the Jacobian is the matrix of first partial derivatives of constraints (these derivatives may be constants)
        # this function evaluates the Jacobian at point x
        
        return self.inputs['constraint_coefficients_sparse'].nzcc
        
     
    def jacobianstructure(self):
        # The structure of the Hessian
        js_col = self.inputs['constraint_coefficients_sparse'].j
        js_row = self.inputs['constraint_coefficients_sparse'].i
       
        return js_row, js_col
        #return (js_row, js_col)

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):
        pass
        # print("Objective value at iteration %d is - %g" % (iter_count, obj_value))



#%% Define the function for running the optimization on a single AGI stub

def runstub(AGI_STUB, tolerances, nzcc, pufbase_state, log_dir, interim_results_dir):

    stub = AGI_STUB

    constraint_value = 1000  # each constraint will be this number when scaled
    constraints_unscaled = tolerances[tolerances.AGI_STUB == stub].target
    constraint_scales = np.where(constraints_unscaled == 0, 1, abs(constraints_unscaled) / constraint_value)
    constraints = constraints_unscaled / constraint_scales
    # constraints

    # create nzcc for the stub, and on each record create i to index constraints and j to index variables (the x elements)

    nzcc_stub = nzcc[nzcc.AGI_STUB == stub].sort_values(by = ['constraint_name', 'RECID'])

    # NOTE!!: create i and j, each of which will be consecutive integersm where
        #   i gives the index for constraints
        #   j gives the index for the RECID (for the variables)
        #   TODO: There should be better ways to do this in python (liek grouop_indeces in R)
        #   Note that the indeces start from 0 here in python. (start from 1 in R)
    #nzcc_stub
    nzcc_stub['i'] = nzcc_stub.constraint_name
    nzcc_stub['j'] = nzcc_stub.RECID
    nzcc_stub.set_index(['i', 'j'], inplace = True)

    rename_dic1 = dict(zip(nzcc_stub.constraint_name.unique(), list(range(len(nzcc_stub.constraint_name.unique())))))
    rename_dic2 = dict(zip(np.sort(nzcc_stub.RECID.unique()), list(range(len(nzcc_stub.RECID.unique())))))

    nzcc_stub.rename(index = rename_dic1, level = 'i', inplace = True)
    nzcc_stub.rename(index = rename_dic2, level = 'j', inplace = True)
    nzcc_stub.reset_index(inplace = True)
    # nzcc_stub[nzcc_stub.i == 1].loc[:, ['i', 'j', 'constraint_name', 'RECID']]

    nzcc_stub['nzcc_unscaled'] = nzcc_stub.nzcc
    nzcc_stub['nzcc'] = nzcc_stub.nzcc_unscaled / np.take(constraint_scales, nzcc_stub.i)


    # Inputs

    inputs = {}
    inputs['p'] = 2
    inputs['wt'] = pufbase_state[pufbase_state.AGI_STUB == stub].sort_values(by = 'RECID').weight_initial # note this is a pd.Series
    inputs['RECID'] = pufbase_state[pufbase_state.AGI_STUB == stub].sort_values(by = 'RECID').RECID
    inputs['constraint_coefficients_sparse'] = nzcc_stub
    inputs['n_variables'] = len(inputs['wt'])
    inputs['n_constraints'] = len(constraints)
    inputs['objscale'] = 1e6 # scaling constant used in the various objective function functions
    inputs['constraint_scales'] = constraint_scales

    xlb = np.repeat(0,   inputs['n_variables']) # arbitrary
    xub = np.repeat(100, inputs['n_variables']) # arbitrary
    x0  = np.repeat(1,   inputs['n_variables'])

    tol = tolerances[tolerances.AGI_STUB == stub].tol_default

    clb = constraints - abs(constraints) * tol
    cub = constraints + abs(constraints) * tol

    clb.fillna(0, inplace = True)
    cub.fillna(0, inplace = True)


    nlp_puf = ipopt.problem(
            n=len(x0),
            m=len(clb),
            problem_obj=puf_obj(inputs),
            lb=xlb,
            ub=xub,
            cl=clb,
            cu=cub
            )


    logfile_name = log_dir + "stub_" + str(stub) + ".out"

    nlp_puf.addOption('print_level', 0)
    nlp_puf.addOption('file_print_level', 5)
    # nlp_puf.addOption('linear_solver', 'ma27') # cyipopt uses MUMPS solver as its default solver
    nlp_puf.addOption('max_iter', 100)
    nlp_puf.addOption('mu_strategy', 'adaptive')
    nlp_puf.addOption('output_file', logfile_name)


    x, info = nlp_puf.solve(x0)
    print(info['status_msg'])

    return pd.DataFrame({'AGI_STUB' : stub, 'RECID': inputs['RECID'], 'wt_int' : inputs['wt'], 'x' : x})

# Test the function
# xdf1 = runstub(1, tolerances, nzcc, pufbase_state, log_dir, interim_results_dir)

#%% run the optimization on one or more agi stubs

stubs = np.sort(pufbase_state.AGI_STUB.unique())

x_list = list()
for stub in stubs:
    x_list.append(runstub(stub, tolerances, nzcc, pufbase_state, log_dir, interim_results_dir))

xdf = pd.concat(x_list)

xdf
xdf.AGI_STUB.value_counts()

xdf.to_csv(outfiles_dir + "x_state.csv")


#%% Create the final file

pufbase_state = pd.read_csv(interim_results_dir + "pufbase_state.csv")
xdf = pd.read_csv(outfiles_dir + "x_state.csv")

pufbase_state = pufbase_state.merge(xdf[['RECID', 'x']], how = 'left', on = 'RECID')
pufbase_state['weight_state'] = pufbase_state['weight_initial'] * pufbase_state['x']

pufbase_state['weight_initial'].sum()
pufbase_state['weight_state'].sum()

(pufbase_state['weight_state'] * pufbase_state['E00200']).sum() / 1e9
# $ 550.8483 billion is the number that I (Yin) get, which is very close to the value given by the R code (550.8716)
# and within 0.2% of the Historical Table 2 target



