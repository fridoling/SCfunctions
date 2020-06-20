import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SloppyCell.ReactionNetworks import *


def fit_exps(exps, nets, params_fixed, params_constrained, params_free, global_fit = True, exp_ids=None, local_it=20, global_it=10000, return_ens=True):
    """ 
    Fit a set of experiments in a SloppyCell model.

    This function generates a SloppyCell model from a set of experiments
    and nets and performs local and global optimization given sets of fixed,
    constrained, and free parameters.
  
    Parameters:
    -----------
    exps : list
        A list of SloppyCell experiment objects.
    nets : list
        A list of SloppyCell network objects
    params_fixed : KeyedList
        A keyed list of fixed parameters
    params_constrained : KeyedList
        A keyed list of constrained parameters
    params_free : KeyedList
        A keyed list of free parameters
    global_fit : bool, optional
        When true, perform ensemble analysis.
    exp_ids : list, optional
        A list of ids to restrict the fit to a subset of experiments.
    local_it : int, optional
        Number of iterations for the local fit.
    global_it : int
        Number of iterations for the ensemble analysis.
    return_ens : bool
        When true, return the parameter ensemble.
  
    Returns:
    --------
    out : tuple of Model and KeyedList and, optionally, two lists
        If `return_ens` is false (the default case), return the SloppyCell 
        model and the optimal parameter set.

        Otherwise return also the parameter ensemble and the corresponding
        costs.
  
    """
    if exp_ids is None:
        exp_ids = exps.keys()

    exp_set = [exps[exp_id] for exp_id in exp_ids]
    net_set = [item for sublist in [nets[key].values() for key in exps.keys()] for item in sublist]

    keys_opt = list(set(params_constrained.keys()).union(set(params_free.keys())))
    
    for net in net_set:
        keys_non_opt = list(set(net.optimizableVars.keys()) - set(keys_opt))    
        for key in keys_non_opt:
            net.set_var_constant(key, is_constant=True)
            net.set_var_optimizable(key, is_optimizable=False)
        for key in keys_opt:
            net.set_var_optimizable(key, True) 
        net.set_var_ics(params_fixed)
        net.set_var_ics(params_constrained)
        net.set_var_ics(params_free)

    m = Model(exp_set, net_set)

    for key, val in params_free.items():
        res = Residuals.PriorInLog(key+'_prior', key, np.log(val), np.log(np.sqrt(1000)))
        m.AddResidual(res)

    for key, val in params_constrained.items():
        res = Residuals.PriorInLog(key+'_prior', key, np.log(val), np.log(np.sqrt(2)))
        m.AddResidual(res)

    print "Performing local optimization ..."
    params_opt = Optimization.fmin_lm_log_params(m, params=m.params, maxiter=local_it, disp=False)
    print " done.\n"


    if global_fit:
        print "Performing ensemble analysis ..."
        Network.full_speed()
        gs_start = np.Inf
        gs_end = 0
        params = params_opt
    
        while np.abs(gs_start-gs_end)>1:
            gs_start = gs_end
            j = m.jacobian_log_params_sens(np.log(params))
            jtj = np.dot(np.transpose(j), j)
            ens, gs, r, sf = Ensembles.ensemble_log_params(m, params, jtj, steps=np.round(global_it/10), temperature=1, save_scalefactors=True)
            gs_end = np.min(gs)
            params = ens[np.argmin(gs)]

        j = m.jacobian_log_params_sens(np.log(params))
        jtj = np.dot(np.transpose(j), j)
        ens, gs, r, sf = Ensembles.ensemble_log_params(m, params, jtj, steps=global_it, temperature=1, save_scalefactors=True) 
        params_opt = ens[np.argmin(gs)]
        print " done.\n"

    m.params = params_opt

    if return_ens:
        if not global_fit:
            ens = []
            gs = []
        return m, params_opt, ens, gs
    else:
        return m, params_opt


def plot_fit(m, nets, params=None, exp_ids=None, xlim=None, ylim=None, file=None):

    if params is not None:
        m.params = params
    else:
        params = m.params

    sf = m.compute_scale_factors(1)[0]

    exps = m.exptColl
    
    if exp_ids is None:
        exp_ids = exps.keys()

    fig, axes = plt.subplots(1, len(exp_ids), figsize=(len(exp_ids)*5,5))

    for i, exp_id in zip(range(len(exp_ids)), exp_ids):
        exp = exps[exp_id]
        if len(exp_ids)>1:
            ax = axes[i]
        else:
            ax = axes
        data = exp.get_data()
        t_lims = get_tlims_from_data(data) 
        plot_vars = [var for var in data[key].keys() for key in data.keys()]
        plot_vars = list(np.unique(plot_vars))        
        
        for var in plot_vars:
            sf_i = sf[exp_id][var]
            for net_key, net in nets[exp_id].items():
                net = net.copy()
                net_id = net.get_id()
                net.set_var_ics(params)
                net.set_var_vals(params)  
                traj = net.integrate([0, t_lims[1]*2])
                p = ax.plot(traj.timepoints-t_lims[0], traj.get_var_traj(var), label=net.get_name())
                [ax.scatter(t-t_lims[0], d[0]/sf_i, color=p[0].get_color()) for (t,d) in data[net_id][var].items()];
        if xlim is None:
            xlim = np.array([-0.1,1.1])*np.ptp(t_lims)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_xlabel('time')
        ax.set_ylabel('concentration')
        ax.legend()
        ax.set_title(exp_id, fontweight='bold')
    plt.tight_layout()
    if file is not None:
        plt.savefig(file)



def get_tlims_from_data(data):
    t_data = []
    for k1 in data.keys():
        d1 = data[k1]
        for k2 in d1.keys():
            t_data.append(d1[k2].keys())
    tmin = np.min(t_data)
    tmax = np.max(t_data)
    return [tmin, tmax]


def plot_pars(ens_list, pars, net, pars_bg = None, labels = None, ax=None, legend=True, file=None):
    '''
    Plot parameters for a given list of parameter ensembles.

    '''
    if ax==None:
        fig,ax = plt.subplots(1,1, figsize=(3+0.8*len(pars),4))
        plot = True
    
    if labels==None:
        labels = ['ens_'+str(i) for i in range(len(ens_list))]

    if len(pars_bg) == 1 and type(pars_bg) == list:
        pars_bg = pars_bg*len(ens_list)

    x_pos = np.linspace(-0.3,0.3,len(ens_list))

    for k, e in zip(range(len(ens_list)), ens_list):
        par_array = np.empty((len(e), len(pars)))
        for i, pars_i in zip(range(len(e)), e):
            net.set_var_vals(pars_bg[k])
            net.set_var_vals(pars_i)
            par_array[i,] = np.log10([net.get_var_val(par) for par in pars])
        for j, par in zip(range(len(pars)), pars):
            par_mean = np.mean(par_array[:,j])
            par_std = np.std(par_array[:,j])
            if j==0:
                p = ax.errorbar(j+x_pos[k], par_mean, yerr=par_std, fmt='.', ms=10, lw=2, capsize=3, label=labels[k])
                col = p.get_children()[0].get_color()
            else:
                ax.errorbar(j+x_pos[k], par_mean, yerr=par_std, fmt='.', color=col, ms=10, lw=2, capsize=3)

    for j in range(len(pars)):
        if j%2:
            ax.axvspan(j-0.5, j+0.5, alpha=0.05, color='k')

    ax.set_xticks(range(len(pars)));
    ax.set_xticklabels(pars);
    ax.set_xlim([-0.5,len(pars)-0.5])
    yticks = ax.get_yticks()
    yticks = yticks[yticks % 1 == 0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([r'$10^{'+str(int(n))+'}$' for n in yticks])
    ax.axhline(y=0, color='k', ls='--', alpha=0.2)
    ax.axhline(y=0, color='k', ls='--', alpha=0.2)
    
    if legend:
        ax.legend(bbox_to_anchor=(1,1))


    if file is not None:
        if plot:
            plt.tight_layout()
            plt.savefig(file+'.pdf')
