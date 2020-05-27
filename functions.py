import matplotlib.pyplot as plt
import numpy as np
from SloppyCell.ReactionNetworks import *


def fit_exps(exps, nets, params_fixed, params_constrained, params_free, global_fit = True, exp_ids=None, local_it=20, global_it=10000, return_ens=True):
    """ 
    Fit a set of experiments in a SloppyCell model.
  
    This function generates a SloppyCell model from a set of experiments and nets and performs local and global optimization 
    given sets of fixed, constrained, and free parameters.
  
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
    out : tuple of Model and KeyedList or tuple of Model and two lists
        If `return_ens` is false (the default case), return the SloppyCell model and the optimal parameter set.

        Otherwise return the model, the parameter ensemble and the corresponding costs.
  
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
    params_opt = Optimization.fmin_lm_log_params(m, params=m.params, maxiter=100, disp=False)
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
            ens = [params_opt]
            gs = [0]
        return m, ens, gs
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
        plot_vars = [data[key].keys()[0] for key in data.keys()]
        plot_vars = list(np.unique(plot_vars))        
        
        for var in plot_vars:
            sf_i = sf[exp_id][var]
            for net_key, net in nets[exp_id].items():
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
