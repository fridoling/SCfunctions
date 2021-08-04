import matplotlib.pyplot as plt
import numpy as np
import re
from libsbml import *
from SloppyCell.ReactionNetworks import *
import warnings


### Function to create an SBML model
def create_model():
    """Returns a simple but complete SBML Level 3 model for illustration."""
    try:
        document = SBMLDocument(3, 1)
    except ValueError:
        raise SystemExit('Could not create SBMLDocument object')

    model = document.createModel()
    model.setTimeUnits("second")
    model.setExtentUnits("mole")
    model.setSubstanceUnits('mole')

    # Create a unit definition we will need later.  Note that SBML Unit
    # objects must have all four attributes 'kind', 'exponent', 'scale'
    # and 'multiplier' defined.

    per_second = model.createUnitDefinition()
    per_second.setId('per_second')
    unit = per_second.createUnit()
    unit.setKind(UNIT_KIND_SECOND)
    unit.setExponent(-1)
    unit.setScale(0)
    unit.setMultiplier(1)

    # Create a compartment inside this model, and set the required
    # attributes for an SBML compartment in SBML Level 3.

    c1 = model.createCompartment()
    c1.setId('c1')
    c1.setConstant(True)
    c1.setSize(1)
    c1.setSpatialDimensions(3)
    c1.setUnits('litre')

    # And we're done creating the basic model.
    # Now return a text string containing the model in XML format.

#     return writeSBMLToString(document)
    return document


## Functions that allow the modification of SBML models
def add_assignment(model, var, species_list, pars, formula, replace = False, type = "species"):
    if model.getAssignmentRuleByVariable(var):
        if replace:
            assignment_old = model.getAssignmentRuleByVariable("keor")
            assignment_old.removeFromParentAndDelete()
        else:
            raise ValueError("Variable '"+var+"' already has assignment.")
    species_ids = [model.getSpecies(i).getId() for i in range(model.getNumSpecies())]
    parameter_ids = [model.getParameter(i).getId() for i in range(model.getNumParameters())]
    if type == "species":
        if var not in species_list:
            species_list += [var]
    elif type == "parameter" and var not in pars:
        pars += [var]
    else:
        raise ValueError("'type' must be either 'species' or 'parameter'")
    assignment = model.createAssignmentRule()
    assignment.setVariable(var)
    assignment.setFormula(formula)
    for par in pars:
        if par not in parameter_ids:
            add_parameter(model, par)
    for species in species_list:
        if species not in species_ids:
            add_species(model, species)
    if type == "species":
        species = model.getSpecies(var)
        species.setBoundaryCondition(True)


def add_reaction(model, reactants, products, modifiers, pars, formula, rx_id, reversible = False, replace = False):
    species_ids = [model.getSpecies(i).getId() for i in range(model.getNumSpecies())]
    parameter_ids = [model.getParameter(i).getId() for i in range(model.getNumParameters())]
    rx_ids = [model.getReaction(i).getId() for i in range(model.getNumReactions())]
    if rx_id in rx_ids:
        if replace:
            reaction_old = model.getReaction(rx_id)
            reaction_old.removeFromParentAndDelete()
        else:
            raise ValueError("reaction '"+rx_id+"' already exists in model.")
    reaction = model.createReaction()
    reaction.setFast(False)
    reaction.setReversible(reversible)
    for par in pars:
        if par not in parameter_ids:
            add_parameter(model, par)
    for reactant in reactants:
        if reactant not in species_ids:
            add_species(model, reactant)
        species = reaction.createReactant()
        species.setSpecies(reactant)
        species.setConstant(True)
        species.setStoichiometry(1)
    for product in products:
        if product not in species_ids:
            add_species(model, product)        
        species = reaction.createProduct()
        species.setSpecies(product)
        species.setConstant(True)
        species.setStoichiometry(1)
    for modifier in modifiers:
        if modifier not in species_ids:
            add_species(model, modifier)
        species = reaction.createModifier()
        species.setSpecies(modifier)
    reaction.setId(rx_id)
    kinetic_law = reaction.createKineticLaw()
    kinetic_law.setFormula(formula)

def add_binding_reaction(model, reactants, products, kon, koff, rx_id, replace = False):
    species_ids = [model.getSpecies(i).getId() for i in range(model.getNumSpecies())]
    parameter_ids = [model.getParameter(i).getId() for i in range(model.getNumParameters())]
    rx_ids = [model.getReaction(i).getId() for i in range(model.getNumReactions())]
    rx_id_f = rx_id+"_f"
    rx_id_r = rx_id+"_r"
    for par in [kon, koff]:
        if par not in parameter_ids:
            add_parameter(model, par)
    for reactant in reactants:
        if reactant not in species_ids:
            add_species(model, reactant)
    for product in products:
        if product not in species_ids:
            add_species(model, product)        
    for rx_id_x in [rx_id_f, rx_id_r]:
        if rx_id_x in rx_ids:
            if replace:
                reaction_old = model.getReaction(rx_id_x)
                reaction_old.removeFromParentAndDelete()
            else:
                raise ValueError("reaction '"+rx_id_x+"' already exists in model.")
    reaction_f = model.createReaction()
    reaction_f.setReversible(False)
    reaction_f.setFast(False)
    reaction_r = model.createReaction()
    reaction_r.setReversible(False)
    reaction_r.setFast(False)
    for reactant in reactants:
        species_f = reaction_f.createReactant()
        species_r = reaction_r.createProduct()
        for species in [species_f, species_r]:
            species.setSpecies(reactant)
            species.setConstant(True)
            species.setStoichiometry(1)
    for product in products:
        species_f = reaction_f.createProduct()
        species_r = reaction_r.createReactant()
        for species in [species_f, species_r]:
            species.setSpecies(product)
            species.setConstant(True)
            species.setStoichiometry(1)
    reaction_f.setId(rx_id_f)
    formula_f = " * ".join([kon]+reactants)
    kinetic_law_f = reaction_f.createKineticLaw()
    kinetic_law_f.setFormula(formula_f)
    reaction_r.setId(rx_id_r)
    formula_r = " * ".join([koff]+products)
    kinetic_law_r = reaction_r.createKineticLaw()
    kinetic_law_r.setFormula(formula_r)


def add_catalytic_reaction(model, reactants, products, catalyst, kcat, rx_id, replace = False):
    species_ids = [model.getSpecies(i).getId() for i in range(model.getNumSpecies())]
    parameter_ids = [model.getParameter(i).getId() for i in range(model.getNumParameters())]
    rx_ids = [model.getReaction(i).getId() for i in range(model.getNumReactions())]
    if rx_id in rx_ids:
        if replace:
            reaction_old = model.getReaction(rx_id)
            reaction_old.removeFromParentAndDelete()
        else:
            raise ValueError("reaction '"+rx_id+"' already exists in model.")
    reaction = model.createReaction()
    reaction.setFast(False)
    reaction.setReversible(False)
    if kcat not in parameter_ids:
        add_parameter(model, kcat)
    for reactant in reactants:
        if reactant not in species_ids:
            add_species(model, reactant)
        species = reaction.createReactant()
        species.setSpecies(reactant)
        species.setConstant(True)
        species.setStoichiometry(1)
    for product in products:
        if product not in species_ids:
            add_species(model, product)        
        species = reaction.createProduct()
        species.setSpecies(product)
        species.setConstant(True)
        species.setStoichiometry(1)
    if catalyst not in species_ids:
        add_species(model, catalyst)
    species = reaction.createModifier()
    species.setSpecies(catalyst)
    reaction.setId(rx_id)
    formula = " * ".join([kcat] + reactants + [catalyst])
    kinetic_law = reaction.createKineticLaw()
    kinetic_law.setFormula(formula)


def add_species(model, species_id, concentration = 0.0, compartment = None, hasOnlySubstanceUnits = False, boundaryCondition = False, constant = False):
    species = model.createSpecies()
    species.setId(species_id)
    if compartment is None:
        compartment = model.getCompartment(0).getId()
    species.setCompartment(compartment)
    species.setConstant(constant)
    species.setInitialConcentration(concentration)
    species.setHasOnlySubstanceUnits(hasOnlySubstanceUnits)
    species.setBoundaryCondition(boundaryCondition)

    
def add_parameter(model, parameter_id, value = 1.0, constant = True):
    parameter = model.createParameter()
    parameter.setId(parameter_id)
    parameter.setConstant(constant)
    parameter.setValue(value)

def add_conservation(model, free_var, tot_var, bound_vars):
    species_ids = [model.getSpecies(i).getId() for i in range(model.getNumSpecies())]
    parameter_ids = [model.getParameter(i).getId() for i in range(model.getNumParameters())]
    if model.getAssignmentRule(free_var):
        raise ValueError("assignment rule for "+free_var+" already exists in model.")
    for bound_var in bound_vars:
        if bound_var not in species_ids:
            add_species(model, bound_var)
    if free_var not in species_ids:
        add_species(model, free_var)
    if tot_var not in parameter_ids:
        add_parameter(model, tot_var)
    rule = model.createAssignmentRule()
    rule.setVariable(free_var)
    rule.setFormula("-".join([tot_var]+bound_vars))

def add_combined_species(model, combined_var, compound_vars):
    species_ids = [model.getSpecies(i).getId() for i in range(model.getNumSpecies())]
    if model.getAssignmentRule(combined_var):
        raise ValueError("assignment rule for "+combined_var+" already exists in model.")
    for compound_var in compound_vars:
        if compound_var not in species_ids:
            add_species(model, compound_var)
    if combined_var not in species_ids:
        add_species(model, combined_var)
    rule = model.createAssignmentRule()
    rule.setVariable(combined_var)
    rule.setFormula("+".join(compound_vars))
    
def add_to_cons(model, free_var, new_var):
    rule = model.getAssignmentRuleByVariable(free_var)
    formula = rule.getFormula()
    if '(' not in formula:
        formula.split(' ')
        components = formula.split(' - ')
        new_formula = components[0] + ' - ' + '(' + ' + '.join(components[1:] + [new_var]) + ')'
    else:
        formula = rule.getFormula()
        new_formula = formula[:-1]+' + '+new_var+')'
    rule.setFormula(new_formula)
    
def add_to_tot(model, tot_var, new_var):
    rule = model.getAssignmentRuleByVariable(tot_var)
    formula = rule.getFormula()    
    new_formula = formula+' + '+new_var
    rule.setFormula(new_formula)

def replace_assignments(model, var = None):
    species_ids = [model.getSpecies(i).getId() for i in range(model.getNumSpecies())]
    reactions = [model.getReaction(i) for i in range(model.getNumReactions())]
    if var is None:
        ass_species_ids = [species_id for species_id in species_ids if model.getAssignmentRule(species_id) is not None]
    else:
        ass_species_ids = [var]
    for ass_species_id in ass_species_ids:
        assignment_formula = model.getAssignmentRule(ass_species_id).formula
        for reaction in reactions:
            modifiers = [modifier.getSpecies() for modifier in reaction.getListOfModifiers()]
            if ass_species_id in modifiers:
                kinetic_law = reaction.getKineticLaw()
                formula = kinetic_law.getFormula()
                formula_sub = re.sub("(^|[-(*+]|\s)"+ass_species_id+"($|[-)*+]|\s)", r"\1("+assignment_formula+r")\2", formula)
                kinetic_law.setFormula(formula_sub)
                reaction.modifiers.remove(ass_species_id)
                for species_id in species_ids:
                    if re.search(r"(^|[-(*+]|\s)"+species_id+"($|[-)*+]|\s)", assignment_formula) is not None:
                        species = reaction.createModifier()
                        species.setSpecies(species_id)


### Functions for Plotting
def parSS(net, outExp='AC', par='ksyn', parLim=[0, 100], nPoints=10, logscale=False):
    """Create steady state values of an expression for given values of a parameter.
    
    Arguments
    ---------
    net -- SloppyCell net used for simulation
    outExp -- expression to be evaluated at steady state (default 'AC')
    par -- bifurcation parameter (default 'ksyn')
    parLim -- interval in which parameter is varied (default [0, 100])
    nPoints -- number of points (default 10)

    Returns
    -------
    tuple of vectors with values of par and outExp
    """
    net_vary = net.copy(new_id='net_vary')
    dynVars = net_vary.dynamicVars.keys()
    fp = np.zeros(len(dynVars))
    ics = net_vary.get_var_ics()
    if logscale:
        X = 10**np.linspace(np.log10(parLim[0]), np.log10(parLim[1]), nPoints)
    else:
        X = np.linspace(parLim[0], parLim[-1], nPoints)
    Y = np.zeros(nPoints)
    for i in range(nPoints):
        net_vary.set_var_ic(par, X[i])
        try:
            fp, st = Dynamics.dyn_var_fixed_point(net_vary, with_logs=False, stability=True, dv0=fp)
            if st==-1:
                for dv in range(len(dynVars)):
                    net_vary.set_var_val(dynVars[dv], fp[dv])
                    net_vary.set_var_ic(dynVars[dv], fp[dv])
                Y[i] = net_vary.evaluate_expr(outExp)
            else:
                try:
                    fp, st = Dynamics.dyn_var_fixed_point(net_vary, with_logs=False, stability=True)
                    for dv in range(len(dynVars)):
                        net_vary.set_var_val(dynVars[dv], fp[dv])
                        net_vary.set_var_ic(dynVars[dv], fp[dv])
                    Y[i] = net_vary.evaluate_expr(outExp)
                except:
                    Y[i] = np.inf
        except:
            try:
                fp, st = Dynamics.dyn_var_fixed_point(net_vary, with_logs=False, stability=True)
                for dv in range(len(dynVars)):
                    net_vary.set_var_val(dynVars[dv], fp[dv])
                    net_vary.set_var_ic(dynVars[dv], fp[dv])
                Y[i] = net_vary.evaluate_expr(outExp)
            except:
                try:
                    net_vary.integrate([0,100])
                    fp, st = Dynamics.dyn_var_fixed_point(net_vary, with_logs=True, stability=True)
                    for dv in range(len(dynVars)):
                        net_vary.set_var_val(dynVars[dv], fp[dv])
                        net_vary.set_var_ic(dynVars[dv], fp[dv])
                    Y[i] = net_vary.evaluate_expr(outExp)                        
                except:
                    Y[i] = np.inf
    return(X,Y)


def plotCumTraj(traj, outVars=None, cols=None, axis=None):
    X = traj.timepoints
    if outVars is None:
        outVars = traj.dynamicVarKeys
    outVecs = list()
    outVecs.append(np.zeros_like(X))
    if cols is None:
        cols = plt.cm.rainbow_r(np.linspace(0,1,len(outVars)))
    elif len(cols)!=len(outVars):
        raise Exception("Number of colors doesn't match number of variables.")
    for i in range(len(outVars)):
        var_i = outVars[i]
        outVecs.append(outVecs[i]+traj.get_var_traj(var_i))
        if axis is None:
            plt.fill_between(X, outVecs[i], outVecs[i+1],label=var_i,color=cols[i])
            plt.legend(loc=2)
        else:
            axis.fill_between(X, outVecs[i], outVecs[i+1],label=var_i,color=cols[i])
            axis.legend(loc=2)

    
def plotCumSS(net, outVars=None, par='ksyn', parLim=[0,100], nPoints=10, cols=None,axis=None):
    X = np.linspace(parLim[0], parLim[-1], nPoints)
    dynVars = net.dynamicVars.keys()
    if outVars is None:
        outVars = dynVars
    outVecs = list()
    outVecs.append(np.zeros(nPoints))
    if cols is None:
        cols = plt.cm.rainbow_r(np.linspace(0,1,len(outVars)))
    elif len(cols)!=len(outVars):
        raise Exception("Number of colors doesn't match number of variables.")    
    for i in range(len(outVars)):
        var_i = outVars[i]
        x,y = parSS(net, outExp=var_i, par=par, parLim=parLim, nPoints=nPoints)
        outVecs.append(outVecs[i]+y)
        if axis is None:
            plt.fill_between(X, outVecs[i], outVecs[i+1],label=var_i,color=cols[i])
            plt.legend(loc=2)
        else:
            axis.fill_between(X, outVecs[i], outVecs[i+1],label=var_i,color=cols[i])
            axis.legend(loc=2)
    
        
def parSS2D(net, parX='ksyn', parY='ksyn', outExp='AC', xLim=[0,100], yLim=[0,100], xPoints=10, yPoints=10, logX=False, logY=False):
    """Create grid of steady state values for an expression depending on two parameters
    
    Arguments
    ---------
    net -- SloppyCell net used for simulation
    outExp -- expression to be evaluated at steady state (default 'AC')
    parX/parY -- bifurcation parameters (default 'ksyn')
    xlim/ylim -- intervals in which parameters are varied (default [0, 100])
    xPoints/yPoints -- number of points for parameter (default 10)
    logX/logY -- should parameter ranges be in log-scale? (default 'False')

    Returns
    -------
    triple of vectors with values of parX, parY and outExp that can be used for 3D plotting
    """
    
    net_vary = net.copy(new_id='net_vary')
    dynVars = net_vary.dynamicVars.keys()
    fp = np.zeros(len(dynVars))
    ics = net_vary.get_var_ics()
    if logX:
        xList = np.exp(np.linspace(np.log(xLim[0]), np.log(xLim[-1]), xPoints))
    else:
        xList = np.linspace(xLim[0], xLim[-1], xPoints)
    if logY:
        yList = np.exp(np.linspace(np.log(yLim[0]), np.log(yLim[-1]), xPoints))
    else:        
        yList = np.linspace(yLim[0], yLim[-1], yPoints)
    X,Y = np.meshgrid(xList, yList)
    Z=np.zeros(X.shape)
    for i in np.ndindex(X.shape):
        net_vary.set_var_ic(parX, X[i])
        net_vary.set_var_ic(parY, Y[i])        
        try:
            fp, st = Dynamics.dyn_var_fixed_point(net_vary, with_logs=False, stability=True)
            if st==-1:
                for dv in range(len(dynVars)):
                    net_vary.set_var_val(dynVars[dv], fp[dv])
                Z[i] = net_vary.evaluate_expr(outExp)
                net_vary.set_var_ics(ics)
            else:
                Z[i] = np.nan
        except:
            net_vary.integrate([0,1000])
            try:
                fp, st = Dynamics.dyn_var_fixed_point(net_vary, with_logs=False, stability=True)
                if st==-1:
                    for dv in range(len(dynVars)):
                        net_vary.set_var_val(dynVars[dv], fp[dv])
                    Z[i] = net_vary.evaluate_expr(outExp)
                    net_vary.set_var_ics(ics)
                else:
                    Z[i] = np.nan
            except:
                Z[i] = np.nan
    return(X,Y,Z)


def plotSScontour(x,y,z, cStep=None, xLab=None, yLab=None, imTitle=None, file=None, logY=False, logX=False, vmin=None, vmax=None):
    ## i,j=where(z<Inf)
    ## x=x[0:max(i),0:max(j)]
    ## y=y[0:max(i),0:max(j)]
    ## z=z[0:max(i),0:max(j)]
    plt.imshow(z,
           interpolation='bilinear',
           origin='lower',
           cmap=plt.cm.YlOrRd,
           aspect='auto',
           vmin=vmin,
           vmax=vmax,
           extent=[np.nanmin(x),np.nanmax(x),np.nanmin(y),np.nanmax(y)])     
    zmin = np.nanmin(z[np.where(z>-np.inf)])
    zmax = np.nanmax(z[np.where(z<np.inf)])
    if cStep is None:
        cStep = 10**np.floor(np.log10(zmax-zmin))
    zmin = np.floor(zmin/cStep)*cStep
    levels=np.arange(zmin, np.ceil(zmax), cStep)
    if levels.size<5:
        levels=np.arange(zmin, np.ceil(zmax), cStep/2)        
    if logX:
        plt.xscale('log')
    if logY:
        plt.yscale('log')
    c=plt.contour(x,y,z, levels=levels, colors='black')
    fstr=np.int(-np.floor(np.log10(cStep)))
    if fstr<0:
        fmt='%1.0f'
    else:
        fmt='%1.'+str(fstr)+'f'
    plt.clabel(c, fontsize=10, inline=1, fmt=fmt)
    if xLab is not None:
        plt.xlabel(xLab)
    if yLab is not None:       
        plt.ylabel(yLab)
    if imTitle is not None:
        plt.title(imTitle)
    if file is not None:
        plt.savefig(file)

def getCDF(res, thresh, offset=0, plot=True, **kwargs):
    """Get adaptation times and plot (empirical) cumulative density distribution.
    Arguments
    ---------
    res -- list of trajectories (nx2 arrays with timepoints and corresponding values of AC)
    thresh -- threshold used to define adaptation events (first crossing)
    offset -- times before are ignored (default 0)
    plot -- if True, plot ECDF
    **kwargs -- additional arguments for plotting

    Returns
    _______
    vector with adaptation times
    
    """
    adT = np.zeros(len(res))
    xmax = 0
    for n in range(len(res)):
        res_n = res[n]
        ind0 = np.argmax(res_n[:,0]>=offset)
        traj_times0 = res_n[:,0]
        traj_times = res_n[ind0::,0]-offset
        xmax = max(traj_times[-1], xmax)
        traj0 = res_n[:,1]
        traj = res_n[ind0::,1]
        if max(traj)<=thresh:
            adT[n] = np.inf
        else:
            ad_ind = np.argmax(traj>thresh)
            adT[n] = traj_times[ad_ind]       
    if plot:
        adT_0 = np.append(adT, 0)
        adT_sorted = np.sort(adT_0)
        ad_inc=np.array(range(len(adT_0)))/float(len(adT))
        plt.step(adT_sorted, ad_inc,  **kwargs)
        plt.xlim([0, xmax])
    return(adT)



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
    net_set = [item for sublist in [nets[key].values() for key in exp_ids] for item in sublist]

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
        res = Residuals.PriorInLog(key+'_prior', key, np.log(val), np.log(np.sqrt(10)))
        m.AddResidual(res)

    for key, val in params_constrained.items():
        res = Residuals.PriorInLog(key+'_prior', key, np.log(val), np.log(np.sqrt(1.1)))
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

    m.params.update(params_opt)
    m.CalculateForAllDataPoints(params_opt);

    if return_ens:
        if not global_fit:
            ens = []
            gs = []
        return m, params_opt, ens, gs
    else:
        return m, params_opt


def plot_fit(m, nets, params=None, exp_ids=None, xlim=None, ylim=None, t_iv = None, plot_vars=None, file=None):

    if params is not None:
        m.params.update(params)
        m.CalculateForAllDataPoints(params)
    else:
        params = m.params

    sf = m.compute_scale_factors(1)[0]

    exps = m.exptColl
    
    if exp_ids is None:
        exp_ids = exps.keys()

    ydim = 1

    exp_vars = {}
    for exp_id in exp_ids:
        data = exps[exp_id].get_data()
        exp_vars_id = [var for key in data.keys() for var in data[key].keys()]
        if plot_vars is not None:
            exp_vars_id = np.intersect1d(plot_vars, exp_vars_id)
        exp_vars[exp_id] = list(np.unique(exp_vars_id))
        ydim = np.max((ydim, len(exp_vars[exp_id])))

    fig, axes = plt.subplots(ydim, len(exp_ids), figsize=(len(exp_ids)*5,ydim*4), sharey='row')

    for j, exp_id in zip(range(len(exp_ids)), exp_ids):
        exp = exps[exp_id]
        data = exp.get_data()
        t_lims = get_tlims_from_data(data)
        if t_iv is None:
            t_iv = t_lims[0]

        
        for i, var in zip(range(len(exp_vars[exp_id])), exp_vars[exp_id]):
            if len(exp_ids)>1 and ydim>1:
                ax = axes[i,j]
            elif len(exp_ids)>1 and ydim==1:
                ax = axes[j]
            elif len(exp_ids)==1 and ydim>1:
                ax = axes[i]
            else:
                ax = axes

            sf_i = sf[exp_id][var]
            for net_key, net in nets[exp_id].items():
                net = net.copy()
                net_id = net.get_id()
                net.set_var_ics(params)
                net.set_var_vals(params)  
                times = np.concatenate((np.linspace(0, t_iv, 101), 
                    np.linspace(t_iv, t_lims[1], 101),
                    np.linspace(t_lims[1], 2*t_lims[1], 101)
                    ))
                traj = net.integrate(times)
                p = ax.plot(traj.timepoints-t_iv, traj.get_var_traj(var), label=net.get_name())
                [ax.scatter(t-t_iv, d[0]/sf_i, color=p[0].get_color()) for (t,d) in data[net_id][var].items()];
            if xlim is None:
                xlim = np.array([-0.1,1.1])*np.ptp(t_lims)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.set_xlabel('time')
            ax.set_ylabel(var)
            ax.legend()
            if i==0:
                ax.set_title(exp_id, fontweight='bold')
    plt.tight_layout()
    if file is not None:
        plt.savefig(file)



def get_tlims_from_data(data):
    t_data = []
    for k1 in data.keys():
        d1 = data[k1]
        for k2 in d1.keys():
            t_data = t_data + d1[k2].keys()
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
    ax.set_xticklabels(pars, rotation=90);
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
