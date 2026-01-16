import shutil
from analyser import parse_model,analyse_model_full,get_mtype
from coder import writePythonCode
from sedTasker import exec_task, report_task
from sedCollector import  get_fit_experiments, get_task_info
import json
from scipy.optimize import Bounds,least_squares,shgo,dual_annealing,differential_evolution,basinhopping
import numpy
import copy
import math
import os
from sedChanger import calc_data_generator_results_sedVarIds
from simulator import  get_observables, sim_OneStep, sim_TimeCourse,get_externals_varies, load_module
from multiprocessing import Pool
import tempfile

# Worker-local globals
_worker_modules = {}
_worker_analysers = {}
_worker_cellml_models = {}
def init_worker(fitExperiments):
    """
    Initialize SWIG objects per worker process.
    """
    global _worker_modules, _worker_analysers, _worker_cellml_models
    for fitid, fitExperiment in fitExperiments.items():
        temp_model_source = fitExperiment['temp_model_source']
        model_base_dir = os.path.dirname(temp_model_source)
        external_variables_info = fitExperiment['external_variables_info']
        cellml_model,parse_issues=parse_model(temp_model_source, False)
        _worker_cellml_models[fitid] = cellml_model
 
        if not cellml_model:
            raise RuntimeError('Model parsing failed!')
        analyser, issues =analyse_model_full(cellml_model,model_base_dir,external_variables_info)       
        if analyser:
            # write Python code to a temporary file
            # make a directory in the model_base_dir for the temporary file if it does not exist
            _worker_analysers[fitid] = analyser
            temp_folder = model_base_dir+os.sep+ fitid+'_temp'
            if not os.path.exists(temp_folder):
                os.makedirs(temp_folder)
            tempfile_py, full_path = tempfile.mkstemp(suffix='.py', prefix=fitid+"_", text=True,dir=temp_folder)
            writePythonCode(analyser, full_path)
            module=load_module(full_path)
            _worker_modules[fitid] = module
            os.close(tempfile_py)
            # and delete temporary file
            os.remove(full_path)
            shutil.rmtree(temp_folder)
    return


def exec_sed_doc(doc, working_dir,base_out_path, rel_out_path=None, external_variables_info={}, external_variables_values=[],ss_time={},cost_type=None):
    """
    Execute a SED document.

    Parameters
    ----------
    doc : SedDocument
        The SED document to execute.
    working_dir : str
        working directory of the SED document (path relative to which models are located)
    base_out_path: str
        The base path to the directory that output files are written to.
    rel_out_path: str
        The relative path to the directory that output files are written to.
         * CSV: directory in which to save outputs to files
            ``{base_out_path}/{rel_out_path}/{report.getId()}.csv``

   external_variables_info: dict, optional
        The external variables to be specified, in the format of {id:{'component': , 'name': }}
    external_variables_values: list, optional
        The values of the external variables to be specified [value1, value2, ...]
    ss_time: dict, optional
        The time point for steady state simulation, in the format of {fitid:time}
    
    """
    doc = doc.clone() # clone the document to avoid modifying the original document
    listOfTasks = doc.getListOfTasks()
    listOfOutputs = doc.getListOfOutputs()
    # Print information about the SED document
    indent=0
    if len(listOfTasks)==0:
        print('SED document does not describe any tasks.')
        return
    print('{}Found {} tasks and {} outputs:\n'.format(
        ' ' * 2 * indent,
        len(listOfTasks),
        len(listOfOutputs),
    ))
    # execute tasks
    variable_results = {}
    for i_task, task in enumerate(doc.getListOfTasks()):
        if task.isSedTask ():
            mtype, module, sim_setting, observables, external_module, task_vars = get_task_info(
            doc,task,working_dir,external_variables_info=external_variables_info,external_variables_values=external_variables_values)
            try:
                current_state, variable_results_i= exec_task(mtype, module, sim_setting, observables, external_module, task_vars,current_state=None)
                print('Task {} executed successfully'.format(task.getId()))
                variable_results.update(variable_results_i)
            except Exception as exception:
                print(exception)
                return           

        elif task.isSedRepeatedTask ():
            raise RuntimeError('RepeatedTask not supported yet')
        
        elif task.isSedParameterEstimationTask ():
            fitExperiments,adjustables,adjustableParameters_info,method, opt_parameters=get_fit_experiments(doc,task,working_dir,external_variables_info)
            try:
                res=exec_parameterEstimationTask(fitExperiments,adjustables,method, opt_parameters,external_variables_values,ss_time,cost_type)
                fit_res_json=os.path.join(working_dir, task.getId()+'.json')
                fit_results={} 
                print('Values of objective function at the solution: {}'.format(res.fun))
                i=0
                fit_results['solution']={}
                for parameter in adjustableParameters_info.values():
                    print('The estimated value for variable {} in component {} is:'.format(parameter['name'],parameter['component']))
                    print(res.x[i])
                    fit_results['solution'][parameter['name']]={}
                    fit_results['solution'][parameter['name']]={'component':parameter['component'],'name':parameter['name'],'newValue':str(res.x[i])}
                    i+=1
                print('The full optimization result is:')
                print(res)
                # save fit results to json file
                with open(fit_res_json, 'w') as outfile:
                    json.dump(fit_results, outfile)
                print('Parameter estimation task {} executed successfully'.format(task.getId()))
            except Exception as exception:
                print(exception)
                return
    report_result = report_task(doc,task, variable_results, base_out_path, rel_out_path, report_formats =['csv'])

def exec_parameterEstimationTask(fitExperiments,adjustables,method, opt_parameters, external_variables_values,ss_time={},cost_type=None):
    """
    Execute a SedTask of type ParameterEstimationTask.
    The model is assumed to be in CellML format.
    The simulation type supported are 'steadyState' and 'timeCourse'.#TODO: add support for steadyState
    The ode solver supported are listed in KISAO_ALGORITHMS (.simulator.py)
    The optimisation algorithm supported are listed in KISAO_ALGORITHMS (.optimiser.py)

    Parameters
    ----------
    fitExperiments: dict
    external_variables_values: list, optional
        The values of the external variables to be specified [value1, value2, ...]
    ss_time: dict, optional
        The time point for steady state simulation, in the format of {fitid:time}
    cost_type: str, optional
        The cost function to be used for the optimisation. Default: None

    Raises
    ------
    RuntimeError
        If any operation failed.

    Returns
    -------
    res: scipy.optimize.OptimizeResult

    """ 	     
    # get optimisation settings and fit experiments
    if 'tol' in opt_parameters:
        tol=opt_parameters['tol']
    else:
        tol=1e-8
    if 'maxiter' in opt_parameters:
        maxiter=int(opt_parameters['maxiter']) 
    else:
        maxiter=1000
    if 'workers' in opt_parameters:
        worker_num=int(opt_parameters['workers'])
    else:
        worker_num=1

    bounds=Bounds(adjustables[0],adjustables[1])
    initial_value=adjustables[2]
    if method=='global optimization algorithm':
        #res= shgo(objective_function, bounds,args=(external_variables_values, fitExperiments, doc, ss_time,cost_type),
                               #options={'ftol': tol, 'maxiter': maxiter})
        with Pool(
        processes=worker_num,
        initializer=init_worker,
        initargs=(fitExperiments,)  # pass original fitExperiments to initializer
            ) as pool:
                res = shgo(
                    objective_function,
                    bounds,
                    args=(external_variables_values, fitExperiments, ss_time,cost_type),            
                    options={'ftol': tol, 'maxiter': maxiter},
                    workers=pool.map
                )
    elif method=='simulated annealing':
        res=dual_annealing(objective_function, bounds,args=(external_variables_values, fitExperiments, ss_time,cost_type),maxiter=maxiter, x0=initial_value)
    elif method=='evolutionary algorithm':
        popsize=opt_parameters['popsize'] if 'popsize' in opt_parameters else 15
        with Pool(
        processes=worker_num,
        initializer=init_worker,
        initargs=(fitExperiments,)  # pass original fitExperiments to initializer
            ) as pool:
            res = differential_evolution(
            objective_function,
            bounds,
            args=(external_variables_values, fitExperiments, ss_time, cost_type),
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            x0=initial_value,
            workers=pool.map,
            disp=True
        )
        #res=differential_evolution(objective_function, bounds,args=(external_variables_values, fitExperiments, doc, ss_time,cost_type),maxiter=maxiter,popsize=popsize, tol=tol,x0=initial_value)
    elif method=='random search':
        res=basinhopping(objective_function, initial_value,minimizer_kwargs={'args':(external_variables_values, fitExperiments,ss_time,cost_type)}) # cannot use bounds
    elif method=='local optimization algorithm':
        #res=least_squares(objective_function, initial_value, args=(external_variables_values, fitExperiments, doc, ss_time,cost_type), 
        #         bounds=bounds, ftol=tol, gtol=tol, xtol=tol, max_nfev=maxiter)
        with Pool(
        processes=worker_num,
        initializer=init_worker,
        initargs=(fitExperiments,)  # pass original fitExperiments to initializer
            ) as pool:
                res = least_squares(
                    objective_function,
                    initial_value, args=(external_variables_values, fitExperiments,ss_time,cost_type), 
                    bounds=bounds, ftol=tol, gtol=tol, xtol=tol, max_nfev=maxiter,
                    workers=pool.map
                )
    else:
        raise RuntimeError('Optimisation method not supported!')
    
    return res

def objective_function(param_vals, external_variables_values, fitExperiments, ss_time,cost_type=None):
    """ Objective function for parameter estimation task.
    The model is assumed to be in CellML format.

    Parameters
    ----------
    param_vals: list
        The values of the adjustable parameters to be specified [value1, value2, ...]
    external_variables_values: list
        The values of the external variables to be specified [value1, value2, ...]
    fitExperiments: dict
        The fit experiments to be specified, 
        in the format of {fitid:{'external_variables_info': , 'cellml_model': , 'analyser': , 
        'mtype': , 'module': , 'fitness_info': , 'parameters': , 'parameters_values': , 
        'adj_param_indices': , 'type': , 'sim_setting': }}
    doc: :obj:`SedDocument`
        An instance of SedDocument
    ss_time: dict
        The time point for steady state simulation, in the format of {fitid:time}

    Raises
    ------
    RuntimeError
        If any operation failed.

    Returns
    -------
    float
        The sum of residuals of all fit experiments.
    """
    residuals_sum=0
    sed_results={}
    for fitid,fitExperiment in fitExperiments.items():
        sub_param_vals=[]
        external_variables_info=fitExperiment['external_variables_info']
        cellml_model=_worker_cellml_models[fitid]
        analyser=_worker_analysers[fitid]
        mtype=get_mtype(analyser)
        module=_worker_modules[fitid]
        fitness_info=fitExperiment['fitness_info']
        parameters_info=fitExperiment['parameters']
        parameters_values=fitExperiment['parameters_values']         
        for param_index in fitExperiment['adj_param_indices']:
            sub_param_vals.append(param_vals[param_index])
        simulation_type=fitExperiment['type']
        sim_setting=fitExperiment['sim_setting']
        observables_info=fitness_info[0]
        observables=get_observables(analyser,cellml_model,observables_info)
        parameters=get_observables(analyser,cellml_model,parameters_info)
        observables_weight=fitness_info[1]
        observables_exp=fitness_info[2]     
        if simulation_type=='timeCourse':
            external_variables_values_extends=external_variables_values+sub_param_vals+parameters_values    
            try:
                external_module=get_externals_varies(analyser, cellml_model, external_variables_info, external_variables_values_extends)
            except ValueError as exception:
                print(exception)
                raise RuntimeError(exception)
            try:
                current_state=sim_TimeCourse(mtype, module, sim_setting, observables, external_module,current_state=None,parameters=parameters)
                sed_results = copy.deepcopy(current_state[-1])
            except Exception as exception:
                print(exception)
                return 1e12

        elif simulation_type=='steadyState':
            observable_exp_temp=observables_exp[list(observables_exp.keys())[0]]
            for i in range(len(observable_exp_temp)): # assume all observables and experimental conditions have the same number of data points
                sim_setting.step=ss_time[fitid]
                sim_setting.output_start_time=sim_setting.step 
                sim_setting.output_end_time=sim_setting.step
                sim_setting.number_of_steps=0
                parameters_value=[]
                for parameter in parameters_values:
                    parameters_value.append(parameter[i])
                external_variables_values_extends=external_variables_values+sub_param_vals+parameters_value
                try:
                    external_module=get_externals_varies(analyser, cellml_model, external_variables_info, external_variables_values_extends)
                except ValueError as exception:
                    print(exception)
                    raise RuntimeError(exception)
                if i==0:
                    try:
                        #current_state=sim_OneStep(mtype, module, sim_setting, observables, external_module,current_state=None,parameters=parameters)
                        current_state=sim_OneStep(mtype, module, sim_setting, observables, external_module,current_state=None,parameters=parameters)
                        sed_results = copy.deepcopy(current_state[-1])
                    except Exception as exception:
                        print(exception)
                        return 1e12
                else:
                    try:
                      #  current_state=sim_OneStep(mtype, module, sim_setting, observables, external_module,current_state=current_state,parameters=parameters)
                        current_state=sim_OneStep(mtype, module, sim_setting, observables, external_module,current_state=None,parameters=parameters)
                        for key, value in current_state[-1].items():
                            sed_results[key]=numpy.append(sed_results[key],value)
                    except Exception as exception:
                        print(exception)
                        return 1e12
        else:
            raise RuntimeError('Simulation type not supported!')
        
        residuals={}
        for key, exp_value_ in observables_exp.items():
            sedVarIds, mathString, workspace,exp_value=exp_value_
            sim_value=calc_data_generator_results_sedVarIds(sedVarIds, mathString, workspace,sed_results)
            if cost_type=='AE':
                residuals[key]=abs(sim_value-exp_value)
                residuals_sum+=numpy.sum(residuals[key]*observables_weight[key])
            elif cost_type=='MIN-MAX':
                residuals[key]=abs(sim_value-exp_value)/(max(exp_value)-min(exp_value))
                residuals_sum+=numpy.sum(residuals[key]*observables_weight[key])
            elif cost_type=='Z-SCORE':
                residuals[key]=abs(sim_value-exp_value)/numpy.std(exp_value)
                residuals_sum+=numpy.sum(residuals[key]*observables_weight[key])
            elif cost_type is None or cost_type=='MSE':
                # MSE is the default cost function
                residuals[key]=(sim_value-exp_value)**2
                residuals_sum+=numpy.sum(residuals[key]*observables_weight[key])/len(exp_value)              
            else:
                raise RuntimeError('Cost type not supported!')
                
        if math.isnan(residuals_sum):
            return 1e12
    return residuals_sum               
            