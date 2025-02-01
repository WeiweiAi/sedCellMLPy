from sedCollector import  get_fit_experiments, get_task_info
from sedChanger import calc_data_generator_results
from simulator import sim_UniformTimeCourse, get_observables, sim_OneStep, sim_TimeCourse,get_externals_varies
from sedReporter import exec_report
import tempfile
import os
import sys
from scipy.optimize import Bounds,least_squares,shgo,dual_annealing,differential_evolution,basinhopping
import numpy
import copy
import math



def exec_task(doc,task,working_dir,external_variables_info={},external_variables_values=[],current_state=None):
    """ Execute a SedTask.
    The model is assumed to be in CellML format.
    The simulation type supported are UniformTimeCourse, OneStep and SteadyState.#TODO: add support for OneStep and SteadyState
    The ode solver supported are listed in KISAO_ALGORITHMS (.simulator.py)

    Parameters
    ----------
    doc: :obj:`SedDocument`
        An instance of SedDocument
    task: :obj:`SedTask`
        The task to be executed.
    working_dir: str
        working directory of the SED document (path relative to which models are located)
    external_variables_info: dict, optional
        The external variables to be specified, in the format of {id:{'component': , 'name': }}
    external_variables_values: list, optional
        The values of the external variables to be specified [value1, value2, ...]
    current_state: tuple, optional
        The format is (voi, states, rates, variables, current_index, sed_results)
    
    Raises
    ------
    RuntimeError
        If any operation failed.

    Returns
    -------
    tuple
        (tuple, dict)
        The current state of the simulation and the variable results of the task. 
        The format of the current state is (voi, states, rates, variables, current_index, sed_results)
        The format of the variable results is {sedVar_id: numpy.ndarray}
        numpy.ndarray is a 1D array of the variable values at each time point.   
    """
    mtype, module, sim_setting, observables, external_variable, task_vars = get_task_info(
        doc,task,working_dir,external_variables_info=external_variables_info,external_variables_values=external_variables_values)
    
    if sim_setting.type=='UniformTimeCourse':
        try:
            current_state=sim_UniformTimeCourse(mtype, module, sim_setting, observables, external_variable, current_state,parameters={})
        except RuntimeError as exception:
            print(exception)
            raise RuntimeError(exception)
    elif sim_setting.type=='OneStep':
        try:
            current_state=sim_OneStep(mtype, module, sim_setting, observables, external_variable, current_state,parameters={})
        except RuntimeError as exception:
            print(exception)
            raise RuntimeError(exception)
    else:
        raise RuntimeError('Simulation type not supported!')
    
    task_variable_results=current_state[-1]
    # check that the expected variables were recorded
    variable_results = {}
    if len(task_variable_results) > 0:
        for i,ivar in enumerate(task_vars):
            variable_results[ivar.getId()] = task_variable_results.get(ivar.getId(), None)            

    return current_state, variable_results

def report_task(doc,task, variable_results, base_out_path, rel_out_path, report_formats =['csv']):
    """ Generate the outputs of a SedTask.

    Parameters
    ----------
    doc: :obj:`SedDocument`
        An instance of SedDocument
    task: :obj:`SedTask`
        The task to be executed.
    variable_results: dict
        The variable results of the task. 
        The format of the variable results is {sedVar_id: numpy.ndarray}
        numpy.ndarray is a 1D array of the variable values at each time point.
    base_out_path: str
        The base path to the directory that output files are written to.
    rel_out_path: str
        The relative path to the directory that output files are written to.
         * CSV: directory in which to save outputs to files
            ``{base_out_path}/{rel_out_path}/{report.getId()}.csv``

    report_formats: list, optional
        The formats of the reports to be generated. Default: ['csv']
    
    Raises
    ------
    RuntimeError
        If some generators could not be produced.
    NotImplementedError
        If the output is not of type SedReport.

    Returns
    -------
    dict
        The results of the reports. 
        The format of the results is {sedReport_id: numpy.ndarray}
        numpy.ndarray is a 1D array of the report values at each time point.   
    """

    indent = 0
    listOfOutputs = doc.getListOfOutputs()
    report_results = {}

    task_contributes_to_output = False
        
    for i_output, output in enumerate(listOfOutputs):
        print('{}Generating output {}: `{}` ...'.format(' ' * 2 * (indent + 2), i_output + 1, output.getId()), end='')
        sys.stdout.flush()
        if output.isSedReport ():
            output_result, output_status, output_exception, task_contributes_to_report = exec_report(
                output, variable_results, base_out_path, rel_out_path,  report_formats, task)
            
            print(' {}'.format(output_status))
            if output_exception:
                print('{}'.format(output_exception))
                raise RuntimeError(output_exception)
            
            task_contributes_to_output = task_contributes_to_output or task_contributes_to_report   
        else:
            # only reports are supported for now
            raise NotImplementedError('Outputs of type {} are not supported.'.format(output.getTypeCode ()))
        if  output_result is not None:
            report_results[output.getId()] = output_result
    
    if not task_contributes_to_output:
        print('Task {} does not contribute to any outputs.'.format(task.getId()))

    return report_results

def exec_parameterEstimationTask( doc,task, working_dir,external_variables_info={},external_variables_values=[],ss_time={},cost_type=None):
    """
    Execute a SedTask of type ParameterEstimationTask.
    The model is assumed to be in CellML format.
    The simulation type supported are 'steadyState' and 'timeCourse'.#TODO: add support for steadyState
    The ode solver supported are listed in KISAO_ALGORITHMS (.simulator.py)
    The optimisation algorithm supported are listed in KISAO_ALGORITHMS (.optimiser.py)

    Parameters
    ----------
    doc: :obj:`SedDocument`
        An instance of SedDocument
    task: :obj:`SedParameterEstimationTask `
        The task to be executed.
    working_dir: str
        working directory of the SED document (path relative to which models are located)
    external_variables_info: dict, optional
        The external variables to be specified, in the format of {id:{'component': , 'name': }}
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
    
    fitExperiments,adjustables,adjustableParameters_info,method, opt_parameters, dfDict=get_fit_experiments(doc,task,working_dir,external_variables_info)
    if 'tol' in opt_parameters:
        tol=opt_parameters['tol']
    else:
        tol=1e-8
    if 'maxiter' in opt_parameters:
        maxiter=int(opt_parameters['maxiter']) 
    else:
        maxiter=1000
    
    bounds=Bounds(adjustables[0],adjustables[1])
    initial_value=adjustables[2]
    if method=='global optimization algorithm':
        res= shgo(objective_function, bounds,args=(external_variables_values, fitExperiments, doc, ss_time,cost_type),
                               options={'ftol': tol, 'maxiter': maxiter})
    elif method=='simulated annealing':
        res=dual_annealing(objective_function, bounds,args=(external_variables_values, fitExperiments, doc, ss_time,cost_type),maxiter=maxiter, x0=initial_value)
    elif method=='evolutionary algorithm':
        res=differential_evolution(objective_function, bounds,args=(external_variables_values, fitExperiments, doc, ss_time,cost_type),maxiter=maxiter, tol=tol,x0=initial_value)
    elif method=='random search':
        res=basinhopping(objective_function, initial_value,minimizer_kwargs={'args':(external_variables_values, fitExperiments, doc, ss_time,cost_type)}) # cannot use bounds
    elif method=='local optimization algorithm':
        res=least_squares(objective_function, initial_value, args=(external_variables_values, fitExperiments, doc, ss_time,cost_type), 
                 bounds=bounds, ftol=tol, gtol=tol, xtol=tol, max_nfev=maxiter)
    else:
        raise RuntimeError('Optimisation method not supported!')
    
    i=0
    for parameter in adjustableParameters_info.values():
        print('The estimated value for variable {} in component {} is:'.format(parameter['name'],parameter['component']))
        print(res.x[i])
        i+=1
    print('Values of objective function at the solution: {}'.format(res.fun))
    print('The full optimization result is:')
    print(res)
    return res

def objective_function(param_vals, external_variables_values, fitExperiments, doc, ss_time,cost_type=None):
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
    print('The current parameter values are:',param_vals)
    residuals_sum=0
    sed_results={}
    for fitid,fitExperiment in fitExperiments.items():
        sub_param_vals=[]
        external_variables_info=fitExperiment['external_variables_info']
        cellml_model=fitExperiment['cellml_model']
        analyser=fitExperiment['analyser']
        mtype=fitExperiment['mtype']
        module=fitExperiment['module']
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
                        current_state=sim_UniformTimeCourse(mtype, module, sim_setting, observables, external_module,current_state=None,parameters=parameters)
                        sed_results = copy.deepcopy(current_state[-1])
                    except Exception as exception:
                        print(exception)
                        return 1e12
                else:
                    try:
                      #  current_state=sim_OneStep(mtype, module, sim_setting, observables, external_module,current_state=current_state,parameters=parameters)
                        current_state=sim_UniformTimeCourse(mtype, module, sim_setting, observables, external_module,current_state=current_state,parameters=parameters)
                        for key, value in current_state[-1].items():
                            sed_results[key]=numpy.append(sed_results[key],value)
                    except Exception as exception:
                        print(exception)
                        return 1e12
        else:
            raise RuntimeError('Simulation type not supported!')
        
        residuals={}
        for key, exp_value in observables_exp.items():
            dataGenerator=doc.getDataGenerator(key)
            sim_value=calc_data_generator_results(dataGenerator, sed_results)
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
    print('The current residuals sum is:',residuals_sum)
    return residuals_sum