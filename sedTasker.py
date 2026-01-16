from simulator import sim_UniformTimeCourse,sim_OneStep
from sedReporter import exec_report
import sys

def exec_task(mtype, module, sim_setting, observables, external_module, task_vars,current_state=None):
    """ Execute a SedTask.
    The model is assumed to be in CellML format.
    The simulation type supported are UniformTimeCourse, OneStep and SteadyState.#TODO: add support for SteadyState
    The ode solver supported are listed in KISAO_ALGORITHMS (.simulator.py)

    Parameters
    ----------
    mtype : str
        The type of the model ('ode' or 'algebraic')
    module : module
        The module containing the Python code
    sim_setting : SimSettings
        The simulation settings
    observables : dict
        The observables of the simulation, the format is 
        {id:{'name': , 'component': , 'index': , 'type': }}
    external_module : object
        The External_module_varies object instance for the model
    current_state : tuple
        The current state of the model.
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
    if sim_setting.type=='UniformTimeCourse':
        try:
            current_state=sim_UniformTimeCourse(mtype, module, sim_setting, observables, external_module, current_state,parameters={})
        except RuntimeError as exception:
            print(exception)
            raise RuntimeError(exception)
    elif sim_setting.type=='OneStep':
        try:
            current_state=sim_OneStep(mtype, module, sim_setting, observables, external_module, current_state,parameters={})
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

def report_task(listOfOutputs,task, variable_results, base_out_path, rel_out_path, report_formats =['csv']):
    """ Generate the outputs of a SedTask.

    Parameters
    ----------
    listOfOutputs: list
        The list of outputs of the SED document
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