"""
This file contains functions that are modified versions of functions from https://github.com/biosimulators/Biosimulators_utils.git.
The original code is licensed under the MIT license.

Original source: https://github.com/biosimulators/Biosimulators_utils/blob/dev/biosimulators_utils/sedml/utils.py

The MIT License (MIT)

Copyright (c) 2020, Center for Reproducible Biomedical Modeling
"""
import requests
import os
import pandas
import tempfile
import re
from sedChanger import resolve_model_and_apply_xml_changes,resolve_model
from simulator import  SimSettings,load_module, get_externals_varies,get_observables
from analyser import parse_model,analyse_model_full,get_mtype,resolve_imports
from coder import writePythonCode,writeCellML
import copy
import numpy as np
from libsedml import readSedMLFromFile
import libsedml
import shutil

CELLML2NAMESPACE ={"cellml":"http://www.cellml.org/cellml/2.0#"}
KISAO_ALGORITHMS = {'KISAO:0000030': 'Euler forward method',
                    'KISAO:0000535': 'VODE',
                    'KISAO:0000088': 'LSODA',
                    'KISAO:0000087': 'dopri5',
                    'KISAO:0000436': 'dop853',
                    'KISAO:0000019': 'CVODE',
                    }

# https://docs.scipy.org/doc/scipy/reference/optimize.html
SCIPY_OPTIMIZE_LOCAL = ['Nelder-Mead','Powell','CG','BFGS','Newton-CG','L-BFGS-B','TNC','COBYLA','SLSQP','trust-constr','dogleg','trust-ncg','trust-exact','trust-krylov']
KISAO_ALGORITHMS_OPT = {'KISAO:0000514': 'Nelder-Mead',
                    'KISAO:0000472': 'global optimization algorithm', # shgo
                    'KISAO:0000471': 'local optimization algorithm', # least_squares
                    'KISAO:0000503': 'simulated annealing',	 # dual_annealing
                    'KISAO:0000520': 'evolutionary algorithm', # differential_evolution
                    'KISAO:0000504': 'random search', # basinhopping
                    }

def read_sedml(file_name):
    """
    Read the SED-ML document from a file
    
    Parameters
    ----------
    file_name: str
        The full path of the file

    Raises
    ------
    FileNotFoundError
        If the file is not found.

    Returns
    -------
    SedDocument
        An instance of SedDocument.
    """

    if not os.path.exists(file_name):
        raise FileNotFoundError('The file {0} is not found.'.format(file_name))
    doc = readSedMLFromFile(file_name)
    return doc

def get_variable_info_CellML(task_variables,model_etree):
    """ Get information about the variables of a task
    Args:
        task_variables (:obj:`list` of :obj:`SedVariable`): variables of a task
        model_etree (:obj:`etree._Element`): model encoded in XML
    Raises:
        :obj:`ValueError`: if a variable is not found
    Returns:
        :obj:`dict`: information about the variables in the format {variable.id: {'name': variable name, 'component': component name}}
    """
    variable_info={}
    for v in task_variables:
        if v.getTarget().rpartition('/@')[-1]=='initial_value': # target initial value instead of variable
            vtemp=v.getTarget().rpartition('/@')
            try:
                variable_element = model_etree.xpath(vtemp[0],namespaces=CELLML2NAMESPACE)[0]
            except:
                raise ValueError('The variable {} is not found!'.format(v.getTarget ()))
        else:
            try:
                variable_element = model_etree.xpath(v.getTarget (),namespaces=CELLML2NAMESPACE)[0]
            except:
                raise ValueError('The variable {} is not found!'.format(v.getTarget ()))
        if variable_element is False:
            raise ValueError('The variable {} is not found!'.format(v.getTarget ()))
        else:
            variable_info[v.getId()] = {
                'name': variable_element.attrib['name'],
                'component': variable_element.getparent().attrib['name']
            }
    return variable_info

def get_KISAO_parameters_opt(algorithm):
    """Get the parameters of the KISAO algorithm.
    Args:
        algorithm (:obj:`dict`): the algorithm of the optimization, 
        the format is {'kisaoID': , 'name': 'optional,Euler forward method' , 
        'listOfAlgorithmParameters':[dict_algorithmParameter] }
        dict_algorithmParameter={'kisaoID':'KISAO:0000483','value':'0.001'}
    Returns:
        :obj:`tuple`:
            * :obj:`str` or None: the method of the optimization algorithm
            * :obj:`dict` or None: the parameters of the integrator, the format is {parameter: value}
    """
    opt_parameters = {}
    if algorithm['kisaoID'] == 'KISAO:0000514': 
        method = KISAO_ALGORITHMS_OPT[algorithm['kisaoID']]       
        for p in algorithm['listOfAlgorithmParameters']:
            if p['kisaoID'] == 'KISAO:0000211':
                opt_parameters['xatol'] = float(p['value'])
            elif p['kisaoID'] == 'KISAO:0000486':
                opt_parameters['maxiter'] = int(p['value'])
            elif p['kisaoID'] == 'KISAO:0000597':
                opt_parameters['tol'] = float(p['value'])

        return method, opt_parameters
    elif algorithm['kisaoID'] == 'KISAO:0000520':
        method = KISAO_ALGORITHMS_OPT[algorithm['kisaoID']]
        for p in algorithm['listOfAlgorithmParameters']:
            if p['kisaoID'] == 'KISAO:0000517':
                opt_parameters['maxiter'] = int(p['value'])
            elif p['kisaoID'] == 'KISAO:0000597':
                opt_parameters['tol'] = float(p['value'])
            elif p['kisaoID'] == 'KISAO:0000519':
                opt_parameters['popsize'] = int(p['value'])
            elif p['kisaoID'] == 'KISAO:0000529':
                opt_parameters['workers'] = int(p['value'])
        return method, opt_parameters

    elif algorithm['kisaoID'] == 'KISAO:0000472':
        method = KISAO_ALGORITHMS_OPT[algorithm['kisaoID']]
        for p in algorithm['listOfAlgorithmParameters']:
            if p['kisaoID'] == 'KISAO:0000486':
                opt_parameters['maxiter'] = int(p['value'])
            elif p['kisaoID'] == 'KISAO:0000597':
                opt_parameters['tol'] = float(p['value'])
            elif p['kisaoID'] == 'KISAO:0000529':
                opt_parameters['workers'] = int(p['value'])
        return method, opt_parameters
    
    elif algorithm['kisaoID'] in KISAO_ALGORITHMS_OPT.keys():
        method = KISAO_ALGORITHMS_OPT[algorithm['kisaoID']]
        for p in algorithm['listOfAlgorithmParameters']:
            if p['kisaoID'] == 'KISAO:0000486':
                opt_parameters['maxiter'] = float(p['value'])
            elif p['kisaoID'] == 'KISAO:0000597':
                opt_parameters['tol'] = float(p['value'])
        return method, opt_parameters
    else:
        print("The algorithm {} is not supported!".format(algorithm['kisaoID']))
        return None, opt_parameters

def get_dict_algorithm(sed_algorithm):
    """Get the information of an algorithm
    
    Parameters
    ----------
    sed_algorithm: SedAlgorithm
        An instance of SedAlgorithm.
    
    Raises
    ------
    ValueError
        If the kisaoID attribute of an algorithm is not set.
        If _get_dict_algorithmParameter(sed_algorithmParameter) failed.

    Notes
    -----
    Assume that the algorithm parameters do not have any algorithm parameters.

    Returns
    -------
    dict
        The dictionary format: {'kisaoID':'KISAO:0000030','name':'optional,e.g,time course simulation over 100 minutes', 'listOfAlgorithmParameters':[dict_algorithmParameter]}
        Only the attributes that are set will be returned.
    """
    def _get_dict_algorithmParameter(sed_algorithmParameter):
        """Get the information of an algorithm parameter

        Parameters
        ----------
        sed_algorithmParameter: SedAlgorithmParameter
            An instance of SedAlgorithmParameter.

        Raises
        ------
        ValueError
            If the kisaoID and value attributes of an algorithm parameter are not set.

        Returns
        -------
        dict
            The dictionary format: {'kisaoID':'KISAO:0000019','value':'1.0','name':'optional, describe the meaning of the param',}
            Only the attributes that are set will be returned.
        """
        dict_algorithmParameter = {}
        if sed_algorithmParameter.isSetName():
            dict_algorithmParameter['name'] = sed_algorithmParameter.getName()
        if sed_algorithmParameter.isSetKisaoID() and sed_algorithmParameter.isSetValue():
            dict_algorithmParameter['kisaoID'] = sed_algorithmParameter.getKisaoID()
            dict_algorithmParameter['value'] = sed_algorithmParameter.getValue()
        else:
            raise ValueError('The kisaoID and value attributes of an algorithm parameter are required.')
        return dict_algorithmParameter

    dict_algorithm = {}
    if not sed_algorithm.isSetKisaoID():
        raise ValueError('The kisaoID attribute of an algorithm is required.')
    dict_algorithm['kisaoID'] = sed_algorithm.getKisaoID()
    if sed_algorithm.isSetName():
        dict_algorithm['name'] = sed_algorithm.getName()
    if sed_algorithm.getNumAlgorithmParameters()>0:
        dict_algorithm['listOfAlgorithmParameters'] = []
        for sed_algorithmParameter in sed_algorithm.getListOfAlgorithmParameters():
            try:
                dict_algorithmParameter = _get_dict_algorithmParameter(sed_algorithmParameter)
            except ValueError as e:
                raise
            dict_algorithm['listOfAlgorithmParameters'].append(dict_algorithmParameter)
    return dict_algorithm

def get_KISAO_parameters(algorithm):
    """Get the parameters of the KISAO algorithm.
    
    Parameters
    ----------
    algorithm : dict
        The dictionary of the KISAO algorithm
        Format: {'kisaoID': , 'name': , 'listOfAlgorithmParameters': [{'kisaoID': , 'name': , 'value': }]}
    
    Raises
    ------
    ValueError
        If the algorithm is not supported
        
    Returns
    -------
    method : str
        The method of the integration. 
        Now the supported methods are 'Euler forward method', 'VODE', 'LSODA', 'dopri5' and 'dop853'.
        None if the method is not supported.
    integrator_parameters : dict
        The parameters of the integrator
        None if the method is not supported.
    """
    method = KISAO_ALGORITHMS[algorithm['kisaoID']]
    integrator_parameters = {}
    if algorithm['kisaoID'] == 'KISAO:0000030':
        # Euler forward method
        if 'listOfAlgorithmParameters' in algorithm:
            for p in algorithm['listOfAlgorithmParameters']:
                if p['kisaoID'] == 'KISAO:0000483':
                    integrator_parameters['step_size'] = float(p['value'])
                else:
                    raise ValueError('The algorithm parameter {} is not supported for the Euler forward method!'.format(p['kisaoID']))

    elif algorithm['kisaoID'] == 'KISAO:0000535':
            # VODE            
            if 'listOfAlgorithmParameters' in algorithm:
                for p in algorithm['listOfAlgorithmParameters']:
                    if p['kisaoID'] == 'KISAO:0000209':
                        integrator_parameters['rtol'] = float(p['value'])
                    elif p['kisaoID'] == 'KISAO:0000211':
                        integrator_parameters['atol'] = float(p['value'])
                    elif p['kisaoID'] == 'KISAO:0000475': 
                        integrator_parameters['method'] = p['value'] # ‘adams’ or ‘bdf’
                    elif p['kisaoID'] == 'KISAO:0000415':
                        integrator_parameters['nsteps'] = int(p['value'])
                    elif p['kisaoID'] == 'KISAO:0000467':
                        integrator_parameters['max_step'] = float(p['value'])
                    elif p['kisaoID'] == 'KISAO:0000485':
                        integrator_parameters['min_step'] = float(p['value'])
                    elif p['kisaoID'] == 'KISAO:0000484':
                        integrator_parameters['order'] = int(p['value'])
                    else:
                        raise ValueError('The algorithm parameter {} is not supported for the VODE method!'.format(p['kisaoID']))
    elif algorithm['kisaoID'] == 'KISAO:0000088':
        # LSODA
        if 'listOfAlgorithmParameters' in algorithm:
            for p in algorithm['listOfAlgorithmParameters']:
                if p['kisaoID'] == 'KISAO:0000209':
                    integrator_parameters['rtol'] = float(p['value'])
                elif p['kisaoID'] == 'KISAO:0000211':
                    integrator_parameters['atol'] = float(p['value'])
                elif p['kisaoID'] == 'KISAO:0000415':
                    integrator_parameters['nsteps'] = int(p['value'])
                elif p['kisaoID'] == 'KISAO:0000467':
                    integrator_parameters['max_step'] = float(p['value'])
                elif p['kisaoID'] == 'KISAO:0000485':
                    integrator_parameters['min_step'] = float(p['value'])
                elif p['kisaoID'] == 'KISAO:0000219':
                    integrator_parameters['max_order_ns'] =int(p['value'])
                elif p['kisaoID'] == 'KISAO:0000220':
                    integrator_parameters['max_order_s'] = int(p['value'])
                else:
                    raise ValueError('The algorithm parameter {} is not supported for the LSODA method!'.format(p['kisaoID']))
    elif algorithm['kisaoID'] == 'KISAO:0000087':
        # dopri5
        if 'listOfAlgorithmParameters' in algorithm:
            for p in algorithm['listOfAlgorithmParameters']:
                if p['kisaoID'] == 'KISAO:0000209':
                    integrator_parameters['rtol'] = float(p['value'])
                elif p['kisaoID'] == 'KISAO:0000211':
                    integrator_parameters['atol'] = float(p['value'])
                elif p['kisaoID'] == 'KISAO:0000415':
                    integrator_parameters['nsteps'] = int(p['value'])
                elif p['kisaoID'] == 'KISAO:0000467':
                    integrator_parameters['max_step'] = float(p['value'])
                elif p['kisaoID'] == 'KISAO:0000541':
                    integrator_parameters['beta'] = float(p['value'])
                else:
                    raise ValueError('The algorithm parameter {} is not supported for the dopri5 method!'.format(p['kisaoID']))
    elif algorithm['kisaoID'] == 'KISAO:0000436':
        # dop853
        if 'listOfAlgorithmParameters' in algorithm:
            for p in algorithm['listOfAlgorithmParameters']:
                if p['kisaoID'] == 'KISAO:0000209':
                    integrator_parameters['rtol'] = float(p['value'])
                elif p['kisaoID'] == 'KISAO:0000211':
                    integrator_parameters['atol'] = float(p['value'])
                elif p['kisaoID'] == 'KISAO:0000415':
                    integrator_parameters['nsteps'] = int(p['value'])
                elif p['kisaoID'] == 'KISAO:0000467':
                    integrator_parameters['max_step'] = float(p['value'])
                elif p['kisaoID'] == 'KISAO:0000541':
                    integrator_parameters['beta'] = float(p['value'])
                else:
                    raise ValueError('The algorithm parameter {} is not supported for the dop853 method!'.format(p['kisaoID']))
    elif algorithm['kisaoID'] == 'KISAO:0000019':
        # CVODE
        if 'listOfAlgorithmParameters' in algorithm:
            for p in algorithm['listOfAlgorithmParameters']:
                if p['kisaoID'] == 'KISAO:0000209':
                    integrator_parameters['rtol'] = float(p['value'])
                elif p['kisaoID'] == 'KISAO:0000211':
                    integrator_parameters['atol'] = float(p['value'])
                elif p['kisaoID'] == 'KISAO:0000415':
                    integrator_parameters['max_num_steps'] = int(p['value'])
                elif p['kisaoID'] == 'KISAO:0000467':
                    integrator_parameters['max_step'] = float(p['value'])
                elif p['kisaoID'] == 'KISAO:0000485':
                    integrator_parameters['min_step'] = float(p['value'])
                elif p['kisaoID'] == 'KISAO:0000475': # multistep method: BDF or Adams
                    integrator_parameters['method'] = p['value']
                elif p['kisaoID'] == 'KISAO:0000477': # Choice of linear solver, defaults to ‘dense’. ‘band’ requires both ‘lband’ and ‘uband’. 
                    integrator_parameters['linsolver'] = p['value']
                elif p['kisaoID'] == 'KISAO:0000665': # Specifies the maximum number of nonlinear solver iterations in one step. The default is 3.. 
                    integrator_parameters['max_nonlin_iters'] = int(p['value'])
                else:
                    raise ValueError('The algorithm parameter {} is not supported for the CVODE method!'.format(p['kisaoID']))
    else:
        print("The algorithm {} is not supported!".format(algorithm['kisaoID']))
        raise ValueError("The algorithm {} is not supported!".format(algorithm['kisaoID']))
    
    return method, integrator_parameters

def getSimSettingFromDict(dict_simulation):
    """Get the simulation settings from the dictionary of the simulation.

    Parameters
    ----------
    dict_simulation : dict
        The dictionary of the simulation
        If the type is 'OneStep', the format is {'type': 'OneStep', 'step': 0.1}
        If the type is 'UniformTimeCourse', the format is {'type': 'UniformTimeCourse',
        'initialTime': 0.0, 'outputStartTime': 0.0, 'outputEndTime': 10.0, 'numberOfSteps': 100}
        If the type is 'SteadyState', the format is {'type': 'SteadyState'}
    
    Raises
    ------
    ValueError
        If the type is not supported
        
    Returns
    -------
    :obj:`SimSettings`: the simulation settings
    """

    simSetting=SimSettings()
    simSetting.type=dict_simulation['type']
    if simSetting.type=='OneStep':
        simSetting.step=dict_simulation['step']
    elif simSetting.type=='UniformTimeCourse':
        simSetting.initial_time=dict_simulation['initialTime']
        simSetting.output_start_time=dict_simulation['outputStartTime']
        simSetting.output_end_time=dict_simulation['outputEndTime']
        simSetting.number_of_steps=dict_simulation['numberOfSteps']
    elif simSetting.type=='SteadyState' or simSetting.type=='steadyState':
        pass
    elif simSetting.type=='timeCourse':
        pass
    else:
        print('The simulation type {} is not supported!'.format(simSetting.type))
        raise ValueError('The simulation type {} is not supported!'.format(simSetting.type))    
    simSetting.method, simSetting.integrator_parameters=get_KISAO_parameters(dict_simulation['algorithm'])
    
    return simSetting
def get_dict_uniformTimeCourse(sim):
    """
    Get the information of a uniform time course simulation

    Parameters
    ----------
    sim: SedUniformTimeCourse
        An instance of SedUniformTimeCourse.
    
    Raises
    ------
    ValueError
        If get_dict_algorithm(sed_algorithm) failed.

    Notes
    -----
    Assume the simulation has been created successfully.

    Returns
    -------
    dict
        The dictionary format:
        {'id':'timeCourse1', 'type': 'UniformTimeCourse','algorithm':dict_algorithm, 'initialTime':0.0,'outputStartTime':0.0,'outputEndTime':10.0,'numberOfSteps':1000}.
   
    """
    dict_uniformTimeCourse = {}
    dict_uniformTimeCourse['id'] = sim.getId()
    dict_uniformTimeCourse['type'] = libsedml.SedTypeCode_toString(sim.getTypeCode())
    dict_uniformTimeCourse['initialTime'] = sim.getInitialTime()
    dict_uniformTimeCourse['outputStartTime'] = sim.getOutputStartTime()
    dict_uniformTimeCourse['outputEndTime'] = sim.getOutputEndTime()
    dict_uniformTimeCourse['numberOfSteps'] = sim.getNumberOfPoints()
    sed_algorithm = sim.getAlgorithm()
    try:
        dict_algorithm = get_dict_algorithm(sed_algorithm)
    except ValueError as e:
        raise
    dict_uniformTimeCourse['algorithm']=dict_algorithm
    return dict_uniformTimeCourse

def get_dict_oneStep(sim):
    """
    Get the information of a one step simulation
    
    Parameters
    ----------
    sim: SedOneStep
        An instance of SedOneStep.
    
    Raises
    ------
    ValueError
        If get_dict_algorithm(sed_algorithm) failed.

    Notes
    -----
    Assume the simulation has been created successfully.

    Returns
    -------
    dict or bool
        The dictionary format: {'id':'oneStep1','type':'OneStep', 'step':0.1,'algorithm':dict_algorithm}
        If the required attributes are not set, return False.    
    """

    dict_oneStep = {}
    dict_oneStep['id'] = sim.getId()
    dict_oneStep['type'] = libsedml.SedTypeCode_toString(sim.getTypeCode())
    dict_oneStep['step'] = sim.getStep()
    sed_algorithm = sim.getAlgorithm()
    try:
        dict_algorithm = get_dict_algorithm(sed_algorithm)
    except ValueError as e:
        raise  
    dict_oneStep['algorithm']=dict_algorithm
    return dict_oneStep
def get_dict_steadyState(sim):
    """
    Get the information of a steady state simulation
    
    Parameters
    ----------
    sim: SedSteadyState
        An instance of SedSteadyState.
    
    Raises
    ------
    ValueError
        If get_dict_algorithm(sed_algorithm) failed.

    Notes
    -----
    Assume the simulation has been created successfully.

    Returns
    -------
    dict
        The dictionary format: {'id':'steadyState1','type':'SteadyState', 'algorithm':dict_algorithm}
    """
    dict_steadyState = {}
    dict_steadyState['id'] = sim.getId()
    dict_steadyState['type'] = libsedml.SedTypeCode_toString(sim.getTypeCode())
    sed_algorithm = sim.getAlgorithm()
    try:     
        dict_algorithm = get_dict_algorithm(sed_algorithm)
    except ValueError as e:
        raise
    dict_steadyState['algorithm'] = dict_algorithm
    return dict_steadyState

def get_dict_simulation(sim):
    """
    Get the information of a simulation

    Parameters
    ----------
    sim: SedSimulation
        An instance of SedSimulation.
    
    Raises
    ------
    ValueError
        If the simulation type is not defined.
        If get_dict_uniformTimeCourse(sim), get_dict_oneStep(sim) or get_dict_steadyState(sim) failed.
    
    Notes
    -----
    Assume the simulation has been created successfully.

    Returns
    -------
    dict
        If the simulation type is UniformTimeCourse, the dictionary format:
        {'id':'timeCourse1', 'type': 'UniformTimeCourse','algorithm':dict_algorithm, 'initialTime':0.0,'outputStartTime':0.0,'outputEndTime':10.0,'numberOfSteps':1000}
        If the simulation type is OneStep, the dictionary format:
        {'id':'oneStep1','type':'OneStep', 'step':0.1,'algorithm':dict_algorithm}
        If the simulation type is SteadyState, the dictionary format:
        {'id':'steadyState1','type':'SteadyState', 'algorithm':dict_algorithm}
    """
    
    if not sim.isSedUniformTimeCourse() and not sim.isSedOneStep() and not sim.isSedSteadyState():
        raise ValueError('The simulation type is not defined.')
    try:
        if sim.isSedUniformTimeCourse():
            dict_simulation = get_dict_uniformTimeCourse(sim)
        elif sim.isSedOneStep():
            dict_simulation = get_dict_oneStep(sim)
        elif sim.isSedSteadyState():
            dict_simulation = get_dict_steadyState(sim)
    except ValueError as e:
        raise
    return dict_simulation

def getSimSettingFromSedSim(sedSimulation):
    """Get the simulation settings from the sedSimulation.

    Parameters
    ----------
    sedSimulation : SedSimulation
        The sedSimulation object

    Raises
    ------
    ValueError
        If the dict_simulation is not valid
        If the simulation type is not supported

    Returns
    -------
    :obj:`SimSettings`: the simulation settings
    """

    try:
        dict_simulation = get_dict_simulation(sedSimulation)
        simSetting = getSimSettingFromDict(dict_simulation)
    except ValueError as e:
        raise ValueError(str(e)) from e
    return simSetting

def get_variables_for_task(doc, task):
    """ Get the variables that a task must record

    Args:
        doc (:obj:`SedDocument`): SED document
        task (:obj:`SedTask`): task

    Returns:
        :obj:`list` of :obj:`SedVariable`: variables that task must record
    """
    data_generators = set()
    for i in range(doc.getNumOutputs()):
        output=doc.getOutput(i)
        data_generators.update(get_data_generators_for_output(output))

    sedDataGenerators = [output.getSedDocument().getDataGenerator(data_generator_name) for data_generator_name in data_generators]
    if None in sedDataGenerators:
        sedDataGenerators.remove(None)

    variables = get_variables_for_data_generators(sedDataGenerators)
    variables_task=[]
    for variable in variables:
        task_name = variable.getTaskReference ()
        if task_name == task.getId ():
            variables_task.append(variable)
    return variables_task

def get_data_generators_for_output(output):
    """ Get the data generators involved in an output, TODO: only SED reports are supported

    Args:
        output (:obj:`SedOutput`): report or plot

    Returns:
        :obj:`set` of :obj:`string`: data generator ids involved in the output
    """
    data_generators = set()
    if output.isSedReport ():
        for sedDataSet in output.getListOfDataSets ():
            data_generator_name=sedDataSet.getDataReference()
            data_generators.add(data_generator_name)
    else:
        print("Only SED reports are supported.")

    return data_generators

def get_variables_for_data_generators(data_generators):
    """ Get the variables involved in a collection of generators

    Args:
        data_generators (:obj:`list` of :obj:`SedDataGenerator`): data generators

    Returns:
        :obj:`set` of :obj:`SedVariables`: variables id involved in the data generators
    """
    variables = set()
    sedVariables=[]    
    for data_gen in data_generators:
        for sedVariable in data_gen.getListOfVariables ():
            if sedVariable.getId () not in variables:
                sedVariables.append(sedVariable)
            variables.add(sedVariable.getId ())
    return sedVariables

def get_variables_for_data_generator(data_generator):
    """ Get the variables involved in a collection of generators

    Args:
        data_generators (:obj:`SedDataGenerator`): data generator

    Returns:
        :obj:`set` of :obj:`SedVariables`: variables id involved in the data generators
    """
    variables = set()
    sedVariables=[]    
    for sedVariable in data_generator.getListOfVariables ():
        if sedVariable.getId () not in variables:
            sedVariables.append(sedVariable)
        variables.add(sedVariable.getId ())
    return sedVariables 

def get_model_changes_for_task(task):
    """ Get the changes to models for a task

    Args:
        task (:obj:`Task`): task

    Returns:
        :obj:`list` of :obj:`ModelChange`: changes to the model
    """
    doc=task.getSedDocument()
    if task.isSedTask():
        return []
    elif task.isSedRepeatedTask():
        changes = [task.getTaskChange(i) for i in range(task.getNumTaskChanges())]

        for sub_task in [task.getSubTask(i) for i in range(task.getNumSubTasks())]:
            itask = sub_task.getTask()           
            changes.extend(get_model_changes_for_task(doc.getTask(itask)))
        return changes
    else:
        print("Only SED tasks are supported.")
        return []   

def get_models_referenced_by_task(doc,task):
    """ Get the models referenced from a task
    
    Parameters
    ----------
    doc: :obj:`SedDocument`
        An instance of SedDocument
    task: :obj:`SedTask`
        An instance of SedTask
    
    Raises
    ------
    ValueError
        If the task is not of type SedTask or SedRepeatedTask or SedParameterEstimationTask
        
    Returns
    -------
    :obj:`list` of :obj:`SedModel`
        A list of SedModel objects
    """
 
    if task.isSedTask ():
        models = set()
        if task.isSetModelReference ():
            models.add(task.getModelReference ())
        sedModels=[doc.getModel(modelReference) for modelReference in models]
        return sedModels

    elif task.isSedRepeatedTask () :
        models = set()
        for change in task.getListOfTaskChanges ():
            models.update(get_models_referenced_by_setValue(change))

        for sub_task in task.getListOfSubTasks ():
            itask = doc.getTask(sub_task.getTask ())
            models.update(get_models_referenced_by_task(doc,itask))
            for change in sub_task.getListOfTaskChanges (): # newly added in Level 4
                models.update(get_models_referenced_by_setValue(change)) 

        if task.isSetRangeId (): # TODO: check if this is already covered by getListOfRanges
            irange = task.getRange(task.getRangeId ())
            models.update(get_models_referenced_by_range(task,irange))

        for range in  task.getListOfRanges ():
            models.update(get_models_referenced_by_range(task,range))
        
        sedModels=[doc.getModel(modelReference) for modelReference in models]
        return sedModels
    
    elif task.isSedParameterEstimationTask ():
        models = set()
        for adjustableParameter in task.getListOfAdjustableParameters ():
            models.update(get_models_referenced_by_adjustableParameter(adjustableParameter)) 

        sedModels=[doc.getModel(modelReference) for modelReference in models]
        return sedModels        
    else:
        raise ValueError('Tasks of type `{}` are not supported.'.format(task.getTypeCode ()))
    
def get_models_referenced_by_range(task,range):
    """ Get the models referenced by a range

    Args:
        range (:obj:`SedRange`): range

    Returns:
        :obj:`set` of :obj:`SedModel`: models
    """
    models = set()
    if range.isSedFunctionalRange () and range.getListOfVariables ():
        models.update(get_models_referenced_by_listOfVariables(range.getListOfVariables ()))
    if range.isSetRange ():
        irange=task.getRange(range.getRange ())
        models.update(get_models_referenced_by_range(task,irange))
    return models

def get_models_referenced_by_setValue(task,setValue):
    """ Get the models referenced by a setValue

    Args:
        task (:obj:`SedRepeatedTask `): SedRepeatedTask 
        setValue (:obj:`SedSetValue`): setValue

    Returns:
        :obj:`set` of :obj:`SedModel`: models
    """
    
    models = set()
    models.add(setValue.getModelReference ())
    if setValue.isSetRange ():
        irange=task.getRange(setValue.getRange ())
        models.update(get_models_referenced_by_range(irange))
    if setValue.getListOfVariables ():
        models.update(get_models_referenced_by_listOfVariables(setValue.getListOfVariables))
    return models

def get_models_referenced_by_computedChange(change):
    """
    Get the models referenced by a computedChange

    Args:
        change (:obj:`SedComputedChange`): computedChange
        
    Returns:
        :obj:`set` of :obj:`SedModel`: models
    """
    models = set()  
    if change.getListOfVariables ():
        models.update(get_models_referenced_by_listOfVariables(change.getListOfVariables))
    return models

def get_models_referenced_by_listOfVariables(listOfVariables):
    """
    Get the models referenced by a list of sedVariables

    Args:
        listOfVariables (:obj:`list` of obj: ` sedVariables` )
    
    Returns:
        :obj:`set` of :obj:`SedModel`: models
    
    """
    models = set()
    for variable in listOfVariables:
        if variable.isSetModelReference ():
            models.add(variable.getModelReference ())
    return models

def get_models_referenced_by_adjustableParameter(adjustableParameter):
    """
    Note: will depreciate in future versions once the issue https://github.com/fbergmann/libSEDML/issues/172 is fixed
    """
    models = set()
    if adjustableParameter.isSetModelReference ():
        models.add(adjustableParameter.getModelReference ())
    return models

def get_df_from_dataDescription(dataDescription, working_dir):
    """
    Return a pandas.DataFrame from a dataDescription.
    Assume the data source file is a csv file.

    Parameters
    ----------
    dataDescription: :obj:`SedDataDescription`
        An instance of SedDataDescription
    working_dir: :obj:`str`
        working directory of the SED document (path relative to which data source files are located)

    Raises
    ------
    FileNotFoundError
        If the data source file does not exist

    Returns
    -------
    :obj:`pandas.DataFrame`
        A pandas.DataFrame object
    
    """

    source = dataDescription.getSource ()
    if re.match(r'^http(s)?://', source, re.IGNORECASE):
        response = requests.get(source)
        try:
            response.raise_for_status()
        except Exception:
            raise FileNotFoundError('Model could not be downloaded from `{}`.'.format(source))

        temp_file, temp_data_source = tempfile.mkstemp()
        os.close(temp_file)
        with open(temp_data_source, 'wb') as file:
            file.write(response.content)
        filename = temp_data_source
    else:
        if os.path.isabs(source):
            filename = source
        else:
            filename = os.path.join(working_dir, source)

        if not os.path.isfile(os.path.join(working_dir, source)):
            raise FileNotFoundError('Data source file `{}` does not exist.'.format(source))
    
    df = pandas.read_csv(filename, skipinitialspace=True,encoding='utf-8')

    return df

def get_value_of_dataSource(doc, dataSourceID,dfDict):
    """
    Return a numpy.ndarray from a data source.
    Assume 2D dimensionDescription and 2D data source.

    Parameters
    ----------
    doc: :obj:`SedDocument`
        An instance of SedDocument
    dataSourceID: :obj:`str`
        The id of the data source
    dfDict: :obj:`dict` of :obj:`pandas.DataFrame`
        A dictionary of pandas.DataFrame objects
        The format is {dataDescription.getId(): pandas.DataFrame}

    Raises
    ------
    ValueError
        If the data source is not defined
        If the data source isSetIndexSet ()
        If the slice number is bigger than 2

    Returns
    -------
    :obj:`numpy.ndarray`
        A numpy.ndarray object
    """
    
    dim1_value=None
    dim2_value=None
    dim1_present=False
    dim2_present=False
    dim1_startIndex=None
    dim1_endIndex=None
    dim2_startIndex=None
    dim2_endIndex=None
    for dataDescription in doc.getListOfDataDescriptions ():
        for dataSource in dataDescription.getListOfDataSources ():
            if dataSource.getId () == dataSourceID: # expect only one data source
                df=dfDict[dataDescription.getId()]
                dimensionDescription=dataDescription.getDimensionDescription ()
                dim1_Description=dimensionDescription.get(0)
                dim1_index=dim1_Description.getId() 
                dim2_Description=dim1_Description.get(0)
                dim2_index=dim2_Description.getId()             
                if dataSource.isSetIndexSet ():# expect only using slice
                    raise ValueError('IndexSet is not supported.')
                else:
                    for sedSlice in dataSource.getListOfSlices (): # up to two slices supported
                        if sedSlice.getReference ()==dim1_index:
                            dim1_present=True
                            if sedSlice.isSetValue ():
                                dim1_value=sedSlice.getValue ()
                            if sedSlice.isSetStartIndex ():
                                dim1_startIndex=sedSlice.getStartIndex ()
                            if sedSlice.isSetEndIndex ():
                                dim1_endIndex=sedSlice.getEndIndex ()
                        elif sedSlice.getReference ()==dim2_index:
                            dim2_present=True
                            if sedSlice.isSetValue (): 
                                dim2_value=sedSlice.getValue ()
                            if sedSlice.isSetStartIndex ():
                                dim2_startIndex=sedSlice.getStartIndex ()
                            if sedSlice.isSetEndIndex ():
                                dim2_endIndex=sedSlice.getEndIndex ()
                        else:
                            raise ValueError('up to two slices supported')

                    if dim1_present and (not dim2_present): 
                        # get the value(s) at index=dim1_value or all values if dim1_value is not set then subdivide the values according to startIndex and endIndex
                        # TODO: need to check if the understanding of the slice is correct   
                        if dim1_value is not None:
                            value=df.iloc[[dim1_value]]
                        elif dim1_startIndex is not None and dim1_endIndex is not None:
                            value=df.iloc[dim1_startIndex:(dim1_endIndex+1)]
                        elif dim1_startIndex is not None and (dim1_endIndex is None):
                            value=df.iloc[dim1_startIndex:]
                        elif (dim1_startIndex is None) and dim1_endIndex is not None:
                            value=df.iloc[:(dim1_endIndex+1)]
                        else:
                            value=df
                        return value.to_numpy()
                    
                    elif dim2_present and (not dim1_present):
                        # get the value(s) of the column and then subdivide the values according to startIndex and endIndex
                        if dim2_value:
                            columnName=dim2_value
                            df_selected=df[columnName]
                            if dim2_startIndex is not None and dim2_endIndex is not None:
                                value=df_selected.iloc[dim2_startIndex:(dim2_endIndex+1)]
                            elif dim2_startIndex is not None and (dim2_endIndex is None):
                                value=df_selected.iloc[dim2_startIndex:]
                            elif (dim2_startIndex is None) and dim2_endIndex is not None:
                                value=df_selected.iloc[:(dim2_endIndex+1)]
                            else:
                                value=df_selected
                            return value.to_numpy()
                        
                    elif dim1_present and dim2_present:
                        # get a single value at index=dim1_value and column=dim2_value
                        columnName=dim2_value
                        df_selected=df[columnName]
                        if dim1_value is not None:
                            df_selected=df_selected.iloc[[float(dim1_value)]]
                        return df_selected.to_numpy()
                    else:
                        raise ValueError('Data source `{}` is not defined.'.format(dataSourceID))

def get_adjustableParameters(model_etree,task):
    """
    Return a tuple containing adjustable parameter information.
    Assume the model is a CellML model.
    
    Parameters
    ----------
    model_etree: :obj:`lxml.etree._ElementTree`
        An instance of lxml.etree._ElementTree
    task: :obj:`SedParameterEstimationTask`

    Raises
    ------
    ValueError
        If the variable is not found in the model

    Returns
    -------
    tuple
        A tuple containing adjustable parameter information
        adjustableParameters_info: dict, {index: {'component':component,'name':name}}
        experimentReferences: dict, {index: [experimentId]}
        lowerBound: list, [lowerBound]
        upperBound: list, [upperBound]
        initial_value: list, [initial_value]  
    """
    adjustableParameters_info={}
    experimentReferences={}
    lowerBound=[]
    upperBound=[]
    initial_value=[]
    for i in range(len(task.getListOfAdjustableParameters())):
        adjustableParameter=task.getAdjustableParameter(i)
        try:
            variables_info = get_variable_info_CellML([adjustableParameter],model_etree)
        except ValueError as exception:
            print('Error in get_variable_info_CellML:',exception)
            raise exception
        for key,variable_info in variables_info.items(): # should be only one variable
            adjustableParameters_info[i]={'component':variable_info['component'],'name': variable_info['name']}

        bonds=adjustableParameter.getBounds()
        lowerBound.append(bonds.getLowerBound ())
        upperBound.append(bonds.getUpperBound ())
        if adjustableParameter.isSetInitialValue ():
            initial_value.append(adjustableParameter.getInitialValue())
        else:
            initial_value.append(bonds.getLowerBound ())
        experimentReferences[i]=[]
        if adjustableParameter.getNumExperimentReferences()>0:
            for experiment in adjustableParameter.getListOfExperimentReferences ():
                experimentReferences[i].append(experiment.getExperimentId () )
        else:
            for experiment in task.getListOfExperiments():
                experimentReferences[i].append(experiment.getId())
        
    return adjustableParameters_info,experimentReferences,lowerBound,upperBound,initial_value

def get_fit_experiments(doc,task,working_dir,external_variables_info={}):
    """
    Return a dictionary containing fit experiment information.
    Assume the model is a CellML model.
    The variables in experiment conditions are set to be parameters.

    Parameters
    ----------
    doc: :obj:`SedDocument`
        An instance of SedDocument
    task: :obj:`SedParameterEstimationTask`
    working_dir: :obj:`str`
        working directory of the SED document (path relative to which models are located)
    dfDict: :obj:`dict` of :obj:`pandas.DataFrame`
    external_variables_info: dict, optional
        The external variables to be specified, in the format of {id:{'component': , 'name': }}

    Raises
    ------
    ValueError
    RuntimeError

    Returns
    -------
    dict
        A dictionary containing fit experiment information
        fitExperiments: dict,
        {experimentId: {'type':type,'algorithm':algorithm,'tspan':tspan,'parameters':parameters,'fitness_info':fitness_info,
        'model':model,'cellml_model':cellml_model,'analyser':analyser, 'module':module, 'mtype':mtype,
        'external_variables_info':external_variables_info,'param_indices':param_indices
        },
        'adjustableParameters_info':adjustableParameters_info,'experimentReferences':experimentReferences,	
        'lowerBound':lowerBound,'upperBound':upperBound,'initial_value':initial_value}
    
    Note
    ----
    If the experimentalCondition (fitMapping) is an array, then treat it as external variable (input)
    If the experimentalCondition (fitMapping) is a scalar, then treat it as parameter with initial value equal to the scalar
    """
    # get the variables recorded by the task
    task_vars = get_variables_for_task(doc, task)
    if len(task_vars) == 0:
        print('Task does not record any variables.')
        raise RuntimeError('Task does not record any variables.')  
    fitExperiments={}
    original_models = get_models_referenced_by_task(doc,task)
    model=original_models[0] # parameter estimation task should have only one model
    # get the optimization algorithm
    dict_algorithm_opt=get_dict_algorithm(task.getAlgorithm())
    method, opt_parameters=get_KISAO_parameters_opt(dict_algorithm_opt)
    # prepare data frames for data sources
    dfDict={}
    for dataDescription in doc.getListOfDataDescriptions() :
        dfDict.update({dataDescription.getId():get_df_from_dataDescription(dataDescription, working_dir)})
    # resolve the model and apply XML changes
    try:
        temp_model_source=resolve_model(model, doc, working_dir)
        if temp_model_source is None:
            temp_model_source =model.getSource()
        cellml_model,parse_issues=parse_model(temp_model_source, False)
        model_base_dir=os.path.dirname(temp_model_source)
        importer,issues_import=resolve_imports(cellml_model, model_base_dir,False)
        flatModel=importer.flattenModel(cellml_model)
        if not flatModel:
            raise RuntimeError('Model flattening failed!')
        else:
            full_path = os.path.join(working_dir, model.getId()+'_flat.cellml')
            writeCellML(flatModel, full_path)
            model.setSource(full_path)
        temp_model, temp_model_source, model_etree = resolve_model_and_apply_xml_changes(model, doc, working_dir) # must set save_to_file=True
        adjustableParameters_info,experimentReferences,lowerBound,upperBound,initial_value=get_adjustableParameters(model_etree,task)
        adjustables=(lowerBound,upperBound,initial_value)
    except ValueError as exception:
        print('Error in resolve_model_and_apply_xml_changes or parse_model:',exception)
        raise exception
    for fitExperiment in task.getListOfFitExperiments():
        external_variables_info_new=copy.deepcopy(external_variables_info)
        fitExperiments[fitExperiment.getId()]={}
        if fitExperiment.getTypeAsString ()=='steadyState':
            fitExperiments[fitExperiment.getId()]['type']='steadyState'
        elif fitExperiment.getTypeAsString ()=='timeCourse':
            fitExperiments[fitExperiment.getId()]['type']='timeCourse'
        else:
            raise ValueError('Experiment type {} is not supported!'.format(fitExperiment.getTypeAsString ()))
        sim_setting=SimSettings()
        sim_setting.number_of_steps=0
        sed_algorithm = fitExperiment.getAlgorithm()
        try:
            dict_algorithm=get_dict_algorithm(sed_algorithm)
            sim_setting.method, sim_setting.integrator_parameters=get_KISAO_parameters(dict_algorithm)
        except ValueError as exception:
            print('Error in get_dict_algorithm or get_KISAO_parameters:',exception)
            raise exception 
        sub_adjustableParameters_info={}
        adj_param_indices=[]
        for i in range(len(experimentReferences)):         
            if fitExperiment.getId() in experimentReferences[i]:
                sub_adjustableParameters_info.update({i:adjustableParameters_info[i]})
                adj_param_indices.append(i)
        
        external_variables_info_new.update(sub_adjustableParameters_info) 
        observables_exp={}
        observables_weight={}
        observables_info={}
        fitExperiments[fitExperiment.getId()]['parameters']={} 
        parameters_values=[]       
        for fitMapping in fitExperiment.getListOfFitMappings ():
            if fitMapping.getTypeAsString ()=='time':
                try:
                    tspan=get_value_of_dataSource(doc, fitMapping.getDataSource(),dfDict)
                except ValueError as exception:
                    print('Error in get_value_of_dataSource:',exception)
                    raise exception
                # should be 1D array
                if tspan.ndim>1:
                    raise ValueError('The time course {} is not 1D array!'.format(fitMapping.getDataSource()))
                else:
                    sim_setting.tspan=tspan

            elif fitMapping.getTypeAsString ()=='experimentalCondition':
                try:
                    initial_value_=get_value_of_dataSource(doc, fitMapping.getDataSource(),dfDict)
                except ValueError as exception:
                    print('Error in get_value_of_dataSource:',exception)
                    raise exception
                if initial_value_.ndim>1:
                    raise ValueError('The experimental condition {} is not 1D array!'.format(fitMapping.getDataSource()))
                elif len(initial_value_)==1:
                    initial_value=initial_value_[0]
                else:
                    #raise ValueError('The experimental condition {} is not a scalar!'.format(fitMapping.getDataSource()))
                    initial_value=initial_value_
                dataGenerator=doc.getDataGenerator(fitMapping.getTarget())
                sedVars=get_variables_for_data_generator(dataGenerator)
                if len(sedVars)>1:
                    raise ValueError('The data generator {} has more than one variable!'.format(fitMapping.getTarget()))
                else:
                    try:
                        parameter_info = get_variable_info_CellML(sedVars,model_etree)
                        if isinstance (initial_value, np.ndarray):
                            external_variables_info_new.update(parameter_info)
                            parameters_values.append(initial_value) 
                        else:
                            fitExperiments[fitExperiment.getId()]['parameters'].update(parameter_info)
                            fitExperiments[fitExperiment.getId()]['parameters'][sedVars[0].getId()]['value']=initial_value                       
                    except ValueError as exception:
                        print('Error in get_variable_info_CellML:',exception)
                        raise exception                          
            elif fitMapping.getTypeAsString ()=='observable':
                try:
                    observable_exp=get_value_of_dataSource(doc, fitMapping.getDataSource(),dfDict)
                except ValueError as exception:
                    print('Error in get_value_of_dataSource:',exception)
                    raise exception
                if observable_exp.ndim>1:
                    raise ValueError('The observable {} is not 1D array!'.format(fitMapping.getDataSource()))             
                dataGenerator=doc.getDataGenerator(fitMapping.getTarget())
                sedVars=get_variables_for_data_generator(dataGenerator)
                sedVarIds=[var.getId() for var in sedVars]
                mathString=dataGenerator.getMath()
                try:
                    observable_info = get_variable_info_CellML(sedVars,model_etree)
                except ValueError as exception:
                    print('Error in get_variable_info_CellML:',exception)
                    raise exception                                   
                observables_info.update(observable_info)
                key=dataGenerator.getId() 
                workspace={}
                for param in dataGenerator.getListOfParameters():
                    workspace[param.getId()] = param.getValue()                                               
                if fitMapping.isSetWeight():
                    weight=fitMapping.getWeight()
                    observables_weight.update({key:weight})
                elif fitMapping.isSetPointWeight ():
                    try:
                        pointWeight=get_value_of_dataSource(doc, fitMapping.getPointWeight(),dfDict)
                    except ValueError as exception:
                        print('Error in get_value_of_dataSource:',exception)
                        raise exception
                    if pointWeight.ndim>1:
                        raise ValueError('The point weight {} is not 1D array!'.format(fitMapping.getPointWeight()))
                    else:
                        # observable_exp and pointWeight should have the same length
                        if len(observable_exp)!=len(pointWeight):
                            raise ValueError('The observable {} and point weight {} do not have the same length!'.format(fitMapping.getDataSource(),fitMapping.getPointWeight()))
                        else:
                            observables_weight.update({key:pointWeight})
                else:
                    raise ValueError('Fit mapping {} does not have a weight!'.format(fitMapping.getId()))
                      
                observables_exp.update({key:(sedVarIds, mathString, workspace,observable_exp)})

            else:
                raise ValueError('Fit mapping type {} is not supported!'.format(fitMapping.getTypeAsString ()))
            
        fitExperiments[fitExperiment.getId()]['fitness_info']=(observables_info,observables_weight,observables_exp)
        fitExperiments[fitExperiment.getId()]['sim_setting']=sim_setting
        fitExperiments[fitExperiment.getId()].update({'temp_model_source':temp_model_source,'external_variables_info':external_variables_info_new,
                                                    'adj_param_indices':adj_param_indices,'parameters_values':parameters_values})      
    return fitExperiments,adjustables,adjustableParameters_info,method, opt_parameters

def get_task_info(doc,task,working_dir,external_variables_info={},external_variables_values=[]):
    """ Collect information for a SedTask.
    The model is assumed to be in CellML format.
    The simulation type supported are UniformTimeCourse, OneStep and SteadyState.#TODO: add support for OneStep and SteadyState
    The ode solver supported are listed in KISAO_ALGORITHMS 
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

    # get the model
    original_models = get_models_referenced_by_task(doc,task)
    if len(original_models) != 1:
        raise RuntimeError('Task must reference exactly one model.')
    
    # get the variables recorded by the task
    task_vars = get_variables_for_task(doc, task)
    if len(task_vars) == 0:
        print('Task does not record any variables.')
        raise RuntimeError('Task does not record any variables.')
    # apply changes to the model if any
    try:
        #need to flatten the model if it is not already flat
        sed_model=original_models[0]
        temp_model_source=resolve_model(original_models[0], doc, working_dir)

        if temp_model_source is None:
            temp_model_source =sed_model.getSource()
        cellml_model,parse_issues=parse_model(temp_model_source, False)
        model_base_dir=os.path.dirname(temp_model_source)
        importer,issues_import=resolve_imports(cellml_model, model_base_dir,False)
        flatModel=importer.flattenModel(cellml_model)
        if not flatModel:
            raise RuntimeError('Model flattening failed!')
        else:
            full_path = os.path.join(working_dir, sed_model.getId()+'_flat.cellml')
            writeCellML(flatModel, full_path)
            sed_model.setSource(full_path)

        temp_model, temp_model_source, model_etree = resolve_model_and_apply_xml_changes(sed_model, doc, working_dir) # must set save_to_file=True
        cellml_model,parse_issues=parse_model(temp_model_source, False)
        # cleanup modified model sources
        os.remove(full_path)
        os.remove(temp_model_source)
        if cellml_model:
            model_base_dir=os.path.dirname(temp_model.getSource())
            analyser, issues =analyse_model_full(cellml_model,model_base_dir,external_variables_info)
            if analyser:
                mtype=get_mtype(analyser)
                external_variable=get_externals_varies(analyser, cellml_model, external_variables_info, external_variables_values)
                # write Python code to a temporary file
                # make a directory in the model_base_dir for the temporary file if it does not exist
                temp_folder = model_base_dir+os.sep+ temp_model.getId()+'_temp'
                if not os.path.exists(temp_folder):
                    os.makedirs(temp_folder)
                tempfile_py, full_path = tempfile.mkstemp(suffix='.py', prefix=temp_model.getId()+"_", text=True,dir=temp_folder)
                writePythonCode(analyser, full_path)
                module=load_module(full_path)
                os.close(tempfile_py)
                # and delete temporary file
                # os.remove(full_path)

    except (ValueError,FileNotFoundError) as exception:
        print(exception)
        raise RuntimeError(exception)
    # remove the temporary folder
    shutil.rmtree(temp_folder)
    if not cellml_model:
        print('Model parsing failed!',parse_issues)
        raise RuntimeError('Model parsing failed!')
    if not analyser:
        print('Model analysis failed!',issues)
        raise RuntimeError('Model analysis failed!')           
    
    # get observables and simulation settings
    try:
        variables_info = get_variable_info_CellML(task_vars,model_etree)
        observables=get_observables(analyser,cellml_model,variables_info)
        sedSimulation=doc.getSimulation(task.getSimulationReference())
        sim_setting=getSimSettingFromSedSim(sedSimulation) 
    except ValueError as exception:
        print(exception)
        raise RuntimeError(exception)    

    return mtype, module, sim_setting, observables, external_variable,task_vars