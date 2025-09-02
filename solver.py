from scipy.integrate import ode
import numpy as np
import functools
from sksundae.cvode import CVODE

"""
======
Solver
======
The solver module provides the functionality to solve a system.
The system is given as a python module generated from a CellML model.
The python module is assumed to be generated using libCellML (version 0.5.0).
The system can be either ODE or algebraic equations.
The system can be solved using Euler method or scipy supported solvers.
The code is modified from
https://github.com/hsorby/cellsolver.git, which is licensed under Apache License 2.0.

The solver module provides the following functions:
    * create_sed_results - create a dictionary to hold the simulation results for each observable.
    * initialize_module - initialize a module based on the given model type and parameters.
    * solve_euler - Euler method solver.
    * solve_scipy - scipy supported solvers.
    * algebra_evaluation - algebraic evaluation.
"""

def create_sed_results(observables, N):
    """
    Create a dictionary to hold the simulation results for each observable.

    Parameters
    ----------
    observables : dict
        A dictionary containing the observables to be recorded.
    N : int
        The number of time points to simulate.

    Returns
    -------
    dict
        A dictionary to hold the simulation results for each observable.
        The dictionary is of the form {id: numpy.ndarray} where id is the
        identifier of the observable and the numpy.ndarray is of size N+1.
    """

    sed_results = {}
    for id in observables.keys():
        sed_results.update({id: np.zeros(N+1)})
    return sed_results

def _initialize_module_ode(module, voi, external_variable=None, parameters={}):
    """
    Initialize the module with ODEs.

    Parameters
    ----------
    module : object
        The module to be initialized.
    external_variable : object, optional
        The function to specify external variable.
    parameters : dict, optional
        The information to modify the parameters.
        The format is {id:{'name':'variable name','component':'component name',
        'type':'state','value':value,'index':index}}

    Raises
    ------
    ValueError
        If the initial value of voi or external variable is specified in the parameters,
        a ValueError will be raised.

    Returns
    -------
    tuple
        A tuple containing the initialized states, rates, and variables.

    Notes
    -----
    If the initial value of a state variable is specified in the parameters,
    the initial value may not be modified accordingly.
    """
    rates = module.create_states_array()
    states = module.create_states_array()
    variables = module.create_variables_array()

    if external_variable:
        module.initialise_variables(voi,states, rates, variables,external_variable)
    else:
       module.initialise_variables(states, rates, variables)

    for id, v in parameters.items():
        if v['type'] == 'state':
            states[v['index']]=v['value']
        elif v['type'] == 'constant' or v['type'] == 'computed_constant' and v['type'] == 'algebraic':
           variables[v['index']]=v['value']
        elif v['type'] == 'variable_of_integration' or v['type'] == 'external':
            raise ValueError('The initial value of voi or external variable cannot be modified!')
        else:
            raise ValueError('The parameter type {} of is not supported!'.format(v['type']))

    module.compute_computed_constants(variables)
    module.compute_computed_constants(variables) # Need to call it twice to update the computed constants;TODO: need to discuss with the libCellML team
    if external_variable:
        module.compute_rates(voi, states, rates, variables,external_variable)
        module.compute_variables(voi, states, rates, variables,external_variable)
    else:
        module.compute_rates(voi, states, rates, variables)
        module.compute_variables(voi, states, rates, variables)

    return states, rates, variables

def _initialize_module_algebraic(module,external_variable=None,parameters={}):
    """
    Initialize the module with only algebraic equations.

    Parameters
    ----------
    module : object
        The module to be initialized.
    external_variable : object, optional
        The function to specify external variable.
    parameters : dict, optional
        The information to modify the parameters.
        The format is {id:{'name':'variable name','component':'component name',
        'type':'state','value':value,'index':index}}

    Raises
    ------
    ValueError
        If other types of variables are specified in the parameters,
        a ValueError will be raised.

    Returns
    -------
    list
        A list containing the initialized variables.
    """
    variables = module.create_variables_array()

    if external_variable:
        module.initialise_variables(variables,external_variable)
    else:
        module.initialise_variables(variables)

    for id, v in parameters.items():
        if v['type'] == 'constant' or v['type'] == 'computed_constant' and v['type'] == 'algebraic':
           variables[v['index']]=v['value']
        else:
            raise ValueError('The parameter type {} is not supported!'.format(v['type']))

    module.compute_computed_constants(variables)
    if external_variable:
        module.compute_variables(variables,external_variable)
    else:
        module.compute_variables(variables)

    return variables

def initialize_module(mtype, observables, N, module, voi=0, external_module=None, parameters={}):
    """
    Initializes a module based on the given model type and parameters.

    Parameters
    ----------
    mtype : str
        The type of model to initialize ('ode' or 'algebraic').
    observables : dict
        A dictionary containing the observables to be recorded.
    N : int
        The number of simulation steps.
    module : object
        The module to initialize.
    voi : float, optional
        The initial value of the variable of integration. Defaults to 0.
    external_module : object, optional
    parameters : dict, optional
        The information to modify the parameters.
        The format is {id:{'name':'variable name','component':'component name',
        'type':'state','value':value,'index':index}}

    Raises
    ------
    ValueError
        If the initial value of voi or external variable is specified in the parameters,
        a ValueError will be raised.

    Returns
    -------
    tuple
        A tuple containing the current state of the module.
        The format is (voi, states, rates, variables, current_index, sed_results).
    """
    sed_results=create_sed_results(observables, N)
    external_variable=None

    if mtype=='ode' or mtype=='dae':
        try:
            if external_module:
                external_variable=functools.partial(external_module.external_variable_ode,result_index=0)
            states, rates, variables=_initialize_module_ode(module,voi, external_variable,parameters)
        except ValueError as e:
            raise ValueError(e)
        current_state = (voi, states, rates, variables, 0,sed_results)

    elif mtype=='algebraic':
        try:
            if external_module:
                external_variable=external_module.external_variable_algebraic
            variables=_initialize_module_algebraic(module,external_variable,parameters)
        except ValueError as e:
            raise ValueError(e)
        current_state = (voi, None, None, variables, 0, sed_results)
    else:
        raise ValueError('The model type {} is not supported!'.format(mtype))

    return current_state

def _update_rates(voi, states, rates, variables, module, external_variable=None):
    """ Update the rates of the module.
    Parameters
    ----------
    voi : float
        The current value of the independent variable.
    states : list
        The current state of the system.
    rates : list
        The current rates of change of the system.
    variables : list
        The current variables of the system.
    module : object
        The module to update.
    external_variable : object, optional
        The function to specify external variable.

    Returns
    -------
    list
        The updated rates of change of the system.
    """
    if external_variable:
     #  _update_variables(voi, states, rates, variables, module, external_variable)
       module.compute_rates(voi, states, rates, variables,external_variable)
       _update_variables(voi, states, rates, variables, module, external_variable)
    else:
    #    _update_variables(voi, states, rates, variables, module)
        module.compute_rates(voi, states, rates, variables)
        _update_variables(voi, states, rates, variables, module)
    return rates

def _update_rates_cvode(voi, states, rates, userdata=()):
    """ Update the rates of the module.
    Parameters
    ----------
    voi : float
        The current value of the independent variable.
    states : list
        The current state of the system.
    rates : list
        The current rates of change of the system.
    variables : list
        The current variables of the system.
    module : object
        The module to update.
    external_variable : object, optional
        The function to specify external variable.

    Returns
    -------
    list
        The updated rates of change of the system.
    """
    variables, module, external_variable = userdata
    if external_variable:
     #  _update_variables(voi, states, rates, variables, module, external_variable)
       module.compute_rates(voi, states, rates, variables,external_variable)
       _update_variables(voi, states, rates, variables, module, external_variable)
    else:
    #    _update_variables(voi, states, rates, variables, module)
        module.compute_rates(voi, states, rates, variables)
        _update_variables(voi, states, rates, variables, module)
    userdata=(variables, module, external_variable)
    return rates

def _update_variables_cvode(voi, states, rates, userdata=()):
    """ Update the variables of the module.

    Parameters
    ----------
    voi : float
        The current value of the independent variable.
    states : list
        The current state of the system.
    rates : list
        The current rates of change of the system.
    variables : list
        The current variables of the system.
    module : object
        The module to update.
    external_variable : object, optional
        The function to specify external variable.

    Side effects
    ------------
    The variables of the module are updated.

    """
    variables, module, external_variable = userdata
    if external_variable:
       if states is None and rates is None: # algebraic
           module.compute_variables(variables,external_variable)
       else:
           module.compute_variables(voi, states, rates, variables,external_variable)
    else:
        if states is None and rates is None:
            module.compute_variables(variables)
        else:
            module.compute_variables(voi, states, rates, variables)
    userdata=(variables, module, external_variable)
def _update_variables(voi, states, rates, variables, module, external_variable=None):
    """ Update the variables of the module.

    Parameters
    ----------
    voi : float
        The current value of the independent variable.
    states : list
        The current state of the system.
    rates : list
        The current rates of change of the system.
    variables : list
        The current variables of the system.
    module : object
        The module to update.
    external_variable : object, optional
        The function to specify external variable.

    Side effects
    ------------
    The variables of the module are updated.

    """
    if external_variable:
       if states is None and rates is None: # algebraic
           module.compute_variables(variables,external_variable)
       else:
           module.compute_variables(voi, states, rates, variables,external_variable)
    else:
        if states is None and rates is None:
            module.compute_variables(variables)
        else:
            module.compute_variables(voi, states, rates, variables)

def _append_current_results(sed_results, index, observables, voi, states, variables):
    """ Append the current results to the results.

    Parameters
    ----------
    sed_results : dict
        The dictionary of the simulation results.
    index : int
        The current index of the results.
    observables : dict
        The dictionary of the observables.
        {id:{'name':'variable name','component':'component name',
        'type':'state','index':index}}
    voi : float
        The current value of the independent variable.
    states : list
        The current state of the system.
    variables : list
        The current variables of the system.

    Side effects
    ------------
    The current results are appended to the results.
    """

    for id, v in observables.items():
        if v['type'] == 'variable_of_integration':
            sed_results[id][index] = voi
        elif v['type'] == 'state':
            sed_results[id][index] = states[v['index']]
        else:
            sed_results[id][index] = variables[v['index']]

def solve_euler(module, current_state, observables, output_start_time, output_end_time,
                number_of_steps, step_size=None, external_module=None):
    """ Euler method solver.

    Parameters
    ----------
    module : object
        The module to solve.
    current_state : tuple
        The current state of the module.
        The format is (voi, states, rates, variables, current_index, sed_results).
    observables : dict
        The dictionary of the observables.
    output_start_time : float
        The start time of the output.
    output_end_time : float
        The end time of the output.
    number_of_steps : int
        The number of steps.
    step_size : float, optional
        The step size. Default is None.
        When step_size is None, the step size is calculated based on the
        output_start_time, output_end_time, and number_of_steps.
    external_variable : object, optional
        The function to specify external variable. Default is None.

    Raises
    ------
    ValueError
        If output_start_time, output_end_time, or number_of_steps is not valid.


    Returns
    -------
    tuple
        The current state of the module.
        The format is (voi, states, rates, variables, current_index, sed_results).
    """

    voi, states, rates, variables, current_index, sed_results = current_state
    external_variable=None
    if external_module:
            external_variable=functools.partial(external_module.external_variable_ode,result_index=current_index)

    if output_start_time > output_end_time or number_of_steps < 0:
        raise ValueError('output_start_time must be less than output_end_time and number_of_steps must be greater than 0.')
    elif output_start_time == output_end_time and number_of_steps>0:
        raise ValueError('when output_start_time = output_end_time, number_of_steps must be 0.')
    elif output_start_time < output_end_time and number_of_steps==0:
        raise ValueError('when output_start_time < output_end_time, number_of_steps must be greater than 0.')
    elif output_start_time == output_end_time and number_of_steps==0:
        if voi > output_start_time:
            raise ValueError('The current value of the independent variable is greater than output_start_time.')
        elif voi == output_start_time:
            return current_state
        else: # voi < output_start_time
            output_step_size = output_start_time-voi
            if step_size is None:
                step_size = output_step_size
            else:
                if step_size > output_step_size:
                    step_size = output_step_size # modify step_size to output_step_size update the states at least once
            # integrate to the output start point
            n=abs((output_start_time-voi)/step_size)
            if n < 1:
                rates=_update_rates(voi, states,  rates, variables, module, external_variable)
                voi = output_start_time
            for i in range(int(n)):
                rates=_update_rates(voi, states,  rates, variables, module, external_variable)
                delta = list(map(lambda var: var * step_size, rates))
                states = [sum(x) for x in zip(states, delta)]
                voi += step_size
            _update_variables( voi, states, rates, variables, module, external_variable)
            # save observables
            _append_current_results(sed_results, current_index, observables, voi, states, variables)
            current_state = (voi, states, rates, variables, current_index, sed_results)
    else: # number_of_steps > 0 and output_start_time < output_end_time
        if voi > output_start_time:
            raise ValueError('The current value of the independent variable is greater than output_start_time.')
        output_step_size = (output_end_time - output_start_time) / number_of_steps
        if step_size is None:
            step_size = output_step_size
        else:
            if step_size > output_step_size:
                step_size = output_step_size # modify step_size to output_step_size update the states at least once
        # integrate to the output start point
        n=abs((output_start_time-voi)/step_size)
        if n < 1:
            rates=_update_rates(voi, states,  rates, variables, module, external_variable)
            voi = output_start_time
        for i in range(int(n)):
            rates=_update_rates(voi, states,  rates, variables, module, external_variable)
            delta = list(map(lambda var: var * step_size, rates))
            states = [sum(x) for x in zip(states, delta)]
            voi += step_size
        _update_variables( voi, states, rates, variables, module, external_variable)
        # save observables
        _append_current_results(sed_results, current_index, observables, voi, states, variables)

        # integrate to the output end point
        n = output_step_size/step_size
        for i in range(int(number_of_steps)):
            current_index = current_index+1
            if external_module:
                external_variable=functools.partial(external_module.external_variable_ode,result_index=current_index)
            for j in range(int(n)):
                rates=_update_rates( voi, states,  rates, variables, module, external_variable)
                delta = list(map(lambda var: var * step_size, rates))
                states = [sum(x) for x in zip(states, delta)]
                voi += step_size

            _update_variables(voi, states, rates, variables, module, external_variable)
            # save observables
            _append_current_results(sed_results, current_index, observables, voi, states, variables)

        current_state = (voi, states, rates, variables, current_index, sed_results)

    return current_state

def solve_scipy(module, current_state, observables, output_start_time, output_end_time,
                number_of_steps, method, integrator_parameters, external_module=None):
    """ Scipy supported solvers.

    Parameters
    ----------
    module : object
        The module to solve.
    current_state : tuple
        The current state of the module.
        The format is (voi, states, rates, variables, current_index, sed_results).
    observables : dict
        The dictionary of the observables.
    output_start_time : float
        The start time of the output.
    output_end_time : float
        The end time of the output.
    number_of_steps : int
        The number of steps.
    method : str
        The name of the integrator.
    integrator_parameters : dict
        The parameters of the integrator.
    external_variable : object, optional
        The function to specify external variable. Default is None.

    Raises
    ------
    RuntimeError
        If the scipy.integrate.ode failed, a RuntimeError will be raised.
    ValueError
        If output_start_time, output_end_time, or number_of_steps is not valid.

    Returns
    -------
    tuple
        The current state of the module.
        The format is (voi, states, rates, variables, current_index, sed_results).
    """
    voi, states, rates, variables, current_index, sed_results = current_state
    external_variable=None
    if external_module:
        external_variable=functools.partial(external_module.external_variable_ode,result_index=current_index)
    # Set the initial conditions and parameters
    solver = ode(_update_rates)
    solver.set_initial_value(states, voi)
    solver.set_f_params(rates, variables, module, external_variable)
    solver.set_integrator(method, **integrator_parameters)

    if output_start_time > output_end_time or number_of_steps < 0:
        raise ValueError('output_start_time must be less than output_end_time and number_of_steps must be greater than 0.')
    elif output_start_time == output_end_time and number_of_steps>0:
        raise ValueError('when output_start_time = output_end_time, number_of_steps must be 0.')
    elif output_start_time < output_end_time and number_of_steps==0:
        raise ValueError('when output_start_time < output_end_time, number_of_steps must be greater than 0.')
    elif output_start_time == output_end_time and number_of_steps==0:
        if voi > output_start_time:
            raise ValueError('The current value of the independent variable is greater than output_start_time.')
        elif voi == output_start_time:
            return current_state
        else: # voi < output_start_time
            output_step_size = output_start_time-voi
            # integrate to the output start point
            n = abs((output_start_time - voi) / output_step_size)
            for i in range(int(n)):
                solver.integrate(solver.t + output_step_size)
                if not solver.successful():
                    raise RuntimeError('scipy.integrate.ode failed.')
            _update_variables(solver.t, solver.y, rates, variables, module, external_variable)
            # save observables
            _append_current_results(sed_results, current_index, observables, solver.t, solver.y, variables)
            current_state = (solver.t, solver.y, rates, variables, current_index, sed_results)
    else: # number_of_steps > 0 and output_start_time < output_end_time
        if voi > output_start_time:
            raise ValueError('The current value of the independent variable is greater than output_start_time.')
        # integrate to the output start point
        if voi < output_start_time:
            solver.integrate(solver.t + (output_start_time - voi))
            if not solver.successful():
                raise RuntimeError('scipy.integrate.ode failed.')
        _update_variables(solver.t, solver.y, rates, variables, module, external_variable)
        # save observables
        _append_current_results(sed_results, current_index, observables, solver.t, solver.y, variables)
        # integrate to the output end point
        output_step_size = (output_end_time - output_start_time) / number_of_steps
        for i in range(number_of_steps):
            current_index = current_index+1
            if external_module:
                external_variable=functools.partial(external_module.external_variable_ode,result_index=current_index)
            solver.integrate(solver.t + output_step_size)
            if not solver.successful():
                raise RuntimeError('scipy.integrate.ode failed.')
            _update_variables(solver.t, solver.y, rates, variables, module, external_variable)
            # save observables
            _append_current_results(sed_results, current_index, observables, solver.t, solver.y, variables)
        current_state = (solver.t, solver.y, rates, variables, current_index, sed_results)
    return current_state


def external_variable(voi, states, rates, variables, index):
    external_values = [1]*100+[2]*101

    for i in range(0, len(external_values)-1):
        voi0 = 0.1*i
        voi1 = 0.1*(i+1)

        if (voi >= voi0) and (voi < voi1):
            value0 = external_values[i]

            return value0 + (voi-voi0)/(voi1-voi0)*(external_values[i+1]-value0)

    return external_values[-1]


def solve_cvode(module, current_state, observables, output_start_time, output_end_time,
                number_of_steps, method, integrator_parameters, external_module=None):
    """ Use the CVODE integrator to solve the system from sksundae.

    Parameters
    ----------
    module : object
        The module to solve.
    current_state : tuple
        The current state of the module.
        The format is (voi, states, rates, variables, current_index, sed_results).
    observables : dict
        The dictionary of the observables.
    output_start_time : float
        The start time of the output.
    output_end_time : float
        The end time of the output.
    number_of_steps : int
        The number of steps.
    method : str
        The name of the integrator.
    integrator_parameters : dict
        The parameters of the integrator.
    external_variable : object, optional
        The function to specify external variable. Default is None.

    Raises
    ------
    RuntimeError
        If the scipy.integrate.ode failed, a RuntimeError will be raised.
    ValueError
        If output_start_time, output_end_time, or number_of_steps is not valid.

    Returns
    -------
    tuple
        The current state of the module.
        The format is (voi, states, rates, variables, current_index, sed_results).
    """
    voi, states, rates, variables, current_index, sed_results = current_state
    # external_variable=None
    # if external_module:
    #     external_variable=functools.partial(external_module.external_variable_ode,result_index=current_index)

    # Set the initial conditions and parameters
    userdata=(variables, module, external_variable)
    solver = CVODE(_update_rates_cvode, userdata=userdata, **integrator_parameters)
    result =solver.init_step(voi,states)
    #solver.set_f_params(rates, variables, module, external_variable)
    #solver.set_integrator(method, **integrator_parameters)

    if output_start_time > output_end_time or number_of_steps < 0:
        raise ValueError('output_start_time must be less than output_end_time and number_of_steps must be greater than 0.')
    elif output_start_time == output_end_time and number_of_steps>0:
        raise ValueError('when output_start_time = output_end_time, number_of_steps must be 0.')
    elif output_start_time < output_end_time and number_of_steps==0:
        raise ValueError('when output_start_time < output_end_time, number_of_steps must be greater than 0.')
    elif output_start_time == output_end_time and number_of_steps==0:
        if voi > output_start_time:
            raise ValueError('The current value of the independent variable is greater than output_start_time.')
        elif voi == output_start_time:
            return current_state
        else: # voi < output_start_time
            output_step_size = output_start_time-voi
            # integrate to the output start point
            n = abs((output_start_time - voi) / output_step_size)
            for i in range(int(n)):
                userdata=(variables, module, external_variable)
                result = solver.step(result.t + output_step_size,tstop=None)
                if not result.success:
                    raise RuntimeError(result.message)
            userdata=(variables, module, external_variable)
            _update_variables_cvode(result.t, result.y, rates,userdata)
            # save observables
            _append_current_results(sed_results, current_index, observables, result.t, result.y, variables)
            current_state = (result.t, result.y, rates, variables, current_index, sed_results)
    else: # number_of_steps > 0 and output_start_time < output_end_time
        if voi > output_start_time:
            raise ValueError('The current value of the independent variable is greater than output_start_time.')
        # integrate to the output start point
        if voi < output_start_time:
            result = solver.step(result.t + (output_start_time - voi),tstop=None)
            if not result.success:
                    raise RuntimeError(result.message)
        userdata=(variables, module, external_variable)
        _update_variables_cvode(result.t, result.y, rates,userdata)
        # save observables
        _append_current_results(sed_results, current_index, observables, result.t, result.y, variables)
        # integrate to the output end point
        output_step_size = (output_end_time - output_start_time) / number_of_steps
        for i in range(number_of_steps):
            current_index = current_index+1
            # if external_module:
            #     external_variable=functools.partial(external_module.external_variable_ode,result_index=current_index)
            userdata=(variables, module, external_variable)
            result = solver.step(result.t + output_step_size)
            if not result.success:
                    raise RuntimeError(result.message)
            userdata=(variables, module, external_variable)
            _update_variables_cvode(result.t, result.y, rates,userdata)
            # save observables
            _append_current_results(sed_results, current_index, observables, result.t, result.y, variables)
        current_state = (result.t, result.y, rates, variables, current_index, sed_results)
    return current_state

def algebra_evaluation(module, current_state, observables, number_of_steps, external_module=None):
    """ Algebraic evaluation.

    Parameters
    ----------
    module : object
        The module to solve.
    current_state : tuple
        The current state of the module.
        The format is (voi, states, rates, variables, current_index, sed_results).
    observables : dict
        The dictionary of the observables.
    number_of_steps : int
        The number of steps.
    external_variable : object, optional
        The function to specify external variable. Default is None.

    Returns
    -------
    tuple
        The current state of the module.
        The format is (voi, states, rates, variables, current_index, sed_results).
    """

    voi, states, rates, variables, current_index, sed_results = current_state
    external_variable=None
    if external_module:
        external_variable=functools.partial(external_module.external_variable_algebraic,result_index=current_index)
    _update_variables(0, None, None, variables, module, external_variable)
    _append_current_results(sed_results, current_index, observables, 0, None, variables)

    for i in range(number_of_steps):
        current_index = current_index + 1
        if external_module:
            external_variable=functools.partial(external_module.external_variable_algebraic,result_index=current_index)
        _update_variables(current_index, None, None, variables, module, external_variable)
        _append_current_results(sed_results, current_index, observables, 0, None, variables)

    current_state = (voi, states, rates, variables, current_index, sed_results)

    return current_state
