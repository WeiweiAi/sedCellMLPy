from sedTasker import exec_task, report_task, exec_parameterEstimationTask

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
            try:
                current_state, variable_results_i= exec_task(doc,task,working_dir,external_variables_info,external_variables_values,current_state=None)
                print('Task {} executed successfully'.format(task.getId()))
                variable_results.update(variable_results_i)
            except Exception as exception:
                print(exception)
                return

        elif task.isSedRepeatedTask ():
            raise RuntimeError('RepeatedTask not supported yet')
        elif task.isSedParameterEstimationTask ():
            try:
                res=exec_parameterEstimationTask(doc,task, working_dir,external_variables_info,external_variables_values,ss_time,cost_type)

            except Exception as exception:
                print(exception)
                return
    report_result = report_task(doc,task, variable_results, base_out_path, rel_out_path, report_formats =['csv'])


