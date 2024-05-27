"""
This file contains functions that are modified versions of functions from https://github.com/biosimulators/Biosimulators_utils.git.
The original code is licensed under the MIT license.

Original source: https://github.com/biosimulators/Biosimulators_utils/blob/dev/biosimulators_utils/report/io.py

The MIT License (MIT)

Copyright (c) 2020, Center for Reproducible Biomedical Modeling
"""

import os
import numpy
import pandas
from sedChanger import calc_data_generator_results

def writeReport(report, results, base_path, rel_path, format='csv'):
    """ Write the results of a report to a file

    Args:
        report (:obj:`SedReport`): report
        results (:obj:`dict`): results of the data sets, format is {data_set.id: numpy.ndarray}
        base_path (:obj:`str`): path to store the outputs

            * CSV: directory in which to save outputs to files
              ``{base_path}/{rel_path}/{report.getId()}.csv``

        rel_path (:obj:`str`, optional): path relative to :obj:`base_path` to store the outputs
        format (:obj:`ReportFormat`, optional): report format (e.g., csv)
    Raises:
        :obj:`NotImplementedError`: if the report format is not supported

    """
    rel_path = os.path.relpath(rel_path, '.')
    results_array = []
    data_set_ids = []
    data_set_labels = []
    data_set_names = []
    data_set_data_types = []
    data_set_shapes = []
    for data_set in report.getListOfDataSets ():
        if data_set.getId() in results:
            data_set_result = results[data_set.getId()]
            results_array.append(data_set_result)
            data_set_ids.append(data_set.getId())
            data_set_labels.append(data_set.getLabel () )
            data_set_names.append(data_set.getName() or '')
            if data_set_result is None:
                data_set_data_types.append('__None__')
                data_set_shapes.append('')
            else:
                data_set_dtype = data_set_result.dtype
                if data_set_dtype in [numpy.dtype('object'), numpy.dtype('void'), numpy.dtype('S'), numpy.dtype('a')]:
                    msg = 'NumPy dtype should be a specific type such as `float64` or `int64` not `{}`.'.format(data_set_dtype.name)
                    raise TypeError(msg)
                data_set_data_types.append(data_set_dtype.name)
                data_set_shapes.append(','.join(str(dim_len) for dim_len in data_set_result.shape))
    results_array = pad_arrays_to_consistent_shapes(results_array)
    results_array = numpy.array(results_array)
    if format in ['csv','tsv','xlsx']:
        if results_array.ndim > 2:
            msg = 'Report has {} dimensions. Multidimensional reports cannot be exported to {}.'.format(
                results_array.ndim, format.upper())
            print(msg)
            return
        if len(set(data_set.getLabel () for data_set in report.getListOfDataSets ())) < len(report.getListOfDataSets ()):
            print('To facilitate machine interpretation, data sets should have unique labels.')
        msg = 'Reports exported to {}.'.format(
            format.upper())
        print(msg)
        #results_df = pandas.DataFrame(results_array, index=data_set_labels)
        results_df = pandas.DataFrame(numpy.transpose(results_array), columns=data_set_labels)
        if format in ['csv','tsv']:
            filename = os.path.join(base_path, rel_path + '.' + format)
            out_dir = os.path.dirname(filename)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            results_df.to_csv(filename, header=True, sep=',' if format == 'csv' else '\t')
        else:
            filename = os.path.join(base_path, os.path.dirname(rel_path) + '.' + format)
            out_dir = os.path.dirname(filename)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            with pandas.ExcelWriter(filename, mode='a' if os.path.isfile(filename) else 'w', engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name=os.path.basename(rel_path), header=False)
    else:
        raise NotImplementedError('Report format {} is not supported'.format(format))

def readReport(report, base_path, rel_path, format='csv'):
    rel_path = os.path.relpath(rel_path, '.')

    if format in ['csv','tsv','xlsx']:
        print('Reports exported to {}.'.format(
            format.upper()))
        if format in ['csv','tsv']:
            filename = os.path.join(base_path, rel_path + '.' + format)
            df = pandas.read_csv(filename,
                                 index_col=0,
                                 header=None,
                                 sep=',' if format == 'csv' else '\t')
        else:
            filename = os.path.join(base_path, os.path.dirname(rel_path) + '.' + format)
            df = pandas.read_excel(filename,
                                   sheet_name=os.path.basename(rel_path),
                                   index_col=0,
                                   header=None,
                                   engine='openpyxl')
        df.columns = pandas.RangeIndex(start=0, stop=df.shape[1], step=1)
        results = {}
        data_set_labels = [data_set.getLabel () for data_set in report.getListOfDataSets ()]
        if df.index.tolist() == data_set_labels:
            data = df.to_numpy()
            for i_data_set, data_set in enumerate(report.getListOfDataSets ()):
                results[data_set.id] = data[i_data_set, :]
            extra_data_sets = set()
        else:
            data_set_label_to_index = {}
            for i_data_set, data_set_label in enumerate(df.index):
                if data_set_label not in data_set_label_to_index:
                    data_set_label_to_index[data_set_label] = i_data_set
                else:
                    data_set_label_to_index[data_set_label] = None
            unreadable_data_sets = []
            for data_set in report.getListOfDataSets ():
                i_data_set = data_set_label_to_index.get(data_set.getLabel (), None)
                if i_data_set is None:
                    # results[data_set.id] = None
                    unreadable_data_sets.append(data_set.getId())
                else:
                    results[data_set.id] = df.loc[data_set.getLabel (), :].to_numpy()
            if unreadable_data_sets:
                print('Some data sets could not be read because their labels are not unique:\n  - {}'.format(
                    '\n'.join('`' + id + '`' for id in sorted(unreadable_data_sets))))
            data_set_id_to_label = {data_set.id: data_set.getLabel () for data_set in report.getListOfDataSets ()}
            extra_data_sets = set(df.index) - set(data_set_id_to_label[id] for id in results.keys()) - set(unreadable_data_sets)
        file_data_set_ids = set(results.keys()) | extra_data_sets
    else:
        raise NotImplementedError('Report format {} is not supported'.format(format))
    report_data_set_ids = set(data_set.id for data_set in report.getListOfDataSets ())
    missing_data_set_ids = report_data_set_ids.difference(file_data_set_ids)
    extra_data_set_ids = file_data_set_ids.difference(report_data_set_ids)
    if missing_data_set_ids:
        print('File does not contain data for the following data sets of the report:\n  - {}'.format(
            '\n'.join('`' + id + '`' for id in sorted(missing_data_set_ids))))
    if extra_data_set_ids:
        print('File contains additional data that could not be mapped to data sets of the report:\n  - {}'.format(
            '\n'.join('`' + id + '`' for id in sorted(extra_data_set_ids))))
    return results

def pad_arrays_to_consistent_shapes(arrays):
    """ Pad a list of NumPy arrays to a consistent shape

    Args:
        arrays (:obj:`list` of :obj:`numpy.ndarray`): list of NumPy arrays

    Returns:
        :obj:`list` of :obj:`numpy.ndarray`: list of padded arrays
    """
    shapes = set()
    for array in arrays:
        if array is not None:
            shape = array.shape
            if not shape and array.size:
                shape = (1,)
            shapes.add(shape)

    if len(shapes) > 1:
        print('Arrays do not have consistent shapes', UserWarning)

    max_shape = []
    for shape in shapes:
        max_shape = max_shape + [1 if max_shape else 0] * (len(shape) - len(max_shape))
        shape = list(shape) + [1 if shape else 0] * (len(max_shape) - len(shape))
        max_shape = [max(x, y) for x, y in zip(max_shape, shape)]

    padded_arrays = []
    for array in arrays:
        if array is None:
            array = numpy.full(max_shape, numpy.nan)

        shape = tuple(list(array.shape)
                      + [1 if array.size else 0]
                      * (len(max_shape) - array.ndim))
        array = array.astype('float64').reshape(shape)

        pad_width = tuple((0, x - y) for x, y in zip(max_shape, shape))

        if pad_width:
            array = numpy.pad(array,
                              pad_width,
                              mode='constant',
                              constant_values=numpy.nan)

        padded_arrays.append(array)

    return padded_arrays

def exec_report(report, variable_results, base_out_path, rel_out_path, formats, task):
    """ Execute a report, generating the data sets which are available

    Args:
        report (:obj:`Report`): report
        variable_results (:obj:`VariableResults`): result of each data generator
        base_out_path (:obj:`str`): path to store the outputs

            * CSV: directory in which to save outputs to files
              ``{base_out_path}/{rel_out_path}/{report.getId()}.csv``

        rel_out_path (:obj:`str`, optional): path relative to :obj:`base_out_path` to store the outputs
        formats (:obj:`list` of :obj:`ReportFormat`, optional): report format (e.g., csv)
        task (:obj:`SedTask`): task

    Returns:
        :obj:`tuple`:

            * :obj:`dict`: results of the data sets, format is {data_set.id: numpy.ndarray}
            * :obj:`Status`: status
            * :obj:`Exception`: exception for failure
            * :obj:`bool`: whether :obj:`task` contribute a variable to the report
    Side effects:
        * writes the results of the report to a file
    """
    # calculate data generators
    data_generators_ids = set()
    doc=report.getSedDocument()
    for data_set in report.getListOfDataSets ():
        data_generators_ids.add(data_set.getDataReference ())
    
    data_generators = [doc.getDataGenerator(data_generator_id) for data_generator_id in data_generators_ids]

    data_gen_results, data_gen_statuses, data_gen_exceptions, task_contributes_to_report = calc_data_generators_results(
        data_generators, variable_results, task, make_shapes_consistent=True)

    # collect data sets
    data_set_results = {}

    running = False
    succeeded = True
    failed = False

    for data_set in report.getListOfDataSets ():
        data_gen_res = data_gen_results[data_set.getDataReference ()]
        data_set_results[data_set.getId()] = data_gen_res

        data_gen_status = data_gen_statuses[data_set.getDataReference ()]
        if data_gen_status == 'FAILED':
            failed = True
        if data_gen_status == 'SUCCEEDED':
            running = True
        else:
            succeeded = False

    for format in formats:
        try:
            writeReport(report, data_set_results, base_out_path, os.path.join(rel_out_path, report.getId()) if rel_out_path else report.getId(), format)
        except Exception as exception:
            failed = True
            failed_msg = 'Failed to write report to {} format: {}'.format(format, str(exception))
            if failed_msg not in str(exception):
                exception = ValueError(failed_msg + '\n' + str(exception))

    if failed:
        status = 'FAILED'

    elif running:
        if succeeded:
            status = 'SUCCEEDED'
        else:
            status = 'RUNNING'	

    else:
        status = 'QUEUED'

    return data_set_results, status, data_gen_exceptions, task_contributes_to_report

def calc_data_generators_results(data_generators, variable_results,task, make_shapes_consistent=True):
    """ Calculator the values of a list of data generators

    Args:
        data_generators (:obj:`list` of :obj:`SedDataGenerator`): SED task
        variable_results (:obj:`VariableResults`): results of the SED variables involved in the data generators
        task (:obj:`SedTask`): SED task
        make_shapes_consistent (:obj:`bool`, optional): where to make the shapes of the data generators consistent
            (e.g., for concatenation into a table for a report)

    Returns:
        :obj:`tuple`:

            * :obj:`dict`: values of the data generators, format is {data_generator.id: numpy.ndarray}
            * :obj:`dict` of :obj:`str` to :obj:`Status`: dictionary that maps the id of each data generator to its status
            * :obj:`Exception`: exception for failures
            * :obj:`bool`: where the task contributes to any of the data generators
    """
    task_contributes_to_data_generators = False
    statuses = {}
    exceptions = []
    results = {}

    for data_gen in data_generators:
        vars_available = True
        vars_failed = False
        for variable in data_gen.getListOfVariables():
            if variable.getTaskReference () == task.getId():
                task_contributes_to_data_generators = True
            if variable.getId() in variable_results:
                if variable_results.get(variable.getId(), None) is None:
                    vars_available = False
                    vars_failed = True
            else:
                vars_available = False

        if vars_failed:
            status = 'FAILED'
            msg = 'Data generator {} cannot be calculated because its variables were not successfully produced.'.format(data_gen.getId())
            exceptions.append(ValueError(msg))
            result = None

        elif vars_available:
            try:
                result = calc_data_generator_results(data_gen, variable_results)
                status = 'SUCCEEDED'
            except Exception as exception:
                result = None
                exceptions.append(exception)
                status = 'FAILED'

        else:
            status = 'QUEUED'	
            result = None

        statuses[data_gen.getId()] = status
        results[data_gen.getId()] = result

    if make_shapes_consistent:
        arrays = results.values()
        consistent_arrays = pad_arrays_to_consistent_shapes(arrays)
        for data_gen_id, result in zip(results.keys(), consistent_arrays):
            results[data_gen_id] = result

    if exceptions:
        exception = ValueError('Some generators could not be produced:\n  - {}'.format(
            '\n  '.join(str(exception) for exception in exceptions)))
    else:
        exception = None

    return results, statuses, exception, task_contributes_to_data_generators