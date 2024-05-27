"""
This file contains functions that are modified versions of functions from https://github.com/biosimulators/Biosimulators_utils.git.
The original code is licensed under the MIT license.

Original source: https://github.com/biosimulators/Biosimulators_utils/blob/dev/biosimulators_utils/sedml/utils.py

The MIT License (MIT)

Copyright (c) 2020, Center for Reproducible Biomedical Modeling
"""
import requests
import tempfile
import os
import re
from lxml import etree
import enum
from math4sedml import compile_math, eval_math, AGGREGATE_MATH_FUNCTIONS
import libsedml
import numpy


CELLML2NAMESPACE ={"cellml":"http://www.cellml.org/cellml/2.0#"}
BIOMODELS_DOWNLOAD_ENDPOINT = 'https://www.ebi.ac.uk/biomodels/model/download/{}?filename={}_url.xml'


class ModelLanguagePattern(str, enum.Enum):
    """ Model language """
    CellML = r'^urn:sedml:language:cellml(\.\d+_\d+)?$'
    SBML = r'^urn:sedml:language:sbml(\.level\-\d+\.version\-\d+)?$'

def is_model_language_encoded_in_xml(language):
    """ Determine if the model language is encoded in XML. Note: only CellML and SBML are supported.

    Args:
        language (:obj:`str`): language

    Returns:
        :obj:`bool`: :obj:`True`, if the model language is encoded in XML
    """
    return (
        language and (
            re.match(ModelLanguagePattern.CellML, language)
            or re.match(ModelLanguagePattern.SBML, language)
        )
    )

def get_namespaces_with_prefixes(namespaces):
    """ Get a dictionary of namespaces less namespaces that have no prefix

    Args:
        namespaces (:obj:`dict`): dictionary that maps prefixes of namespaces their URIs

    Returns:
        :obj:`dict`: dictionary that maps prefixes of namespaces their URIs
    """
    if None in namespaces:
        namespaces = dict(namespaces)
        namespaces.pop(None)
    return namespaces

def eval_xpath(element, xpath, namespaces):
    """ Get the object(s) at an XPath

    Args:
        element (:obj:`etree._ElementTree`): element tree
        xpath (:obj:`str`): XPath
        namespaces (:obj:`dict`): dictionary that maps the prefixes of namespaces to their URIs

    Returns:
        :obj:`list` of :obj:`etree._ElementTree`: object(s) at the XPath
    """
    try:
        return element.xpath(xpath, namespaces=get_namespaces_with_prefixes(namespaces))
    except etree.XPathEvalError as exception:
        if namespaces:
            msg = 'XPath `{}` is invalid with these namespaces:\n  {}\n\n  {}'.format(
                xpath,
                '\n  '.join(' - {}: {}'.format(prefix, uri) for prefix, uri in namespaces.items()),
                exception.args[0].replace('\n', '\n  ')),
        else:
            msg = 'XPath `{}` is invalid without namespaces:\n\n  {}'.format(
                xpath, exception.args[0].replace('\n', '\n  ')),

        exception.args = (msg)
        raise

def get_value_of_variable_model_xml_targets(variable, model_etrees):

    """ Get the value of a variable of a model

    Args:
        variable (:obj:`Variable`): variable
        model_etrees (:obj:`dict` of :obj:`str` to :obj:`etree._Element`): dictionary that maps the
            ids of models to paths to files which contain their XML definitions
    Raise:
        NotImplementedError, ValueError

    Returns:
        :obj:`float`: value
    Note: only support explicit model elements and concept references
    """
    if variable.unsetTarget (): # 
        raise NotImplementedError('Compute model change variable `{}` must have a target'.format(variable.getId()))

    if variable.getTarget ().startswith('#'):
        raise NotImplementedError('Variable references to data descriptions are not supported.')

    obj_xpath, sep, attr = variable.getTarget ().rpartition('/@')
    if sep != '/@':
        raise NotImplementedError('the value of target ' + variable.getTarget() +
                                  ' cannot be obtained by examining the XML, as the target is not an attribute of a model element')

    et = model_etrees[variable.getModelReference()]
    obj = eval_xpath(et, obj_xpath, CELLML2NAMESPACE)
    if len(obj) != 1:
        raise ValueError('xpath {} must match a single object in model {}'.format(obj_xpath, variable.getModelReference()))

    ns, _, attr = attr.rpartition(':')
    if ns:
        attr = '{{{}}}{}'.format(CELLML2NAMESPACE[ns], attr)

    value = obj[0].get(attr)
    if value is None:
        raise ValueError('Target `{}` is not defined in model `{}`.'.format(variable.getTarget (), variable.getModelReference()))
    try:
        value = float(value)
    except ValueError:
        raise ValueError('Target `{}` in model `{}` must be a float.'.format(variable.getTarget (), variable.getModelReference()))

    return value

def resolve_model(model, sed_doc, working_dir):
    """ Resolve the source of a model

    Side effects:
        * Modifies the source of the model, 
          if the model needed to be resolved from a remote source

    Args:
        model (:obj:`Model`): model whose ``source`` is one of the following

            * A path to a file
            * A URL
            * A MIRIAM URN for an entry in the BioModelsl database (e.g., ``urn:miriam:biomodels.db:BIOMD0000000012``)
            * A reference to another model, using the ``id`` of the other model (e.g., ``#other-model-id``).
              In this case, the model also inherits changes from the parent model.

        sed_doc (:obj:`SedDocument`): parent SED document; used to resolve sources defined by reference to other models
        working_dir (:obj:`str`): working directory of the SED document (path relative to which models are located)
    
    Raise:
        ValueError, NotImplementedError, FileNotFoundError


    Returns:
        :obj:`str` or None: temporary path to the source of the modified model, if the model needed to be resolved from
            a remote source.
            None, if the model did not need to be resolved from a remote source.

    """
    source = model.getSource () 

    if source.lower().startswith('urn:'):
        if source.lower().startswith('urn:miriam:biomodels.db:'):
            biomodels_id = source.lower().replace('urn:miriam:biomodels.db:', '')
            url = BIOMODELS_DOWNLOAD_ENDPOINT.format(biomodels_id, biomodels_id)
            response = requests.get(url)
            try:
                response.raise_for_status()
            except Exception:
                raise ValueError('Model `{}` could not be downloaded from BioModels.'.format(biomodels_id))

            temp_file, temp_model_source = tempfile.mkstemp(dir=os.path.dirname(working_dir+'/temp/'))
            model.setSource(temp_model_source)
            os.close(temp_file)
            with open(temp_model_source, 'wb') as file:
                file.write(response.content)
        else:
            raise NotImplementedError('URN model source `{}` could be resolved.'.format(source))

        return model.getSource ()

    elif re.match(r'^http(s)?://', source, re.IGNORECASE):
        response = requests.get(source)
        try:
            response.raise_for_status()
        except Exception:
            raise ValueError('Model could not be downloaded from `{}`.'.format(source))

        temp_file, temp_model_source = tempfile.mkstemp(dir=os.path.dirname(working_dir+'/temp/'))
        model.setSource(temp_model_source)
        os.close(temp_file)
        with open(temp_model_source, 'wb') as file:
            file.write(response.content)

        return model.getSource ()

    elif source.startswith('#'):
        other_model_id = source[1:]
        models=[sed_doc.getModel(i) for i in range(sed_doc.getNumModels())]
        other_model = next((m for m in models if m.getId() == other_model_id), None)
        if other_model is None:
            raise ValueError('Relative model source `{}` does not exist.'.format(source))
        
        model.setSource(other_model.getSource())
        for change in other_model.getListOfChanges ():
            model.addChange(change)

        return resolve_model(model, sed_doc, working_dir)

    else:
        if os.path.isabs(source):
            model.setSource(source)
        else:
            model.setSource(os.path.join(working_dir, source)) 

        if not os.path.isfile(model.getSource()):
            raise FileNotFoundError('Model source file `{}` does not exist.'.format(source))

        return None
    
def resolve_model_and_apply_xml_changes(orig_model, sed_doc, working_dir,
                                        apply_xml_model_changes=True, save_to_file=True,
                                        ):
    """ Resolve the source of a model and, optionally, apply XML changes to the model.

    Args:
        model (:obj:`Model`): model whose ``source`` is one of the following

            * A path to a file
            * A URL
            * A MIRIAM URN for an entry in the BioModelsl database (e.g., ``urn:miriam:biomodels.db:BIOMD0000000012``)
            * A reference to another model, using the ``id`` of the other model (e.g., ``#other-model-id``).
              In this case, the model also inherits changes from the parent model.

        sed_doc (:obj:`SedDocument`): parent SED document; used to resolve sources defined by reference to other models
        working_dir (:obj:`str`): working directory of the SED document (path relative to which models are located)
        apply_xml_model_changes (:obj:`bool`, optional): if :obj:`True`, apply any model changes specified in the SED-ML file before
            calling :obj:`task_executer`.
        save_to_file (:obj:`bool`): whether to save the resolved/modified model to a file       
   
    Raise:
        ValueError    
        
    Returns:
        :obj:`tuple`:

            * :obj:`Model`: modified model (changes are removed, and the source is set to the path to the modified temporary model)
            * :obj:`str`: temporary path to the source of the modified model, if the model needed to be resolved from
              a remote source of modified
            * :obj:`etree._Element`: element tree for the resolved/modified model
    """
    model = orig_model.clone()
    model_etree=None
    # resolve model
    try:
        temp_model_source = resolve_model(model, sed_doc, working_dir)
    except Exception as exception:
        raise ValueError('The model could not be resolved: {}'.format(str(exception)))
    
    # apply changes to model
    if apply_xml_model_changes and model.isSetLanguage() and is_model_language_encoded_in_xml(model.getLanguage ()):
        # read model from file
        try:
            model_etree = etree.parse(model.getSource())
        except Exception as exception:
            raise ValueError('The model could not be parsed because the model is not a valid XML document: {}'.format(str(exception)))

        if model.getListOfChanges () :
            # apply changes
            try:
                apply_changes_to_xml_model(model, model_etree, sed_doc, working_dir)
            except Exception as exception:
                raise ValueError('The model could not be modified: {}'.format(str(exception)))
            model.getListOfChanges ().clear()
    # write model to file
    if save_to_file:
        if temp_model_source is None:
            modified_model_file, temp_model_source = tempfile.mkstemp(suffix='.xml',prefix=model.getId()+"_", dir=os.path.dirname(model.getSource()))
            os.close(modified_model_file)
            model.setSource(temp_model_source)
        
        model_etree.write(model.getSource(),
                          xml_declaration=True,
                          encoding="utf-8",
                          standalone=False,
                          )
        

    return model, model.getSource(), model_etree

def apply_changes_to_xml_model(model, model_etree,sed_doc=None, working_dir=None,
                               variable_values=None, range_values=None, validate_unique_xml_targets=True):
    """ Modify an XML-encoded model according to a model change

    Side effects:
        * Modifies the model_etree

    Args:
        model (:obj:`Model`): model
        model_etree (:obj:`etree._ElementTree`): element tree for model
        sed_doc (:obj:`SedDocument`, optional): parent SED document; used to resolve sources defined by reference to other models;
            required for compute changes
        working_dir (:obj:`str`, optional): working directory of the SED document (path relative to which models are located);
            required for compute changes
        variable_values (:obj:`dict`, optional): dictionary which contains the value of each variable of each
            compute model change
        range_values (:obj:`dict`, optional): dictionary which contains the value of each range of each
            set value compute model change
        validate_unique_xml_targets (:obj:`bool`, optional): whether to validate the XML targets match
            uniue objects
    
    Raise:
        NotImplementedError, ValueError
            
    """

    # First pass:  Must-be-XML changes:
    non_xml_changes = []
    for change in model.getListOfChanges ():
        if change.isSedChangeAttribute():
            obj_xpath, sep, attr = change.getTarget ().rpartition('/@')
            if sep != '/@':
                # change.setModelReference(model.getId())
                non_xml_changes.append(change)
                continue
            # get object to change
            obj_xpath, sep, attr = change.getTarget ().rpartition('/@')
            if sep != '/@':
                raise NotImplementedError('target ' + change.getTarget () + ' cannot be changed by XML manipulation, as the target '
                                          'is not an attribute of a model element')
            objs = eval_xpath(model_etree, obj_xpath, CELLML2NAMESPACE)
            if validate_unique_xml_targets and len(objs) != 1:
                raise ValueError('xpath {} must match a single object'.format(obj_xpath))

            ns_prefix, _, attr = attr.rpartition(':')
            if ns_prefix:
                ns = CELLML2NAMESPACE.get(ns_prefix, None)
                if ns is None:
                    raise ValueError('No namespace is defined with prefix `{}`'.format(ns_prefix))
                attr = '{{{}}}{}'.format(ns, attr)

            # change value
            for obj in objs:
                obj.set(attr, change.getNewValue () )

        elif change.isSedComputeChange ():
            if variable_values is None:
                model_etrees = {model.getId(): model_etree}
                iter_variable_values = get_values_of_variable_model_xml_targets_of_model_change(change, sed_doc, model_etrees, working_dir)
            else:
                iter_variable_values = variable_values

            # calculate new value
            new_value = calc_compute_model_change_new_value(change, variable_values=iter_variable_values, range_values=range_values)
            if new_value == int(new_value):
                new_value = str(int(new_value))
            else:
                new_value = str(new_value)

            # get object to change
            obj_xpath, sep, attr = change.getTarget ().rpartition('/@')
            if sep != '/@':
                # Save this for the next pass:
                # change.setModelReference(model.getId())
                change.setNewValue (new_value)
                non_xml_changes.append(change)
                continue
            objs = eval_xpath(model_etree, obj_xpath, CELLML2NAMESPACE)
            if validate_unique_xml_targets and len(objs) != 1:
                raise ValueError('xpath {} must match a single object'.format(obj_xpath))

            ns_prefix, _, attr = attr.rpartition(':')
            if ns_prefix:
                ns = CELLML2NAMESPACE.get(ns_prefix, None)
                if ns is None:
                    raise ValueError('No namespace is defined with prefix `{}`'.format(ns_prefix))
                attr = '{{{}}}{}'.format(ns, attr)

            # change value
            for obj in objs:
                obj.set(attr, new_value)
        else:
            raise NotImplementedError('Change{} of type {} is not supported.'.format(
                ' ' + change.getId() if change.getId() else '', change.getTypeCode ()))

    return None

def get_values_of_variable_model_xml_targets_of_model_change(change, sed_doc, model_etrees, working_dir):
    """ Get the values of the model variables of a compute model change

    Args:
        change (:obj:`ComputeModelChange`): compute model change
        sed_doc (:obj:`SedDocument`): SED document
        model_etrees (:obj:`dict` of :obj:`str` to :obj:`etree._Element`): map from the ids of models to element
            trees of their sources
        working_dir (:obj:`str`): working directory of the SED document (path relative to which models are located)
    Raise:
        ValueError
    Returns:
        :obj:`dict`: dictionary which contains the value of each variable of each
            compute model change
    """
    variable_values = {}
    for variable in change.getListOfVariables():
        variable_model = variable.model
        if variable_model.getId() not in model_etrees:
            try:
                copy_variable_model, temp_model_source, variable_model_etree= resolve_model_and_apply_xml_changes(
                    variable_model, sed_doc, working_dir,
                    apply_xml_model_changes=True,
                    save_to_file=False)
                model_etrees[variable_model.getId()] = variable_model_etree
            except Exception as exception:
                raise ValueError('The model could not be resolved: {}'.format(str(exception)))

            if temp_model_source:
                os.remove(temp_model_source)
        try:
            variable_values[variable.getId()] = get_value_of_variable_model_xml_targets(variable, model_etrees)
        except Exception as exception:
            raise ValueError('The value of variable {} could not be obtained: {}'.format(variable.getId(), str(exception)))

    return variable_values

def calc_compute_model_change_new_value(setValue, variable_values=None, range_values=None):
    """ Calculate the new value of a compute model change

    Args:
        change (:obj:`ComputeModelChange`): change
        variable_values (:obj:`dict`, optional): dictionary which contains the value of each variable of each
            compute model change
        range_values (:obj:`dict`, optional): dictionary which contains the value of each range of each
            set value compute model change
    Raise:
        ValueError
    Returns:
        :obj:`float`: new value
    """
    compiled_math = compile_math(libsedml.formulaToString(setValue.getMath()))

    workspace = {}

    if setValue.isSetRange ():
        workspace[setValue.getRange ()] = range_values.get(setValue.getRange (), None)
        if workspace[setValue.getRange ()] is None:
            raise ValueError('Value of range `{}` is not defined.'.format(setValue.getRange ()))

    for param in setValue.getListOfParameters ():
        workspace[param.getId()] = param.getValue()

    for var in setValue.getListOfVariables ():
        workspace[var.getId()] = variable_values.get(var.getId(), None)
        if workspace[var.getId()] is None:
            raise ValueError('Value of variable `{}` is not defined.'.format(var.getId()))

    return eval_math(libsedml.formulaToString(setValue.getMath()), compiled_math, workspace)

def calc_data_generator_results(data_generator, variable_results):
    """ Calculate the results of a data generator from the results of its variables

    Args:
        data_generator (:obj:`DataGenerator`): data generator
        variable_results (:obj:`VariableResults`): results for the variables of the data generator
    Raise:
        NotImplementedError
    Returns:
        :obj:`numpy.ndarray`: result of data generator
    """
    var_shapes = set()
    max_shape = []
    for var in data_generator.getListOfVariables():
        var_res = variable_results[var.getId()]
        var_shape = var_res.shape
        if not var_shape and var_res.size:
            var_shape = (1,)
        var_shapes.add(var_shape)

        max_shape = max_shape + [1 if max_shape else 0] * (var_res.ndim - len(max_shape))
        for i_dim in range(var_res.ndim):
            max_shape[i_dim] = max(max_shape[i_dim], var_res.shape[i_dim])

    if len(var_shapes) > 1:
        print('Variables for data generator {} do not have consistent shapes'.format(data_generator.getId()),
             )

    compiled_math = compile_math(libsedml.formulaToString(data_generator.getMath()))

    workspace = {}
    for param in data_generator.getListOfParameters():
        workspace[param.getId()] = param.getValue()

    if not var_shapes:
        value = eval_math(libsedml.formulaToString(data_generator.getMath()), compiled_math, workspace)
        result = numpy.array(value)

    else:
        for aggregate_func in AGGREGATE_MATH_FUNCTIONS:
            if re.search(aggregate_func + r' *\(', libsedml.formulaToString(data_generator.getMath())):
                msg = 'Evaluation of aggregate mathematical functions such as `{}` is not supported.'.format(aggregate_func)
                raise NotImplementedError(msg)

        padded_var_shapes = []
        for var in data_generator.getListOfVariables():
            var_res = variable_results[var.getId()]
            padded_var_shapes.append(
                list(var_res.shape)
                + [1 if var_res.size else 0] * (len(max_shape) - var_res.ndim)
            )

        result = numpy.full(max_shape, numpy.nan)
        n_dims = result.ndim
        for i_el in range(result.size):
            el_indices = numpy.unravel_index(i_el, result.shape)

            vars_available = True
            for var, padded_shape in zip(data_generator.getListOfVariables(), padded_var_shapes):
                var_res = variable_results[var.getId()]
                if var_res.ndim == 0:
                    if i_el == 0 and var_res.size:
                        workspace[var.getId()] = var_res.tolist()
                    else:
                        vars_available = False
                        break

                else:
                    for x, y in zip(padded_shape, el_indices):
                        if (y + 1) > x:
                            vars_available = False
                            break
                    if not vars_available:
                        break

                    workspace[var.getId()] = var_res[el_indices[0:var_res.ndim]]

            if not vars_available:
                continue

            result_el = eval_math(libsedml.formulaToString(data_generator.getMath()), compiled_math, workspace)

            if n_dims == 0:
                result = numpy.array(result_el)
            else:
                result.flat[i_el] = result_el

    return result

def resolve_range(range, model_etrees=None):
    """ Resolve the values of a range

    Args:
        range (:obj:`Range`): range
        model_etrees (:obj:`dict` of :obj:`str` to :obj:`etree._Element`): map from the ids of models to element
            trees of their sources; required to resolve variables of functional ranges

    Returns:
        :obj:`list` of :obj:`float`: values of the range

    Raises:
        :obj:`NotImplementedError`: if range isn't an instance of :obj:`UniformRange`, :obj:`VectorRange`,
            or :obj:`FunctionalRange`.
    """
    if range.isSedUniformRange ():
        if range.getType() == 'linear':
            return numpy.linspace(range.getStart(), range.getEnd(), range.getNumberofSteps() + 1).tolist()

        elif range.getType() == 'log':
            return numpy.logspace(numpy.log10(range.getStart()), numpy.log10(range.getEnd()), range.getNumberofSteps() + 1).tolist()

        else:
            raise NotImplementedError('UniformRanges of type `{}` are not supported.'.format(range.getType()))

    elif range.isSedVectorRange ():
        return range.getValues()

    elif range.isSedFunctionalRange ():
        # compile math
        compiled_math = compile_math(libsedml.formulaToString(range.getMath()))

        # setup workspace to evaluate math
        workspace = {}
        for param in range.getListOfParameters ():
            workspace[param.getId()] = param.getValue()

        for var in range.getListOfVariables ():
            if var.isSetSymbol ():
                raise NotImplementedError('Symbols are not supported for variables of functional ranges')
            if model_etrees[var.getModelReference()] is None:
                raise NotImplementedError('Functional ranges that involve variables of non-XML-encoded models are not supported.')
            workspace[var.getId()] = get_value_of_variable_model_xml_targets(var, model_etrees)

        # calculate the values of the range
        values = []
        for child_range_value in resolve_range(range.getRange(), model_etrees=model_etrees):
            workspace[range.getRange().getId()] = child_range_value

            value = eval_math(libsedml.formulaToString(range.getMath()), compiled_math, workspace)
            values.append(value)

        # return values
        return values

    else:
        raise NotImplementedError('Ranges of type `{}` are not supported.'.format(range.getTypeCode()))