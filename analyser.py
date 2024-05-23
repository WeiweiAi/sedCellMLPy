from libcellml import  Parser, Validator, Analyser, AnalyserExternalVariable, Importer, cellmlElementTypeAsString, AnalyserModel
import json
import os
from pathlib import Path, PurePath

"""
========
analyser
========
This module provides functions to parse, validate, resolve imports and analyse a CellML model.
It also provides functions to get the type of the model.

Functions
---------
* parse_model(filename, strict_mode=False)
    Parse a CellML file to a CellML model.
* validate_model(model)
    Validate a CellML model.
* resolve_imports(model, base_dir,strict_mode=True)
    Resolve the imports of a CellML model.
* analyse_model(flatModel,external_variables_dic={})
    Analyse a flattened CellML model.
* analyse_model_full(model,base_dir,external_variables_info={},strict_mode=True)
    Fully validate and analyse a cellml model.
* get_mtype(analyser)
    Get the type of the model.

"""

def _dump_issues(source_method_name, logger):
    """"
    Dump the issues found by the parser, validator, importer or analyser.

    Parameters
    ----------
    source_method_name: str
        The name of the source method that generated the logger.
    logger: Parser, Validator, Importer or Analyser
        The libcellml logger instance.

    Returns
    -------
    str
        The issues found by the logger.
        The format is for html display.
        When no issues are found, the string is empty.
    """

    issues = ''
    if logger.issueCount() > 0:
        issue_sum = 'The method "{}" found {} issues:'.format(source_method_name, logger.issueCount())
        issues=issues+issue_sum+'<br>'
        for i in range(0, logger.issueCount()):
            issuei=logger.issue(i).description()
            issues=issues+issuei+'<br>'
            issues=issues+'The element type is: '+ cellmlElementTypeAsString(logger.issue(i).item().type())+'<br>'
            issues=issues+'The reference rule is: '+logger.issue(i).referenceHeading( )+'<br>'
    return issues

def parse_model(filename, strict_mode=False):
    """ 
    Parse a CellML file to a CellML model.

    Parameters
    ----------
    filename: str
        The fullpath of the CellML file.
    strict_mode: bool, optional
        Whether to use strict mode to parse the CellML file. Default: False.
        If False, the parser can parse CellML 1.0 and 1.1 files. TODO: check this
    
    Raises
    ------
    FileNotFoundError
        If the CellML file does not exist.

    Returns
    -------
    tuple
        (Model, str)
        The CellML model and the issues found by the parser.
        If issues are found, the model could be None.
    """

    if not os.path.isfile(filename):
        raise FileNotFoundError('Model source file `{}` does not exist.'.format(filename))
    
    parser = Parser(strict_mode)
    with open(filename, 'r') as f:
        model = parser.parseModel(f.read())
    issues = _dump_issues("parse_model", parser)
    if issues !='':
        return model, issues
    else:
        issues="parse_model: No issues found!<br>"
        return model, issues

def validate_model(model):
    """ 
    Validate a CellML model.
    
    Parameters
    ----------
    model: Model
        The CellML model to be validated.

    Returns
    -------
    tuple
        (bool, str)
        Whether the model is valid and the issues found by the validator.
        If issues are found, the model is not valid (False).
    """
    validator = Validator()
    validator.validateModel(model)
    issues = _dump_issues("validate_model", validator)
    if issues=='':
        issues="validate_model: No issues found!<br>"
        return True, issues
    else:
        return False, issues

def resolve_imports(model, base_dir,strict_mode=True):
    """ 
    Resolve the imports of a CellML model.
    
    Parameters
    ----------
    model: Model
        The CellML model for which the import needs to be resolved.
    base_dir: str
        The full path to the directory that import URLs are relative to.
    strict_mode: bool, optional
        Whether to use strict mode to resolve imports. Default: True.

    Returns
    -------
    tuple
        (Importer, str)
        The Importer instance and the issues found by the importer.
        If issues are found, the importer is None.
    """

    importer = Importer(strict_mode)
    # check if base_dir is a string
    if not isinstance(base_dir, str):
        if isinstance(base_dir, Path):
            base_dir = str(base_dir)
        elif isinstance(base_dir,PurePath):
            base_dir = str(base_dir)
        else:
            raise TypeError('The base_dir should be a string or a Path object.')

    importer.resolveImports(model, base_dir)
    issues = _dump_issues("resolve_imports", importer)
    if issues=='' and (not model.hasUnresolvedImports()):
        issues="resolve_imports: No issues found!<br>"
        return importer, issues
    else:
        return importer, issues

def analyse_model(flatModel,external_variables_dic={}):
    """ 
    Analyse a flattened CellML model.
    
    Parameters
    ----------
    flatModel: Model
        The flattened CellML model.
    external_variables_dic: dict, optional
        The dictionary of external variables in the flattened model,
        in the format of {external_variable:[dependency1,dependency2]}.

    Returns
    -------
    tuple
        (Analyser, str)
        The Analyser instance and the issues found by the analyser.
        If issues are found, the analyser is None.
    """ 

    analyser = Analyser()
    for external_variable in external_variables_dic.keys():
        aev=AnalyserExternalVariable(external_variable)
        for dependency in external_variables_dic[external_variable]:
            aev.addDependency(dependency)
        if not analyser.addExternalVariable(aev):        
            return None, "analyse_model: Unable to add external variable {} to the analyser! <br> ".format(external_variable.name())	
    analyser.analyseModel(flatModel)
    issues = _dump_issues("analyse_model", analyser)
    if issues=='':
        return analyser, issues
    else:
        return analyser, issues # TODO: need to check when analyser is not None and issues is not empty.
        #return None, issues
    
def analyse_model_full(model,base_dir,external_variables_info={},strict_mode=True):
    """ 
    Fully validate and analyse a cellml model.
   
    Parameters
    ----------
    model: Model
        The CellML model to be analysed.
    base_dir: str
        The full path to the directory that import URLs are relative to.
    external_variables_info: dict, optional
        The external variables to be specified, in the format of {id:{'component': , 'name': }}
    strict_mode: bool, optional
        Whether to use strict mode to resolve imports. Default: True.

    Returns
    -------
    tuple
        (Analyser, str)
        The Analyser instance and the issues found by the analysing process.
        If issues are found, the analyser is None.
    """
    modelIsValid,issues_validate=validate_model(model)
    if modelIsValid:
        importer,issues_import=resolve_imports(model, base_dir,strict_mode)
        if importer:
            flatModel=importer.flattenModel(model)
            if not flatModel:
                return None, json.dumps(issues_import)
            try:
                external_variables_dic=_ext_var_dic(flatModel,external_variables_info)
            except ValueError as err:
                return None, json.dumps(str(err))            
            analyser,issues_analyse=analyse_model(flatModel,external_variables_dic)
            issues=issues_validate+issues_import+issues_analyse
            if analyser:
                return analyser, json.dumps(issues)
        else:
            issues=issues_validate+issues_import
    else:
        issues=issues_validate
    return None, json.dumps(issues)

def get_mtype(analyser):
    """ Get the type of the model.
    
    Parameters
    ----------
    analyser: Analyser
        The Analyser instance of the CellML model.

    Returns
    -------
    str
        The type of the model.
        The type can be 'unknown', 'algebraic', 'dae', 'invalid', 'nla', 'ode', 
        'overconstrained', 'underconstrainted' or 'unsuitably_constrained'.
        Refer to https://libcellml.org/documentation/v0.5.0/api/classlibcellml_1_1AnalyserModel
  
    """
    return AnalyserModel.typeAsString(analyser.model().type())
    
def _ext_var_dic(flatModel,external_variables_info):
    """
    Create a dictionary of external variables in the flattened model.
    
    Parameters
    ----------
    flatModel: Model
        The flattened CellML model.
    external_variables_info: dict
        The external variables to be specified, in the format of {id:{'component': , 'name': }}

    Raises
    ------
    ValueError
        If an external variable is not found in the flattened model.

    Returns
    -------
    dict
        The dictionary of external variables in the flattened model, in the format of {external_variable:[]}
    
    Notes
    -----
        No dependency is specified for the external variables.
    """

    external_variables_dic={}
    for _, ext_var_info in external_variables_info.items():
        flat_ext_var=flatModel.component(ext_var_info['component']).variable(ext_var_info['name']) # TODO: check if equivalent variables should be considered
        if flat_ext_var:
           external_variables_dic[flat_ext_var]=[]
        else:
            raise ValueError ("The external variable {} in component {} is not found in the flattened model!"
                              .format(ext_var_info['component'],ext_var_info['name']))

    return external_variables_dic