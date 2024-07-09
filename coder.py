from libcellml import Generator, GeneratorProfile, Printer
from analyser import parse_model,analyse_model_full,validate_model,resolve_imports,_ext_var_dic,analyse_model
import os
"""
=================
Code generation
=================
This module contains functions for generating code from a CellML model.

The following functions are defined:
    * writeCellML: write a CellML model to a CellML file.
    * writePythonCode: generate python file from a CellML model.
"""

def writeCellML(model, full_path):  
    """ 
    Write a CellML model to a CellML file.

    Parameters
    ----------
    model: Model
        The CellML model to be written.
    full_path: str
        The full path of the CellML file (including the file name and extension).

    Side effect
    -----------
    The CellML model is written to the specified file.
    """
    
    printer = Printer()
    serialised_model = printer.printModel(model) 
    with open(full_path, "w") as f:
        f.write(serialised_model)   

    print('CellML model saved to:',full_path)

def writeCellML_flat(model, full_path, base_dir,external_variables_info={},strict_mode=True):  
    """ 
    Write a flattend CellML model to a CellML file.

    Parameters
    ----------
    model: Model
        The CellML model to be written.
    full_path: str
        The full path of the CellML file (including the file name and extension).
    base_dir: str
        The base directory of the CellML model.
    strict_mode: bool
        If True, the model is checked against the CellML 2.0 specification.

    Side effect
    -----------
    The CellML model is written to the specified file.
    """
    modelIsValid,issues_validate=validate_model(model)
    if modelIsValid:
        importer,issues_import=resolve_imports(model, base_dir,strict_mode)
        if importer:
            flatModel=importer.flattenModel(model)
            if not flatModel:
                return None, issues_import
            else:  
                printer = Printer()
                serialised_model = printer.printModel(flatModel) 
                with open(full_path, "w") as f:
                    f.write(serialised_model)   

                print('CellML model saved to:',full_path)
                try:
                    external_variables_dic=_ext_var_dic(flatModel,external_variables_info)
                except ValueError as err:
                    return None, str(err)           
                analyser,issues_analyse=analyse_model(flatModel,external_variables_dic)
                issues=issues_validate+issues_import+issues_analyse
                
                return flatModel, issues
        else:
            return None, issues_import
    else:
        return None, issues_validate    

def writePythonCode(analyser, full_path):
    """ 
    Generate Python code from a CellML model
    and write the code to the specified file.

    Parameters
    ----------
    analyser: Analyser
        The Analyser instance of the CellML model.
    full_path: str
        The full path of the python file (including the file name and extension).

    Side effect
    -----------
    The python code is written to the specified file.
    """

    generator = Generator()
    generator.setModel(analyser.model())
    profile = GeneratorProfile(GeneratorProfile.Profile.PYTHON)
    generator.setProfile(profile)
    implementation_code_python = generator.implementationCode()                   
    with open(full_path, "w") as f:
        f.write(implementation_code_python)

def toCellML2(oldPath, newPath, external_variables_info={},strict_mode=True, py_full_path=None):
    """ 
    Convert a CellML 1.X model to CellML 2.0.

    Parameters
    ----------
    oldPath: str
        The full path of the CellML 1.X file (including the file name and extension).
    newPath: str
        The full path of the CellML 2.0 file (including the file name and extension).
    external_variables_info: dict
        A dictionary of external variables information, in the format of {id:{'component': , 'name': }}.
        Empty by default.
    strict_mode: bool
        If True, the model is checked against the CellML 2.0 specification.
    py_full_path: str
        The full path of the python file (including the file name and extension).    

    Side effect
    -----------
    The new CellML 2.0 model is written to the specified file.
    If the model is valid, the python code is written to the specified file.
    """
    try:
        model_parse, issues=parse_model(oldPath, False)
    except Exception as e:
        print(e)  
    print(issues)
    writeCellML(model_parse,newPath)
    base_dir=os.path.dirname(newPath)
    analyser,issues=analyse_model_full(model_parse,base_dir,external_variables_info,strict_mode)
    print(issues)
    if py_full_path is not None:
        writePythonCode(analyser, py_full_path)
    