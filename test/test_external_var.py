import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sedCollector import read_sedml
from sedExecutor import exec_sed_doc
from pathlib import Path

files=['external_var_euler','external_var_cvode']
for file in files:
    full_path=Path('../models/{}.sedml'.format(file)).as_posix()
    doc=read_sedml(full_path)
    working_dir=Path(full_path).parent
    volts=[1]*100+[2]*101
    exec_sed_doc(doc, working_dir, working_dir, rel_out_path='./', external_variables_info={1:{'component': 'test_external_var', 'name': 'external_var'}},
                  external_variables_values=[volts],ss_time={},cost_type=None)