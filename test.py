
from sedCollector import read_sedml
from sedExecutor import exec_sed_doc
from pathlib import Path

full_path="./models/SGLT1_BG.sedml"
doc=read_sedml(full_path)
working_dir=Path("./models/").absolute()
exec_sed_doc(doc, working_dir, working_dir, rel_out_path='data', external_variables_info={},
                  external_variables_values=[],ss_time={},cost_type=None)