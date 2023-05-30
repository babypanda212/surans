# project related imports
from lib.Turb_solver import solve_turb_model
from lib import params_class_base as pc


params, fparams, output = pc.initialize_project_variables()

solve_turb_model(fparams, params, output)





