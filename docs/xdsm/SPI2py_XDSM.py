from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT, DOE, IGROUP

# Stage 1: EMI

x = XDSM(use_sfmath=True)

x.add_system("DOE",     DOE, (r"\text{Feasible Layouts}"))

x.add_system("opt",     OPT, r"\text{Optimizer}")
x.add_system("Kinematics", FUNC, r"\text{Kinematics}")
x.add_system("F",       FUNC, (r"\textbf{Objective}", r"\text{Bounding Volume}"))
x.add_system("G",       FUNC, (r"\textbf{Constraints}", r"\text{Physics}"))
x.add_system("LumpedParameterPhysics", IGROUP, (r"\textbf{Lumped Parameter}",r"\text{Head Loss}"))
x.add_system("Projection", FUNC, r"\text{Projection}")
x.add_system("solver",  SOLVER, r"\text{Solver}")


x.add_system("Physics", IGROUP, r"\textbf{Physics}")
x.add_system("Continuum", IGROUP, (r"\textbf{Continuum}", r"\text{Temperature}"))
x.add_system("Finite Element", IGROUP, (r"\textbf{Finite Element}", r"\text{Temperature}"))



x.connect("Kinematics", "LumpedParameterPhysics", "u_{spheres}")
x.connect("LumpedParameterPhysics", "G", "u_{pressure}")
x.connect("opt", "Kinematics", "x")
x.connect("Kinematics", "Projection", "u_{spheres}")

x.connect("FiniteElementPhysics", "solver", r"\mathcal{R}(u_{temp})")
x.connect("solver", "FiniteElementPhysics", "u_{temp}")


x.connect("Kinematics", "F", "u_{spheres}")
x.connect("Projection", "FiniteElementPhysics", "u_{temp}")
x.connect("solver", "G", "u_{temp}")

x.connect("F", "opt", "f")
x.connect("G", "opt", "g")

x.connect("DOE", "opt", r"x_i, y_i, z_i, \alpha_i, \gamma_i, \beta_i")
x.connect("opt", "DOE", r"x_f, y_f, z_f, \alpha_f, \gamma_f, \beta_f")

# x.add_output("opt", r"x_f, y_f, z_f, \alpha_f, \gamma_f, \beta_f")


x.write("mdf_revised")
