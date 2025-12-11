import numpy as np
import ast
import time

# -------------------------------------------------------------
# Conversión a matriz numpy
# -------------------------------------------------------------
def convertir_a_matriz(C):
    """
    Acepta:
     - numpy.ndarray
     - lista de listas
     - string tipo "[[...]]"
    Devuelve numpy.ndarray(float)
    """
    if isinstance(C, np.ndarray):
        return C.astype(float)

    if isinstance(C, list):
        return np.array(C, dtype=float)

    if isinstance(C, str):
        return np.array(ast.literal_eval(C), dtype=float)

    raise ValueError("Formato de matriz no reconocido en C")


# -------------------------------------------------------------
#                  MTZ - CPLEX
# -------------------------------------------------------------
def solve_mtz_cplex(C, time_limit=3600, log_output=False):
    try:
        from docplex.mp.model import Model
    except:
        raise ImportError("No se pudo importar docplex (CPLEX).")

    C = convertir_a_matriz(C)
    n = C.shape[0]

    model = Model(name="ATSP_MTZ", log_output=log_output)

    # Variables x(i,j)
    x = model.binary_var_matrix(n, n, name="x")
    # Variables u(i)
    u = model.continuous_var_list(n, lb=0, ub=n-1, name="u")

    # Objetivo
    model.minimize(model.sum(C[i, j] * x[i, j] for i in range(n) for j in range(n)))

    # No ciclos i->i
    for i in range(n):
        model.add_constraint(x[i, i] == 0)

    # Salida única e ingreso único
    for i in range(n):
        model.add_constraint(model.sum(x[i, j] for j in range(n)) == 1)
        model.add_constraint(model.sum(x[j, i] for j in range(n)) == 1)

    # Restricciones MTZ
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.add_constraint(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2)

    model.parameters.timelimit = time_limit

    t0 = time.time()
    sol = model.solve()
    t1 = time.time()

    if sol is None:
        return {
            "time": t1 - t0,
            "status": "no_solution",
            "mip_gap_reported": 100,
            "best_bound": None,
            "objective": None,
            "num_vars": model.number_of_variables,
            "num_constraints": model.number_of_constraints
        }

    gap = model.solve_details.mip_relative_gap
    if gap is None:
        gap = 1.0

    return {
        "time": t1 - t0,
        "status": model.solve_details.status,
        "mip_gap_reported": gap * 100,
        "best_bound": model.solve_details.best_bound,
        "objective": sol.objective_value,
        "num_vars": model.number_of_variables,
        "num_constraints": model.number_of_constraints
    }



# -------------------------------------------------------------
#                  MTZ - GUROBI
# -------------------------------------------------------------
def solve_mtz_gurobi(C, time_limit=3600, log_output=False):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except:
        raise ImportError("No se pudo importar Gurobi.")

    C = convertir_a_matriz(C)
    n = C.shape[0]

    model = gp.Model("ATSP_MTZ")
    if not log_output:
        model.Params.LogToConsole = 0

    model.Params.TimeLimit = time_limit

    # Variables x(i,j)
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    # Variables u(i)
    u = model.addVars(n, lb=0, ub=n-1, vtype=GRB.CONTINUOUS, name="u")

    # Objetivo
    model.setObjective(gp.quicksum(C[i, j] * x[i, j] for i in range(n) for j in range(n)))

    # No ciclos i->i
    for i in range(n):
        model.addConstr(x[i, i] == 0)

    # Salida única e ingreso único
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
        model.addConstr(gp.quicksum(x[j, i] for j in range(n)) == 1)

    # MTZ
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2)

    model.optimize()

    if model.SolCount == 0:
        return {
            "time": model.Runtime,
            "status": "no_solution",
            "mip_gap_reported": 100,
            "best_bound": None,
            "objective": None,
            "num_vars": model.NumVars,
            "num_constraints": model.NumConstrs
        }

    return {
        "time": model.Runtime,
        "status": model.Status,
        "mip_gap_reported": model.MIPGap * 100 if model.MIPGap is not None else None,
        "best_bound": model.ObjBound,
        "objective": model.objVal,
        "num_vars": model.NumVars,
        "num_constraints": model.NumConstrs
    }
