import time
from docplex.mp.model import Model
import gurobipy as gp
from gurobipy import GRB


# ============================================================
#  MTZ - GUROBI
# ============================================================
def solve_MTZ_gurobi(cost):
    n = len(cost)
    model = gp.Model("ATSP_MTZ")
    model.Params.LogToConsole = 0

    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    u = model.addVars(n, lb=0, ub=n-1, vtype=GRB.CONTINUOUS, name="u")

    model.setObjective(
        gp.quicksum(cost[i][j] * x[i, j] for i in range(n) for j in range(n)),
        GRB.MINIMIZE
    )

    for i in range(n):
        model.addConstr(gp.quicksum(x[i,j] for j in range(n)) == 1)
        model.addConstr(gp.quicksum(x[j,i] for j in range(n)) == 1)
        model.addConstr(x[i,i] == 0)

    # MTZ
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + n * x[i, j] <= n-1)

    model.Params.TimeLimit = 3600

    t0 = time.time()
    model.optimize()
    total_time = time.time() - t0

    vars_count = model.NumVars
    cons_count = model.NumConstrs

    if model.SolCount > 0:
        best_obj = model.ObjVal
        best_bound = model.ObjBound
        mip_gap = model.MIPGap
    else:
        best_obj = None
        best_bound = None
        mip_gap = 1.0

    return vars_count, cons_count, total_time, best_obj, best_bound, mip_gap


# ============================================================
#  MTZ - CPLEX
# ============================================================
def solve_MTZ_cplex(cost):
    n = len(cost)
    model = Model("ATSP_MTZ")
    model.context.solver.log_output = False

    x = model.binary_var_matrix(n, n, name="x")
    u = model.continuous_var_list(n, lb=0, ub=n-1, name="u")

    model.minimize(sum(cost[i][j] * x[i,j] for i in range(n) for j in range(n)))

    for i in range(n):
        model.add_constraint(sum(x[i,j] for j in range(n)) == 1)
        model.add_constraint(sum(x[j,i] for j in range(n)) == 1)
        model.add_constraint(x[i,i] == 0)

    # MTZ
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.add_constraint(u[i] - u[j] + n * x[i,j] <= n-1)

    model.parameters.timelimit = 3600

    t0 = time.time()
    sol = model.solve()
    total_time = time.time() - t0

    vars_count = model.number_of_variables
    cons_count = model.number_of_constraints

    if sol is not None:
        best_obj = model.objective_value
        best_bound = model.solve_details.best_bound
        gap = model.solve_details.mip_relative_gap
        if gap is None:
            gap = 1.0
    else:
        best_obj = None
        best_bound = None
        gap = 1.0

    return vars_count, cons_count, total_time, best_obj, best_bound, gap
