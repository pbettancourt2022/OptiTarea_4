import os
import time
import csv

from docplex.mp.model import Model
import gurobipy as gp
from gurobipy import GRB


# ============================================================
#  LECTOR ATSP
# ============================================================
def load_atsp(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    data = []
    reading = False

    for line in lines:
        line = line.strip()
        if line.startswith("EDGE_WEIGHT_SECTION"):
            reading = True
            continue
        if reading:
            if line == "EOF":
                break
            row = list(map(int, line.split()))
            data.append(row)
    return data


# ============================================================
#  SOLVER: GG - Gurobi
# ============================================================
def solve_GG_gurobi(cost):
    n = len(cost)
    model = gp.Model("ATSP_GG")
    model.Params.LogToConsole = 0  # sin ruido

    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    u = model.addVars(n, lb=0, ub=n - 1, vtype=GRB.CONTINUOUS, name="u")

    # Objetivo
    model.setObjective(
        gp.quicksum(cost[i][j] * x[i, j] for i in range(n) for j in range(n)),
        GRB.MINIMIZE
    )

    # Grado de entrada y salida
    for i in range(n):
        model.addConstr(sum(x[i, j] for j in range(n)) == 1)
        model.addConstr(sum(x[j, i] for j in range(n)) == 1)
        model.addConstr(x[i, i] == 0)

    # Restricciones GG
    for i in range(n):
        for j in range(n):
            if i != j:
                model.addConstr(u[j] >= u[i] + 1 - n * (1 - x[i, j]))

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
#  SOLVER: GG - CPLEX
# ============================================================
def solve_GG_cplex(cost):
    n = len(cost)
    model = Model("ATSP_GG")
    model.context.solver.log_output = False

    x = model.binary_var_matrix(n, n, name="x")
    u = model.continuous_var_list(n, lb=0, ub=n - 1, name="u")

    # Objetivo
    model.minimize(sum(cost[i][j] * x[i, j] for i in range(n) for j in range(n)))

    # Grado
    for i in range(n):
        model.add_constraint(sum(x[i, j] for j in range(n)) == 1)
        model.add_constraint(sum(x[j, i] for j in range(n)) == 1)
        model.add_constraint(x[i, i] == 0)

    # GG
    for i in range(n):
        for j in range(n):
            if i != j:
                model.add_constraint(u[j] >= u[i] + 1 - n * (1 - x[i, j]))

    model.parameters.timelimit = 3600

    t0 = time.time()
    sol = model.solve()
    total_time = time.time() - t0

    vars_count = model.number_of_variables
    cons_count = model.number_of_constraints

    if sol is not None:
        best_obj = model.objective_value
        best_bound = model.solve_details.best_bound
        mip_gap = model.solve_details.mip_relative_gap
    else:
        best_obj = None
        best_bound = None
        mip_gap = 1.0

    return vars_count, cons_count, total_time, best_obj, best_bound, mip_gap


# ============================================================
#  MAIN: CORRER TODAS LAS INSTANCIAS DE UNA CARPETA
# ============================================================
def run_all(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".atsp")]
    files.sort()  # orden alfabético

    results = []

    for file in files:
        path = os.path.join(folder, file)
        print(f"\n=== Resolviendo {file} ===")

        cost = load_atsp(path)
        n = len(cost)

        # GUROBI
        print("   GG - Gurobi")
        g_vars, g_cons, g_time, g_obj, g_bound, g_gap = solve_GG_gurobi(cost)

        # CPLEX
        print("   GG - CPLEX")
        c_vars, c_cons, c_time, c_obj, c_bound, c_gap = solve_GG_cplex(cost)

        results.append([
            file, n,
            g_vars, g_cons, g_time, g_gap, g_obj, g_bound,
            c_vars, c_cons, c_time, c_gap, c_obj, c_bound
        ])

    # Guardar CSV
    with open("resultados_GG.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Instancia", "N",
            "Gurobi_Vars", "Gurobi_Cons", "Gurobi_Time", "Gurobi_Gap",
            "Gurobi_Obj", "Gurobi_Bound",
            "CPLEX_Vars", "CPLEX_Cons", "CPLEX_Time", "CPLEX_Gap",
            "CPLEX_Obj", "CPLEX_Bound"
        ])
        writer.writerows(results)

    print("\nCSV generado: resultados_GG.csv")


# ============================================================
#  EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    folder = input("Carpeta con instancias ATSP: ")
    run_all(folder)
