import os
import time
import csv

import gurobipy as gp
from gurobipy import GRB


# ============================================================
#  LECTOR ATSP
# ============================================================
def load_atsp(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    reading = False
    buffer = []

    for line in lines:
        line = line.strip()

        if line.startswith("DIMENSION"):
            n = int(line.split()[-1])

        if line.startswith("EDGE_WEIGHT_SECTION"):
            reading = True
            continue
        
        if reading:
            if line == "EOF":
                break
            nums = list(map(int, line.split()))
            buffer.extend(nums)

    if len(buffer) != n * n:
        raise ValueError(f"Error: matriz incompleta en {filename}. "
                         f"Se leyeron {len(buffer)} números para un tamaño {n}x{n}")

    cost = []
    idx = 0
    for i in range(n):
        cost.append(buffer[idx:idx+n])
        idx += n

    return cost



# ============================================================
#  SOLVER: GG - SOLO Gurobi (CORREGIDO)
# ============================================================
def solve_GG_gurobi(cost):
    # -> ahora es una versión MTZ que funciona correctamente para ATSP
    n = len(cost)
    model = gp.Model("ATSP_MTZ")
    model.Params.LogToConsole = 0

    # Variables x(i,j)
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    # prohibir autolazos
    for i in range(n):
        x[i, i].ub = 0

    # Variables MTZ u (continuas)
    u = model.addVars(n, lb=0.0, ub=n - 1.0, vtype=GRB.CONTINUOUS, name="u")

    # Objetivo (solo i != j)
    model.setObjective(
        gp.quicksum(cost[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j),
        GRB.MINIMIZE
    )

    # Restricciones de grado (salida e entrada)
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1, name=f"out_{i}")
        model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1, name=f"in_{i}")

    # Fijar u[0] = 0 para romper simetría
    model.addConstr(u[0] == 0, name="u0")

    # Restricciones MTZ: para i != j y i,j != 0 (clásico)
    # Forma equivalente y segura: u[i] - u[j] + n * x[i,j] <= n - 1   for i != j and i != 0 and j != 0
    # También se suele aplicar para todos i != j (funciona igualmente si u[0] está fijado)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f"mtz_{i}_{j}")

    # Parámetros
    model.Params.TimeLimit = 3600

    # Optimizar
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
        # si no hay solución factible, devolver inf en gap para tu CSV consistente
        mip_gap = float("inf")

    return vars_count, cons_count, total_time, best_obj, best_bound, mip_gap


# ============================================================
#  MAIN SOLO Gurobi
# ============================================================
def run_all_gurobi(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".atsp")]
    files = sorted(files, key=lambda f: os.path.getsize(os.path.join(folder, f)))

    results = []

    for file in files:
        path = os.path.join(folder, file)
        print(f"\n=== Resolviendo {file} ===")

        cost = load_atsp(path)
        n = len(cost)

        g_vars, g_cons, g_time, g_obj, g_bound, g_gap = solve_GG_gurobi(cost)

        results.append([
            file, n,
            g_vars, g_cons, g_time, g_gap, g_obj, g_bound
        ])

    # Guardar CSV
    with open("resultados_GG_gurobi.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Instancia", "N",
            "Gurobi_Vars", "Gurobi_Cons", "Gurobi_Time", "Gurobi_Gap",
            "Gurobi_Obj", "Gurobi_Bound"
        ])
        writer.writerows(results)

    print("\nCSV generado: resultados_GG_gurobi.csv")



# ============================================================
#  EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    folder = input("Carpeta con instancias ATSP: ")
    run_all_gurobi(folder)
