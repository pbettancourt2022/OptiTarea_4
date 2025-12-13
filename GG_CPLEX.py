import os
import time
import csv
from docplex.mp.model import Model

# ============================================================
# LECTOR DE INSTANCIAS ATSP
# ============================================================
def load_atsp(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    n = 0
    buffer = []
    reading = False
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
            buffer.extend(map(int, line.split()))

    if len(buffer) != n*n:
        raise ValueError(f"Matriz incompleta: {len(buffer)} números para {n}x{n}")

    cost = [buffer[i*n:(i+1)*n] for i in range(n)]
    return cost

# ============================================================
# SOLVER GG – CPLEX
# ============================================================

def solve_GG_cplex(cost):
    n = len(cost)
    mdl = Model("ATSP_GG")
    mdl.context.solver.log_output = True

    # ========================================================
    # Variables
    # ========================================================
    # Variables de decisión
    x = mdl.binary_var_matrix(n, n, name="x")

    # Variables de flujo (single-commodity flow)
    g = mdl.continuous_var_matrix(n, n, lb=0, name="g")

    # ========================================================
    # Función objetivo
    # ========================================================
    mdl.minimize(
        mdl.sum(cost[i][j] * x[i, j] for i in range(n) for j in range(n))
    )

    # ========================================================
    # Restricciones de asignación (GG base)
    # ========================================================
    for i in range(n):
        mdl.add_constraint(mdl.sum(x[i, j] for j in range(n)) == 1)
        mdl.add_constraint(mdl.sum(x[j, i] for j in range(n)) == 1)
        mdl.add_constraint(x[i, i] == 0)

    # ========================================================
    # Restricciones de flujo – GG (Gavish–Graves)
    # Nodo 0 es el depósito
    # ========================================================
    for i in range(1, n):
        mdl.add_constraint(
            mdl.sum(g[j, i] for j in range(n)) -
            mdl.sum(g[i, j] for j in range(n)) == 1
        )

    mdl.add_constraint(
        mdl.sum(g[0, j] for j in range(n)) -
        mdl.sum(g[j, 0] for j in range(n)) == n - 1
    )

    # ========================================================
    # Enlace flujo–arcos
    # ========================================================
    for i in range(n):
        for j in range(n):
            mdl.add_constraint(g[i, j] <= (n - 1) * x[i, j])

    # ========================================================
    # Parámetros CPLEX
    # ========================================================
    mdl.parameters.mip.tolerances.mipgap = 0.0
    mdl.parameters.timelimit = 3600

    # Resolver
    t0 = time.time()
    sol = mdl.solve()
    t_total = time.time() - t0

    if sol:
        best_obj = mdl.objective_value
        status = mdl.solve_details.status
        mip_gap = mdl.solve_details.mip_relative_gap
        best_bound = mdl.solve_details.best_bound
    else:
        best_obj = None
        status = None
        mip_gap = 1.0
        best_bound = None

    return mdl.number_of_variables, mdl.number_of_constraints, t_total, best_obj, best_bound, mip_gap


# ============================================================
# Ejecutar todas las instancias en una carpeta
# ============================================================
def run_all(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".atsp")],
                   key=lambda f: os.path.getsize(os.path.join(folder, f)))

    results = []
    for file in files:
        path = os.path.join(folder, file)
        print(f"\n=== Resolviendo {file} ===")
        cost = load_atsp(path)
        n = len(cost)
        vars_count, cons_count, t_time, obj, bound, gap = solve_GG_cplex(cost)
        results.append([file, n, vars_count, cons_count, t_time, gap, obj, bound])

    with open("resultados_GG_cplexdou.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Instancia","N","Vars","Cons","Tiempo","Gap","Obj","Bound"])
        writer.writerows(results)

    print("\nCSV generado: resultados_GG_cplexdou.csv")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    folder = input("Carpeta con instancias ATSP: ")
    run_all(folder)

