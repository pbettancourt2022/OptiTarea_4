import os
import time
import csv
from docplex.mp.model import Model

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
        raise ValueError(
            f"Error: matriz incompleta en {filename}. "
            f"Se leyeron {len(buffer)} números para un tamaño {n}x{n}"
        )

    cost = []
    idx = 0
    for i in range(n):
        cost.append(buffer[idx:idx+n])
        idx += n

    return cost

# ============================================================
#  SOLVER: GG – CPLEX
# ============================================================
def solve_GG_cplex(cost):
    n = len(cost)
    model = Model("ATSP_GG")
    model.context.solver.log_output = True

    # ============================
    #   VARIABLES
    # ============================
    x = model.binary_var_matrix(n, n, name="x")
    # u[i] va de 1 a n, u[0] = 1
    u = [model.continuous_var(lb=1, ub=n, name=f"u{i}") for i in range(n)]

    # ============================
    #   OBJETIVO
    # ============================
    model.minimize(sum(cost[i][j] * x[i, j] for i in range(n) for j in range(n)))

    # ============================
    #   RESTRICCIONES
    # ============================
    for i in range(n):
        model.add_constraint(sum(x[i, j] for j in range(n)) == 1)
        model.add_constraint(sum(x[j, i] for j in range(n)) == 1)
        model.add_constraint(x[i, i] == 0)

    model.add_constraint(u[0] == 1)

    # Subtour elimination desde i,j = 1 a n-1
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.add_constraint(u[j] >= u[i] + 1 - n * (1 - x[i, j]))

    # ============================
    #   HEURÍSTICAS Y PARAMETROS
    # ============================
    mdl_params = model.parameters
    mdl_params.mip.strategy.fpheur = 1          # Feasibility Pump
    mdl_params.mip.strategy.heuristicfreq = 1   # Frecuencia de heurísticas
    mdl_params.mip.strategy.rinsheur = 50       # RINS frecuente
    mdl_params.mip.tolerances.mipgap = 0.1      # Relajar gap de optimización
    mdl_params.timelimit = 3600                 # Tiempo límite de 1 hora

    # ============================
    #   RESOLVER
    # ============================
    t0 = time.time()
    sol = model.solve()
    total_time = time.time() - t0

    # ============================
    #   OBTENER RESULTADOS
    # ============================
    vars_count = model.number_of_variables
    cons_count = model.number_of_constraints

    if model.solve_details is not None:
        status = model.solve_details.status
        best_bound = model.solve_details.best_bound
        mip_gap = model.solve_details.mip_relative_gap
    else:
        status = None
        best_bound = None
        mip_gap = 1.0

    print("Status:", status)
    print("Best bound:", best_bound)
    print("MIP gap:", mip_gap)

    if sol is not None:
        best_obj = model.objective_value
    else:
        best_obj = None
        print("No se encontró solución factible.")

    return vars_count, cons_count, total_time, best_obj, best_bound, mip_gap

# ============================================================
#  MAIN
# ============================================================
def run_all(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".atsp")]
    files = sorted(files, key=lambda f: os.path.getsize(os.path.join(folder, f)))

    results = []

    for file in files:
        path = os.path.join(folder, file)
        print(f"\n=== GG (CPLEX) resolviendo {file} ===")
        cost = load_atsp(path)
        n = len(cost)

        vars_count, cons_count, t_time, obj, bound, gap = solve_GG_cplex(cost)

        results.append([
            file, n,
            vars_count, cons_count, t_time, gap, obj, bound
        ])

    with open("resultados_GG_cplex.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Instancia", "N",
            "CPLEX_Vars", "CPLEX_Cons", "CPLEX_Time",
            "CPLEX_Gap", "CPLEX_Obj", "CPLEX_Bound"
        ])
        writer.writerows(results)

    print("\nCSV generado: resultados_GG_cplex.csv")

# ============================================================
#  EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    folder = input("Carpeta con instancias ATSP: ")
    run_all(folder)

