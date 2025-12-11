import os
import time
import csv

from mtz_atsp_solvers import solve_MTZ_cplex, solve_MTZ_gurobi


# ============================================================
# LECTOR ATSP
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
# CORRER TODAS LAS INSTANCIAS AUTOMÁTICAMENTE
# ============================================================
def run_all(folder):
    print(f"Carpeta detectada: {folder}")

    files = [f for f in os.listdir(folder) if f.endswith(".atsp")]
    files.sort()

    print("\nInstancias encontradas:")
    for f in files:
        print("  -", f)

    results = []

    for file in files:
        path = os.path.join(folder, file)
        print(f"\n=== MTZ Resolviendo {file} ===")

        cost = load_atsp(path)
        n = len(cost)

        # GUROBI
        print("   MTZ - Gurobi")
        g_vars, g_cons, g_time, g_obj, g_bound, g_gap = solve_MTZ_gurobi(cost)

        # CPLEX
        print("   MTZ - CPLEX")
        c_vars, c_cons, c_time, c_obj, c_bound, c_gap = solve_MTZ_cplex(cost)

        results.append([
            file, n,
            g_vars, g_cons, g_time, g_gap, g_obj, g_bound,
            c_vars, c_cons, c_time, c_gap, c_obj, c_bound
        ])

    # Guardar resultados
    with open("resultados_MTZ.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Instancia", "N",
            "Gurobi_Vars", "Gurobi_Cons", "Gurobi_Time", "Gurobi_Gap",
            "Gurobi_Obj", "Gurobi_Bound",
            "CPLEX_Vars", "CPLEX_Cons", "CPLEX_Time", "CPLEX_Gap",
            "CPLEX_Obj", "CPLEX_Bound"
        ])
        writer.writerows(results)

    print("\nCSV generado: resultados_MTZ.csv")


# ============================================================
# EJECUCIÓN AUTOMÁTICA
# ============================================================
if __name__ == "__main__":
    FOLDER = "instanciasdou"
    run_all(FOLDER)
