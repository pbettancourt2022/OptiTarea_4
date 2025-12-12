import os
import time
import csv

from mtz_atsp_solvers import solve_MTZ_gurobi  # Solo importamos Gurobi


# ============================================================
# LECTOR ATSP
# ============================================================
def load_atsp(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    data = []
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
        raise ValueError(f"Error leyendo {filename}, se esperaban {n*n} números y llegaron {len(buffer)}")

    cost = []
    idx = 0
    for i in range(n):
        cost.append(buffer[idx:idx+n])
        idx += n

    return cost


# ============================================================
# CORRER TODAS LAS INSTANCIAS SOLO CON GUROBI
# ============================================================
def run_all(folder):
    print(f"Carpeta detectada: {folder}")
    
    # Buscar archivos ATSP
    files = [f for f in os.listdir(folder) if f.endswith(".atsp")]
    files = sorted(files, key=lambda f: os.path.getsize(os.path.join(folder, f)))

    print("\nInstancias encontradas:")
    for f in files:
        print("  -", f)

    results = []

    for file in files:
        path = os.path.join(folder, file)
        print(f"\n=== MTZ Resolviendo {file} con Gurobi ===")

        cost = load_atsp(path)
        n = len(cost)

        # Ejecutar solver Gurobi
        g_vars, g_cons, g_time, g_obj, g_bound, g_gap = solve_MTZ_gurobi(cost)

        # Guardar fila de resultados
        results.append([
            file, n,
            g_vars, g_cons, g_time, g_gap, g_obj, g_bound
        ])

    # Guardar CSV
    out_name = "resultados_MTZ_gurobi.csv"

    with open(out_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Instancia", "N",
            "Gurobi_Vars", "Gurobi_Cons", "Gurobi_Time", "Gurobi_Gap",
            "Gurobi_Obj", "Gurobi_Bound"
        ])
        writer.writerows(results)

    print(f"\nCSV generado: {out_name}")


# ============================================================
# EJECUCIÓN AUTOMÁTICA
# ============================================================
if __name__ == "__main__":
    FOLDER = "instanciasdou"
    run_all(FOLDER)


