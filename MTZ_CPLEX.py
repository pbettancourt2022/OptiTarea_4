import os
import time
import csv

from mtz_atsp_solvers import solve_MTZ_cplex


# ============================================================
# LECTOR ATSP
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
            f"Error leyendo {filename}, se esperaban {n*n} números y llegaron {len(buffer)}"
        )

    cost = []
    idx = 0
    for i in range(n):
        cost.append(buffer[idx:idx+n])
        idx += n

    return cost


# ============================================================
# CORRER TODAS LAS INSTANCIAS AUTOMÁTICAMENTE (solo CPLEX)
# ============================================================
def run_all_cplex(folder):
    print(f"Carpeta detectada: {folder}")
    print("Método seleccionado: CPLEX")

    # Buscar archivos ATSP y ordenarlos por tamaño
    files = [f for f in os.listdir(folder) if f.endswith(".atsp")]
    files = sorted(files, key=lambda f: os.path.getsize(os.path.join(folder, f)))

    print("\nInstancias encontradas:")
    for f in files:
        print("  -", f)

    results = []

    for file in files:
        path = os.path.join(folder, file)
        print(f"\n=== MTZ (CPLEX) Resolviendo {file} ===")

        cost = load_atsp(path)
        n = len(cost)

        # Resolver con CPLEX
        vars_count, cons_count, t_time, obj, bound, gap = solve_MTZ_cplex(cost)

        # Guardar resultados
        results.append([
            file, n,
            vars_count, cons_count, t_time, gap, obj, bound
        ])

    # Guardar CSV
    out_name = "resultados_MTZ_cplex.csv"

    with open(out_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Instancia", "N",
            "CPLEX_Vars", "CPLEX_Cons", "CPLEX_Time",
            "CPLEX_Gap", "CPLEX_Obj", "CPLEX_Bound"
        ])
        writer.writerows(results)

    print(f"\nCSV generado: {out_name}")


# ============================================================
# EJECUCIÓN AUTOMÁTICA
# ============================================================
if __name__ == "__main__":
    FOLDER = "instanciasdou"
    run_all_cplex(FOLDER)

