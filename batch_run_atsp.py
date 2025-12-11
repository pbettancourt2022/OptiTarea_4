# batch_run_atsp.py
# Ejecuta MTZ para inst1.csv ... inst10.csv en formato:
# n,n,"[[coords]]","[[matrix]]",route,opt_value

import pandas as pd
import numpy as np
import ast
from pathlib import Path
from mtz_atsp_solvers import solve_mtz_cplex, solve_mtz_gurobi

INST_DIR = Path("instancias")
OUTPUT_FILE = "resultados_mtz.csv"
TIME_LIMIT = 3600  # 1 hora

def extraer_matriz_desde_csv(path):
    """
    Lee un archivo instX.csv con formato:
    n,n,"[[coords]]","[[mat]]",route,val
    y devuelve SOLO la matriz como numpy array.
    """
    df = pd.read_csv(path, header=None)
    matriz_str = df.iloc[0, 3]   # columna 3 = matriz
    matriz = np.array(ast.literal_eval(matriz_str), dtype=float)
    return matriz


def main():
    archivos = sorted([f for f in INST_DIR.glob("inst*.csv")])

    print("\nInstancias detectadas:")
    for f in archivos:
        print(" -", f.name)

    resultados = []

    for archivo in archivos:
        print("\n=======================")
        print("Procesando:", archivo.name)
        print("=======================")

        C = extraer_matriz_desde_csv(archivo)
        n = C.shape[0]

        num_vars = n*(n-1) + (n-1)
        num_constraints = 2*n + (n-1)*(n-1)

        # ---- CPLEX ----
        print("Ejecutando MTZ - CPLEX...")
        res_cplex = solve_mtz_cplex(C, time_limit=TIME_LIMIT)

        # ---- GUROBI ----
        print("Ejecutando MTZ - Gurobi...")
        res_gurobi = solve_mtz_gurobi(C, time_limit=TIME_LIMIT)

        resultados.append({
            "instance": archivo.name,
            "n": n,

            "cplex_time": res_cplex["time"],
            "cplex_status": res_cplex["status"],
            "cplex_gap": res_cplex["mip_gap_reported"],
            "cplex_best_bound": res_cplex["best_bound"],
            "cplex_obj": res_cplex["objective"],

            "gurobi_time": res_gurobi["time"],
            "gurobi_status": res_gurobi["status"],
            "gurobi_gap": res_gurobi["mip_gap_reported"],
            "gurobi_best_bound": res_gurobi["best_bound"],
            "gurobi_obj": res_gurobi["objective"],

            "num_vars": num_vars,
            "num_constraints": num_constraints
        })

        pd.DataFrame(resultados).to_csv(OUTPUT_FILE, index=False)

    print("\nFIN. Resultados guardados en", OUTPUT_FILE)


if __name__ == "__main__":
    main()
