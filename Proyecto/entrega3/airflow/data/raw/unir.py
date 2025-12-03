import pandas as pd
import os

# Nombre de archivos
ARCHIVO_BASE = "transacciones.parquet"
ARCHIVO_BATCH = "batch_t1.parquet"

def main():
    # Obtener carpeta del script
    carpeta = os.path.dirname(os.path.abspath(__file__))

    path_base = os.path.join(carpeta, ARCHIVO_BASE)
    path_batch = os.path.join(carpeta, ARCHIVO_BATCH)

    print(f"📂 Leyendo archivo base: {path_base}")
    df_trans = pd.read_parquet(path_base)

    print(f"📂 Leyendo batch nuevo: {path_batch}")
    df_batch = pd.read_parquet(path_batch)

    print("🔄 Concatenando archivos...")
    df_final = pd.concat([df_trans, df_batch], ignore_index=True)

    print("💾 Guardando archivo actualizado...")
    df_final.to_parquet(path_base, index=False)

    print("✔️ Listo! transacciones.parquet fue actualizado.")

if __name__ == "__main__":
    main()