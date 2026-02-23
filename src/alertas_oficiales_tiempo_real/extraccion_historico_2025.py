import argparse
import requests
import os
from datetime import datetime
import sys
import json


# ==============================
# CONFIGURACIÓN DATASET
# ==============================
BASE_URL = "https://data.ny.gov/resource/7kct-peq7.json"
#fuente : DATA.NY.GOV
# ejecutable por consola con la siguiente instruccion: cd hasta donde está el .py y 
# python src/alertas_oficiales_tiempo_real/extraccion_historico_2025.py --start 2025-01-01 --end 2025-01-07 --output ./temp(temp es opcional es donde te saca los json)

def fetch_data(start_date, end_date, limit=50000):
    """
    Extrae datos históricos usando paginación.
    """
    all_results = []
    offset = 0

    while True:
        params = {
            # IMPORTANTE: cambiar nombre de columna si fuera necesario
            "$where": f"date between '{start_date}' and '{end_date}'",
            "$limit": limit,
            "$offset": offset
        }

        print(f"Descargando registros con offset {offset}...")

        response = requests.get(BASE_URL, params=params)

        if response.status_code != 200:
            print(f"Error en la petición: {response.status_code}")
            sys.exit(1)

        data = response.json()

        if not data:
            break

        all_results.extend(data)
        offset += limit

    return all_results


def save_raw(data, output_base, start_date, end_date):
    """
    Guarda los datos sin modificar en formato JSON.
    """
    now = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

    path = os.path.join(
        output_base,
        "source=historical",
        f"range={start_date}_to_{end_date}"
    )

    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, f"extracted_at={now}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"Datos guardados en: {file_path}")


def validate_date(date_str):
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError("Formato de fecha inválido. Use YYYY-MM-DD.")
    return date_str


def main():
    parser = argparse.ArgumentParser(
        description="Extracción histórica de avisos oficiales MTA"
    )

    parser.add_argument("--start", required=True, type=validate_date,
                        help="Fecha inicio (YYYY-MM-DD)")

    parser.add_argument("--end", required=True, type=validate_date,
                        help="Fecha fin (YYYY-MM-DD)")

    parser.add_argument("--output", required=True,
                        help="Ruta base donde guardar los datos RAW")

    args = parser.parse_args()

    print("Iniciando extracción histórica...")
    data = fetch_data(args.start, args.end)

    print(f"Total registros descargados: {len(data)}")

    save_raw(data, args.output, args.start, args.end)


if __name__ == "__main__":
    main()