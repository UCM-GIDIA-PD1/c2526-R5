import os
import time
import logging

from src.ETL.pipelines.preprocess_realtime_lgbm import CACHE_FILE, update_lag_state
from src.common.minio_client import delete_object

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def run_worker():
    log.info("Limpiando estado anterior de lags en MinIO...")
    try:
        delete_object(
            access_key=os.environ["MINIO_ACCESS_KEY"],
            secret_key=os.environ["MINIO_SECRET_KEY"],
            object_name=CACHE_FILE,
        )
        log.info("Estado anterior borrado.")
    except Exception:
        log.info("No había estado previo en MinIO.")

    log.info("Iniciando worker: actualizando estado de lags en MinIO cada 90s...")
    while True:
        try:
            update_lag_state()
            log.info("Estado de lags actualizado correctamente.")
        except Exception as e:
            log.error("Error en el worker RT: %s", e)

        time.sleep(90)


if __name__ == "__main__":
    run_worker()
