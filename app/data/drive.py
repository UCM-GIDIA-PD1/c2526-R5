"""Google Drive client for downloading real-time sliding-window parquets.

Reads from the same folder that upload_realtime_window.py maintains:
  MTA_Realtime_Windows/
    ventana_YYYY-MM-DD_HH-MM.parquet
    ventana_YYYY-MM-DD_HH-MM.parquet
    ...  (up to N_WINDOWS files, oldest deleted automatically by the ETL)

Authentication mirrors upload_realtime_window.py:
  - In CI / production: set GDRIVE_TOKEN_JSON to the token.json contents.
  - Locally: run the OAuth flow once to generate token_drive.json.
"""
import io
import logging
import os
from pathlib import Path

import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/drive"]
_FOLDER_NAME = "MTA_Realtime_Windows"

# Default token path matches upload_realtime_window.py
_DEFAULT_TOKEN_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src" / "ETL" / "alertas_oficiales_tiempo_real" / "token_drive.json"
)


def _get_service(token_path: Path) -> object:
    # GitHub Actions / production: reconstruct token from env var
    token_json = os.getenv("GDRIVE_TOKEN_JSON")
    if token_json:
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(token_json)

    if not token_path.exists():
        raise FileNotFoundError(
            f"token_drive.json not found at {token_path}. "
            "Run the OAuth flow locally once to generate it, or set GDRIVE_TOKEN_JSON."
        )

    creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _get_folder_id(service, folder_name: str) -> str:
    """Resolve folder name → folder ID (same logic as upload_realtime_window.py)."""
    result = service.files().list(
        q=(
            f"name = '{folder_name}' "
            f"and mimeType = 'application/vnd.google-apps.folder' "
            f"and trashed = false"
        ),
        fields="files(id, name)",
    ).execute()
    files = result.get("files", [])
    if not files:
        raise ValueError(
            f"Drive folder '{folder_name}' not found. "
            "It is created automatically on the first ETL run."
        )
    return files[0]["id"]


def download_windows(
    n_windows: int = 8,
    token_path: Path | None = None,
    folder_name: str = _FOLDER_NAME,
) -> list[pd.DataFrame]:
    """Download the most recent n_windows parquet files from the Drive folder.

    Returns a list of DataFrames ordered oldest → newest.
    Files are named ventana_YYYY-MM-DD_HH-MM.parquet; sorted lexicographically
    which matches chronological order.
    """
    token_path = token_path or _DEFAULT_TOKEN_PATH
    service = _get_service(token_path)
    folder_id = _get_folder_id(service, folder_name)

    result = service.files().list(
        q=(
            f"'{folder_id}' in parents "
            f"and name contains 'ventana_' "
            f"and name contains '.parquet' "
            f"and trashed = false"
        ),
        orderBy="name desc",
        pageSize=n_windows,
        fields="files(id, name)",
    ).execute()

    files = result.get("files", [])
    if not files:
        raise ValueError(
            f"No parquet windows found in Drive folder '{folder_name}'. "
            "The ETL pipeline (upload_realtime_window.py) must run first."
        )

    files = files[:n_windows]  # newest-first from Drive

    windows: list[pd.DataFrame] = []
    for info in reversed(files):  # oldest first in output
        req = service.files().get_media(fileId=info["id"])
        buf = io.BytesIO()
        dl = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        buf.seek(0)
        df = pd.read_parquet(buf)
        windows.append(df)
        logger.debug("Drive window loaded: %s (%d rows)", info["name"], len(df))

    logger.info("Loaded %d windows from Drive folder '%s'", len(windows), folder_name)
    return windows
