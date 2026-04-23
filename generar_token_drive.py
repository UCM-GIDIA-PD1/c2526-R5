"""
Genera token_drive.json mediante el flujo OAuth 2.0 del usuario.

Requisito previo:
    Descarga credentials.json desde Google Cloud Console:
    APIs & Services → Credentials → OAuth 2.0 Client IDs → Desktop App → Download JSON
    Colócalo en la raíz del proyecto con el nombre credentials.json

Uso:
    uv run python generar_token_drive.py

El token resultante se guarda en:
    src/ETL/alertas_oficiales_tiempo_real/token_drive.json
"""
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/drive"]

CREDENTIALS_PATH = Path("credentials.json")
TOKEN_PATH = Path("src/ETL/alertas_oficiales_tiempo_real/token_drive.json")


def main():
    if not CREDENTIALS_PATH.exists():
        raise FileNotFoundError(
            "No se encuentra credentials.json en la raíz del proyecto.\n"
            "Descárgalo desde Google Cloud Console:\n"
            "  APIs & Services → Credentials → OAuth 2.0 Client IDs → Desktop App → Download JSON\n"
            f"  y renómbralo a {CREDENTIALS_PATH}"
        )

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            print("Token refrescado.")
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
            print("Autorización completada.")

    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json())
    print(f"Token guardado en: {TOKEN_PATH}")


if __name__ == "__main__":
    main()
