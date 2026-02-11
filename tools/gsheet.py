import io
import os
import re
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")


def _get_credentials() -> Credentials:
    """取得或刷新 Google OAuth2 憑證。"""
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"找不到 credentials.json，請將 Google OAuth2 憑證檔案放置於 {CREDENTIALS_PATH}"
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_PATH, "w") as token_file:
            token_file.write(creds.to_json())

    return creds


def extract_spreadsheet_id(url: str) -> str:
    """從 Google Sheet URL 擷取 spreadsheet ID。"""
    patterns = [
        r"/spreadsheets/d/([a-zA-Z0-9-_]+)",
        r"/file/d/([a-zA-Z0-9-_]+)",
        r"^([a-zA-Z0-9-_]+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"無法從 URL 擷取 spreadsheet ID: {url}")


def _read_via_sheets_api(file_id: str, creds: Credentials) -> list[dict]:
    """使用 Sheets API 讀取原生 Google Sheets。"""
    service = build("sheets", "v4", credentials=creds)

    spreadsheet_meta = service.spreadsheets().get(spreadsheetId=file_id).execute()
    first_sheet_title = spreadsheet_meta["sheets"][0]["properties"]["title"]

    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=file_id, range=first_sheet_title)
        .execute()
    )
    values = result.get("values", [])

    if not values:
        raise ValueError("Google Sheet 中沒有資料")

    headers = values[0]
    rows = []
    for row in values[1:]:
        padded_row = row + [""] * (len(headers) - len(row))
        rows.append(dict(zip(headers, padded_row)))

    return rows


def _read_via_drive_api(file_id: str, creds: Credentials, mime_type: str) -> list[dict]:
    """使用 Drive API 下載 Excel/CSV 檔案，再以 pandas 讀取。"""
    service = build("drive", "v3", credentials=creds)

    if mime_type == "application/vnd.google-apps.spreadsheet":
        # 原生 Google Sheets → 匯出為 CSV
        request = service.files().export_media(fileId=file_id, mimeType="text/csv")
    else:
        # Excel / CSV 等上傳檔案 → 直接下載
        request = service.files().get_media(fileId=file_id)

    content = request.execute()
    buf = io.BytesIO(content)

    if mime_type in (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ):
        df = pd.read_excel(buf)
    else:
        # 嘗試 CSV
        buf.seek(0)
        df = pd.read_csv(buf)

    if df.empty:
        raise ValueError("檔案中沒有資料")

    return df.astype(str).to_dict(orient="records")


def read_sheet(url: str) -> list[dict]:
    """
    讀取 Google Sheet 或 Drive 上的 Excel/CSV 資料，回傳 list[dict]。

    Args:
        url: Google Sheet URL、Drive 檔案 URL 或 file ID

    Returns:
        list[dict]: 每列一個 dict，key 為標題列的欄位名稱
    """
    file_id = extract_spreadsheet_id(url)
    creds = _get_credentials()

    # 先用 Drive API 取得檔案 MIME type
    drive_service = build("drive", "v3", credentials=creds)
    file_meta = drive_service.files().get(fileId=file_id, fields="mimeType,name").execute()
    mime_type = file_meta.get("mimeType", "")

    if mime_type == "application/vnd.google-apps.spreadsheet":
        # 原生 Google Sheets → 優先用 Sheets API
        try:
            return _read_via_sheets_api(file_id, creds)
        except Exception:
            return _read_via_drive_api(file_id, creds, mime_type)
    else:
        # Excel / CSV 等上傳檔案 → 用 Drive API 下載
        return _read_via_drive_api(file_id, creds, mime_type)
