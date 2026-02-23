import os
import base64
import re
import pandas as pd
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from bs4 import BeautifulSoup

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def parse_mta_body(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Get text and squash multiple spaces/newlines into one single space
    text = soup.get_text(separator=' ')
    clean_text = re.sub(r'\s+', ' ', text).strip()
    text_lower = clean_text.lower()
    
    # 1. NEW: Summary Category Column
    # We prioritize "Resumed" first because it changes the status of an old alert
    if any(word in text_lower for word in ["resumed", "regular service", "resolved"]):
        category = "Service Resumed"
    elif "preparing for" in text_lower and "storm" in text_lower:
        category = "Weather Prep"
    elif any(word in text_lower for word in ["delay", "held", "waiting", "slower"]):
        category = "Delay"
    elif any(word in text_lower for word in ["running local", "running express", "rerouted", "bypass"]):
        category = "Service Change"
    elif "planned work" in text_lower:
        category = "Planned Work"
    else:
        category = "Info/Other"

    # 2. Subway Lines
    line_pattern = r'\b([1-7]|A|B|C|D|E|F|G|J|L|M|N|Q|R|S|W|Z)\b'
    lines = sorted(list(set(re.findall(line_pattern, clean_text))))
    
    # 3. Specific Reason
    reason = "Unknown"
    if "door" in text_lower: reason = "Mechanical (Doors)"
    elif "signal" in text_lower: reason = "Signal Problems"
    elif "person on the tracks" in text_lower or "ems" in text_lower: reason = "Medical/Police"
    elif "track" in text_lower and "work" in text_lower: reason = "Maintenance"
    elif "winter storm" in text_lower: reason = "Weather"
    
    # 4. Location
    location = "Multiple/System-wide"
    loc_match = re.search(r'(?:at|from|to|near)\s+([A-Z][a-z0-9]+(?:\s[A-Z][a-z0-9]+)*)', clean_text)
    if loc_match:
        location = loc_match.group(1)

    return ", ".join(lines), reason, category, location, clean_text[:500]

def main():
    service = get_gmail_service()
    data_log = []
    page_token = None
    
    print("Starting extraction of all emails...")

    while True:
        # q='label:mta_alerts' ensures we only get filtered emails
        results = service.users().messages().list(
            userId='me', q='label:mta_alerts', pageToken=page_token
        ).execute()
        
        messages = results.get('messages', [])
        if not messages:
            break

        print(f"Processing batch of {len(messages)}...")

        for msg in messages:
            try:
                m = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
                timestamp = pd.to_datetime(int(m['internalDate']), unit='ms')

                def get_html_part(payload):
                    if payload.get('mimeType') == 'text/html':
                        return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
                    if 'parts' in payload:
                        for part in payload['parts']:
                            html = get_html_part(part)
                            if html: return html
                    return None

                html_body = get_html_part(m.get('payload', {}))
                
                if html_body:
                    lines, reason, category, location, clean_text = parse_mta_body(html_body)
                    data_log.append({
                        'timestamp': timestamp,
                        'category': category,      # NEW COLUMN
                        'lines': lines,
                        'reason': reason,
                        'location': location,
                        'text_snippet': clean_text
                    })
            except Exception:
                continue

        page_token = results.get('nextPageToken')
        if not page_token:
            break

    if data_log:
        df = pd.DataFrame(data_log)
        df = df.sort_values(by='timestamp', ascending=False)
        df.to_csv('mta_dataset.csv', index=False)
        print(f"Dataset created with {len(df)} rows.")

if __name__ == '__main__':
    main()