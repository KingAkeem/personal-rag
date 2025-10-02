# gmail_ingest.py
import os
import base64
import re
import tempfile
from typing import List, Tuple, Dict, Optional, Callable

import gradio as gr
from bs4 import BeautifulSoup

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# ===== Gmail OAuth / Fetch Helpers =====
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n", strip=True)

def _clean_text(s: str) -> str:
    s = s.replace("\r", "")
    s = re.sub(r"\n{3,}", "\n\n", s)  # collapse big gaps
    return "\n".join(line.rstrip() for line in s.splitlines()).strip()

def _get_gmail_service(token_path: str, client_secret_path: str):
    creds = None
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, GMAIL_SCOPES)
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Save the refreshed token
            with open(token_path, "w") as f:
                f.write(creds.to_json())
        else:
            if not os.path.exists(client_secret_path):
                raise RuntimeError("Missing client_secrets.json for Gmail OAuth.")
            
            from google_auth_oauthlib.flow import InstalledAppFlow
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secret_path, GMAIL_SCOPES
            )
            creds = flow.run_local_server(port=0)  # or run_console() if you prefer
            
        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)

def _list_message_ids(service, max_results: int, query: Optional[str]) -> List[str]:
    ids, next_page = [], None
    while len(ids) < max_results:
        resp = service.users().messages().list(
            userId="me",
            q=query or "",
            maxResults=min(100, max_results - len(ids)),
            pageToken=next_page
        ).execute()
        ids.extend([m["id"] for m in resp.get("messages", [])])
        next_page = resp.get("nextPageToken")
        if not next_page:
            break
    return ids

def _extract_text_and_attachments(service, msg_dict) -> Tuple[str, List[Tuple[str, bytes]]]:
    attachments: List[Tuple[str, bytes]] = []

    def walk_parts(parts):
        chunks = []
        for p in parts:
            mime = p.get("mimeType", "")
            body = p.get("body", {})
            data = body.get("data")
            att_id = body.get("attachmentId")
            filename = p.get("filename") or ""
            if filename and att_id:
                att = service.users().messages().attachments().get(
                    userId="me", messageId=msg_dict["id"], id=att_id
                ).execute()
                attachments.append((filename, base64.urlsafe_b64decode(att["data"])))
            elif mime == "text/plain" and data:
                text = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                chunks.append(text)
            elif mime == "text/html" and data:
                html = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                chunks.append(_html_to_text(html))
            elif mime.startswith("multipart/") and "parts" in p:
                chunks.append(walk_parts(p["parts"]))
        return "\n".join([c for c in chunks if c])

    payload = msg_dict.get("payload", {})
    text = ""
    if payload.get("mimeType", "").startswith("multipart/") and payload.get("parts"):
        text = walk_parts(payload["parts"])
    else:
        body_data = payload.get("body", {}).get("data")
        if body_data:
            if payload.get("mimeType") == "text/html":
                html = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")
                text = _html_to_text(html)
            else:
                text = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")
    return _clean_text(text), attachments

def _fetch_messages(service, message_ids: List[str]) -> List[Dict]:
    out = []
    for mid in message_ids:
        m = service.users().messages().get(userId="me", id=mid, format="full").execute()
        headers = {h["name"].lower(): h["value"] for h in m.get("payload", {}).get("headers", [])}
        text, attachments = _extract_text_and_attachments(service, m)
        out.append({
            "id": m["id"],
            "threadId": m.get("threadId"),
            "from": headers.get("from"),
            "to": headers.get("to"),
            "subject": headers.get("subject", "") or "(no subject)",
            "date": headers.get("date"),
            "labels": m.get("labelIds", []),
            "snippet": m.get("snippet", ""),
            "text": text,
            "attachments": attachments,  # list[(filename, bytes)]
        })
    return out

# ===== Public UI Mount (no Blocks param needed) =====
def add_gmail_tab(
    store_email_fn: Callable[[Dict], str],
    default_query: str = "in:inbox newer_than:30d",
):
    with gr.Tab("Gmail"):
        gr.Markdown("### Connect Gmail and ingest emails into your knowledge base")

        # Per-session state
        session_token_dir = gr.State(value=None)
        session_auth_ok = gr.State(value=False)
        auth_url_state = gr.State(value="")  # Store auth URL

        client_secret_uploader = gr.File(
            label="Upload your Google OAuth client_secrets.json",
            file_types=[".json"],
            type="filepath"
        )

        with gr.Row():
            gmail_query = gr.Textbox(
                label="Gmail search query",
                value=default_query,
                placeholder="e.g., from:boss@company.com newer_than:90d"
            )
            gmail_max = gr.Slider(1, 2000, value=200, step=1, label="Max emails")

        with gr.Row():
            authorize_btn = gr.Button("1) Authorize Gmail")
            auth_code_input = gr.Textbox(
                label="Enter authorization code",
                placeholder="Paste the code from Google here"
            )
            submit_code_btn = gr.Button("Submit Code", variant="primary")
            fetch_btn = gr.Button("2) Fetch & Ingest", variant="primary")

        auth_url_display = gr.Markdown()
        gmail_log = gr.Textbox(label="Logs", lines=18)

        def init_session_dir():
            d = tempfile.mkdtemp(prefix="gmail_oauth_")
            return d, False, "", "Session directory created."
        
        def do_authorize(client_secret_path, token_dir):
            try:
                if not token_dir:
                    token_dir = tempfile.mkdtemp(prefix="gmail_oauth_")

                if not client_secret_path:
                    return token_dir, False, "", "❌ Please upload client_secrets.json first."

                # Get actual file path from Gradio file upload
                actual_client_secret_path = client_secret_path['name'] if isinstance(client_secret_path, dict) else client_secret_path

                from google_auth_oauthlib.flow import Flow
                
                flow = Flow.from_client_secrets_file(
                    client_secrets_file=actual_client_secret_path,
                    scopes=GMAIL_SCOPES
                )
                
                # Use the correct OOB redirect URI
                flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
                
                auth_url, state = flow.authorization_url(
                    prompt='consent',
                    access_type='offline'
                )
                
                # Store only the necessary data (not the flow object)
                auth_data = {
                    'client_secret_path': actual_client_secret_path,
                    'state': state,
                    'redirect_uri': flow.redirect_uri,
                    'client_config': flow.client_config  # Store client config to recreate flow
                }
                
                auth_data_path = os.path.join(token_dir, "auth_data.json")
                with open(auth_data_path, 'w') as f:
                    import json
                    json.dump(auth_data, f)
                
                # Also store client secrets separately for future use
                secrets_backup_path = os.path.join(token_dir, "client_secrets_backup.json")
                import shutil
                shutil.copy2(actual_client_secret_path, secrets_backup_path)

                auth_instructions = f"""
                ## Authorization Required

                1. Visit this URL: [{auth_url}]({auth_url})
                2. Grant permissions  
                3. Copy the authorization code
                4. Paste it in the box below and click "Submit Code"
                """
                
                return token_dir, False, auth_url, auth_instructions
            except Exception as e:
                return token_dir, False, "", f"❌ Authorization failed: {e}"

        def submit_auth_code(token_dir, auth_code):
            try:
                if not token_dir:
                    return token_dir, False, "❌ No session directory."
                
                token_path = os.path.join(token_dir, "token.json")
                auth_data_path = os.path.join(token_dir, "auth_data.json")
                secrets_backup_path = os.path.join(token_dir, "client_secrets_backup.json")
                
                if not os.path.exists(auth_data_path):
                    return token_dir, False, "❌ No authorization data found. Please click 'Authorize Gmail' first."
                
                # Load auth data
                import json
                with open(auth_data_path, 'r') as f:
                    auth_data = json.load(f)
                
                # Recreate the flow using the stored client config
                from google_auth_oauthlib.flow import Flow
                
                # Use the backup client secrets file
                flow = Flow.from_client_secrets_file(
                    client_secrets_file=secrets_backup_path,
                    scopes=GMAIL_SCOPES,
                    state=auth_data['state']
                )
                flow.redirect_uri = auth_data['redirect_uri']
                
                # Exchange the code for tokens
                flow.fetch_token(code=auth_code)
                creds = flow.credentials
                
                # Save the credentials
                with open(token_path, "w") as f:
                    f.write(creds.to_json())
                
                # Clean up temporary files
                os.remove(auth_data_path)
                # Keep client_secrets_backup.json for future token refreshes
                
                return token_dir, True, "✅ Authorization successful! You can now fetch emails."
            except Exception as e:
                return token_dir, False, f"❌ Failed to exchange code: {e}"
        
        def do_fetch_and_ingest(token_dir, query, max_results):
            if not token_dir:
                return "❌ No session token directory. Click 'Authorize Gmail' first."
            
            token_path = os.path.join(token_dir, "token.json")
            secrets_backup_path = os.path.join(token_dir, "client_secrets_backup.json")
            
            if not os.path.exists(token_path):
                return "❌ Missing token.json. Click 'Authorize Gmail' first."

            if not os.path.exists(secrets_backup_path):
                return "❌ Missing client_secrets in this session. Re-authorize."

            try:
                # Use the backup client_secrets file that we stored during authorization
                service = _get_gmail_service(token_path, secrets_backup_path)
                ids = _list_message_ids(service, int(max_results), query)
                if not ids:
                    return f"ℹ️ No messages found for query: '{query or ''}'."

                logs = [f"Found {len(ids)} messages. Fetching…"]
                total = 0

                for start in range(0, len(ids), 50):
                    chunk_ids = ids[start:start+50]
                    emails = _fetch_messages(service, chunk_ids)
                    logs.append(f"- Downloaded {len(emails)} (total {start+len(emails)}/{len(ids)})")
                    for e in emails:
                        try:
                            msg = store_email_fn(e)
                            logs.append("  • " + msg)
                        except Exception as ex:
                            logs.append(f"  • ERR {e.get('subject','(no subject)')[:80]} — {ex}")
                    total += len(emails)

                logs.append(f"✅ Done. Ingested {total} messages.")
                return "\n".join(logs[-1000:])
            except Exception as e:
                return f"❌ Error during fetch/ingest: {e}"

        # Initialize
        _dir, _ok, _url, _msg = init_session_dir()
        session_token_dir.value = _dir
        session_auth_ok.value = _ok
        auth_url_state.value = _url
        gmail_log.value = _msg

        authorize_btn.click(
            fn=do_authorize,
            inputs=[client_secret_uploader, session_token_dir],
            outputs=[session_token_dir, session_auth_ok, auth_url_state, gmail_log]
        )

        submit_code_btn.click(
            fn=submit_auth_code,
            inputs=[session_token_dir, auth_code_input],
            outputs=[session_token_dir, session_auth_ok, gmail_log]
        )

        # Update auth URL display
        def update_auth_display(auth_url):
            if auth_url:
                return f"**Authorization URL:** [Click here]({auth_url})"
            return ""
        
        auth_url_state.change(
            fn=update_auth_display,
            inputs=[auth_url_state],
            outputs=[auth_url_display]
        )

        fetch_btn.click(
            fn=do_fetch_and_ingest,
            inputs=[session_token_dir, gmail_query, gmail_max],
            outputs=[gmail_log]
        )