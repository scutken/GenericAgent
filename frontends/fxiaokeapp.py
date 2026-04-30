import asyncio, json, os, random, re, sys, threading, time
from typing import Dict, Optional
from urllib.parse import urlencode

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentmain import GeneraticAgent
from chatapp_common import AgentChatMixin, ensure_single_instance, public_access, redirect_log, require_runtime, split_text
from llmcore import mykeys

# ── Config ──────────────────────────────────────────────────────────
BASE_URL = str(mykeys.get("fxiaoke_base_url", "https://open.fxiaoke.com") or "https://open.fxiaoke.com").strip().rstrip("/")
APP_ID = str(mykeys.get("fxiaoke_app_id", "") or "").strip()
APP_SECRET = str(mykeys.get("fxiaoke_app_secret", "") or "").strip()
ALLOWED = {str(x).strip() for x in mykeys.get("fxiaoke_allowed_users", []) if str(x).strip()}
EVENT_VERSION = str(mykeys.get("fxiaoke_event_version", "1.3.0") or "1.3.0").strip()
CONNECT_TIMEOUT = int(mykeys.get("fxiaoke_connect_timeout", 15) or 15)
READ_TIMEOUT = int(mykeys.get("fxiaoke_read_timeout", 90) or 90)
BOT_MENTION_NAMES = {str(x).strip().lstrip("@").casefold() for x in mykeys.get("fxiaoke_bot_names", []) if str(x).strip()}

MENTION_RE = re.compile(r"@[^\s@]+")


def _strip_bot_mentions(content: str) -> str:
    """Remove Fxiaoke group-chat bot mentions so slash commands are parsed.

    Fxiaoke may deliver group messages as either "@Bot /cmd" or
    "/cmd @Bot".  The gateway event currently does not expose a stable bot
    display name, so configured names are stripped precisely; otherwise we
    only strip the edge mention when doing so reveals a slash command.
    """
    text = (content or "").strip()
    if not text:
        return ""

    def is_configured_bot(match: re.Match) -> bool:
        return match.group(0).lstrip("@").casefold() in BOT_MENTION_NAMES

    # Always remove configured bot mentions at message edges.
    changed = True
    while BOT_MENTION_NAMES and changed:
        changed = False
        m = MENTION_RE.match(text)
        if m and is_configured_bot(m):
            text = text[m.end():].strip()
            changed = True
        m = list(MENTION_RE.finditer(text))[-1:] if text else []
        if m and m[0].end() == len(text) and is_configured_bot(m[0]):
            text = text[:m[0].start()].strip()
            changed = True

    # Heuristic fallback for group bot commands: mention may be prepended or
    # appended to a slash command, e.g. "@阿乖-GA /help" or "/help @阿乖-GA".
    m = MENTION_RE.match(text)
    if m and text[m.end():].lstrip().startswith("/"):
        text = text[m.end():].strip()
    if text.startswith("/"):
        text = re.sub(r"\s+@[^\s@]+\s*$", "", text).strip()
    return text


agent = GeneraticAgent()
agent.verbose = False
USER_TASKS: Dict[str, dict] = {}


class FxiaokeApp(AgentChatMixin):
    """纷享销客企信 IM Gateway adapter.

    API doc:
      - POST /im-gateway/auth/token
      - GET  /im-gateway/bot/events?token=...&version=1.3.0 (SSE)
      - POST /im-gateway/qixin/message/send
    """

    label, source, split_limit, ping_interval = "Fxiaoke", "fxiaoke", 1800, 25

    def __init__(self):
        super().__init__(agent, USER_TASKS)
        self.access_token: Optional[str] = None
        self.token_expiry = 0.0
        self.token_lock = threading.Lock()
        self.last_event_id: Optional[str] = None
        self.retry_ms = 1000
        self.background_tasks = set()
        self.latest_chat_by_user: Dict[str, str] = {}

    # ── HTTP helpers ─────────────────────────────────────────────────
    def _auth_url(self):
        return f"{BASE_URL}/im-gateway/auth/token"

    def _events_url(self, token):
        return f"{BASE_URL}/im-gateway/bot/events?" + urlencode({"token": token, "version": EVENT_VERSION})

    def _send_url(self):
        return f"{BASE_URL}/im-gateway/qixin/message/send"

    def _get_access_token_sync(self, force=False):
        with self.token_lock:
            if not force and self.access_token and time.time() < self.token_expiry:
                return self.access_token

            resp = requests.post(
                self._auth_url(),
                json={"appId": APP_ID, "appSecret": APP_SECRET},
                timeout=(CONNECT_TIMEOUT, 30),
            )
            body = resp.text
            if resp.status_code != 200:
                raise RuntimeError(f"token HTTP {resp.status_code}: {body[:300]}")
            data = resp.json()
            if data.get("code") != 0:
                raise RuntimeError(f"token API code={data.get('code')}: {data.get('msg') or body[:300]}")

            payload = data.get("data") or {}
            token = payload.get("accessToken")
            if not token:
                raise RuntimeError(f"token response missing accessToken: {body[:300]}")
            self.access_token = token
            self.token_expiry = time.time() + int(payload.get("expiresIn") or 7200) - 60
            print(f"[Fxiaoke] token refreshed, expires in {int(self.token_expiry - time.time())}s")
            return token

    async def _get_access_token(self, force=False):
        return await asyncio.to_thread(self._get_access_token_sync, force)

    def _send_text_sync(self, chat_id, text, reply_message_id=None, force_token=False):
        token = self._get_access_token_sync(force=force_token)
        payload = {"chat_id": chat_id, "text": text}
        if reply_message_id is not None:
            payload["reply_message_id"] = reply_message_id
        resp = requests.post(
            self._send_url(),
            json=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=(CONNECT_TIMEOUT, 30),
        )
        body = resp.text
        if resp.status_code != 200:
            raise RuntimeError(f"send HTTP {resp.status_code}: {body[:300]}")
        data = resp.json()
        code = data.get("code")
        if code == 0:
            return data.get("data") or {}
        # Token expired/invalid: refresh once and retry.
        if code in (40100, 40101) and not force_token:
            return self._send_text_sync(chat_id, text, reply_message_id, force_token=True)
        raise RuntimeError(f"send API code={code}: {data.get('msg') or body[:300]}")

    def _sender_allowed(self, sender_id: str) -> bool:
        if public_access(ALLOWED):
            return True
        sender_id = str(sender_id or "").strip()
        candidates = {sender_id}
        if sender_id.startswith("E."):
            candidates.add(sender_id[2:])
        elif sender_id:
            candidates.add(f"E.{sender_id}")
        return bool(candidates & ALLOWED)

    async def send_text(self, chat_id, content, **ctx):
        reply_message_id = ctx.get("reply_message_id")
        for part in split_text(content, self.split_limit):
            try:
                await asyncio.to_thread(self._send_text_sync, chat_id, part, reply_message_id)
            except Exception as e:
                print(f"[Fxiaoke] send error: {e}")
                break

    # ── SSE parsing ──────────────────────────────────────────────────
    def _parse_sse_stream(self, resp):
        event, data_lines, event_id, retry = None, [], None, None
        for raw in resp.iter_lines(decode_unicode=False):
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            if raw is None:
                continue
            line = raw.strip("\r")
            if not line:
                if data_lines or event or event_id:
                    yield {"event": event or "message", "data": "\n".join(data_lines), "id": event_id, "retry": retry}
                event, data_lines, event_id, retry = None, [], None, None
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "event":
                event = value
            elif field == "data":
                data_lines.append(value)
            elif field == "id":
                event_id = value
            elif field == "retry":
                try:
                    retry = int(value)
                except Exception:
                    pass

    def _sse_once_sync(self, loop):
        token = self._get_access_token_sync()
        headers = {"Accept": "text/event-stream"}
        if self.last_event_id:
            headers["Last-Event-ID"] = self.last_event_id

        print(f"[Fxiaoke] connecting SSE version={EVENT_VERSION} last_event_id={self.last_event_id or '-'}")
        resp = requests.get(
            self._events_url(token),
            headers=headers,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        if resp.status_code == 401:
            self._get_access_token_sync(force=True)
            raise RuntimeError("SSE unauthorized, token refreshed")
        if resp.status_code != 200:
            raise RuntimeError(f"SSE HTTP {resp.status_code}: {resp.text[:300]}")

        for evt in self._parse_sse_stream(resp):
            if evt.get("retry"):
                self.retry_ms = int(evt["retry"])
            if evt.get("id"):
                self.last_event_id = str(evt["id"])
            fut = asyncio.run_coroutine_threadsafe(self.on_sse_event(evt), loop)
            fut.add_done_callback(lambda f: f.exception() and print(f"[Fxiaoke] event task error: {f.exception()}"))

    async def on_sse_event(self, evt):
        name = evt.get("event") or "message"
        raw = evt.get("data") or ""
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except Exception:
            print(f"[Fxiaoke] non-json {name}: {raw[:300]}")
            return

        if name == "connected" or payload.get("type") == "connected":
            info = payload.get("data") or {}
            if info.get("retry"):
                self.retry_ms = int(info["retry"])
            print(f"[Fxiaoke] connected bot={info.get('bot_full_id')} protocol={info.get('protocol_version')} retry={self.retry_ms}ms")
            return

        if name == "reset" or payload.get("type") == "reset":
            print(f"[Fxiaoke] reset: {payload}")
            self.last_event_id = None
            return

        if name != "message" and payload.get("type") != "message":
            print(f"[Fxiaoke] ignored event {name}: {payload}")
            return

        data = payload.get("data") or {}
        await self.on_message(data)

    async def on_message(self, data):
        try:
            chat_id = str(data.get("chat_id") or "").strip()
            if not chat_id:
                print(f"[Fxiaoke] message missing chat_id: {data}")
                return

            sender = data.get("from") or {}
            sender_id = str(sender.get("id") or data.get("sender_id") or "unknown")
            sender_name = str(sender.get("name") or sender_id)
            message_id = data.get("message_id")
            msg_obj = data.get("message") if isinstance(data.get("message"), dict) else {}
            content = str(data.get("text") or msg_obj.get("content") or "").strip()
            content = _strip_bot_mentions(content)

            if not content:
                print(f"[Fxiaoke] empty/non-text message: id={message_id} type={data.get('message_type') or msg_obj.get('type')}")
                return
            if not self._sender_allowed(sender_id):
                print(f"[Fxiaoke] unauthorized user: {sender_id}; allow list={sorted(ALLOWED)}")
                return

            self.latest_chat_by_user[sender_id] = chat_id
            print(f"[Fxiaoke] message from {sender_name} ({sender_id}) chat={chat_id}: {content}")
            ctx = {"reply_message_id": message_id}
            if content.startswith("/"):
                return await self.handle_command(chat_id, content, **ctx)

            task = asyncio.create_task(self.run_agent(chat_id, content, **ctx))
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        except Exception:
            import traceback
            print("[Fxiaoke] handle_message error")
            traceback.print_exc()

    async def start(self):
        loop = asyncio.get_running_loop()
        print(f"[Fxiaoke] bot starting, gateway={BASE_URL}")
        backoff = 1.0
        while True:
            try:
                await asyncio.to_thread(self._sse_once_sync, loop)
                backoff = 1.0
            except Exception as e:
                print(f"[Fxiaoke] SSE error: {e}")
                wait = max(self.retry_ms / 1000.0, backoff) + random.uniform(0, 0.3 * backoff)
                print(f"[Fxiaoke] reconnect in {wait:.1f}s...")
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, 60.0)


if __name__ == "__main__":
    _LOCK_SOCK = ensure_single_instance(19532, "Fxiaoke")
    require_runtime(agent, "Fxiaoke", fxiaoke_app_id=APP_ID, fxiaoke_app_secret=APP_SECRET)
    redirect_log(__file__, "fxiaokeapp.log", "Fxiaoke", ALLOWED)
    threading.Thread(target=agent.run, daemon=True).start()
    asyncio.run(FxiaokeApp().start())