import asyncio, json, os, queue as Q, random, re, sys, threading, time
from typing import Dict, Optional
from urllib.parse import urlencode

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentmain import GeneraticAgent
from chatapp_common import AgentChatMixin, FILE_HINT, build_done_text, clean_reply, ensure_single_instance, public_access, redirect_log, require_runtime, split_text
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

# ── Progress/final text helpers (fxiaoke-local) ────────────────────
_TURN_SPLIT_RE = re.compile(r'(\**LLM Running \(Turn \d+\) \.\.\.\**)')
_TURN_MARK_RE = re.compile(r'^\s*\**LLM Running \(Turn \d+\) \.\.\.\**\s*$', re.M)
_TOOL_LINE_RE = re.compile(r'^(\s*🛠️\s*[A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$', re.M)
_TOOL_LINE_FINAL_RE = re.compile(r'^\s*🛠️\s*[A-Za-z_][A-Za-z0-9_]*\(.*\)\s*$', re.M)


def _compress_tool_line(m: re.Match) -> str:
    head, args = m.group(1), m.group(2)
    args = re.sub(r'\s+', ' ', args).strip()
    if len(args) > 120:
        args = args[:117] + '...'
    return f"{head}({args})"


def _clean_final(t: str) -> str:
    t = _TURN_MARK_RE.sub('', t or '')
    t = _TOOL_LINE_FINAL_RE.sub('', t)
    return clean_reply(t)


def _clean_progress(t: str) -> str:
    t = clean_reply(t or '')
    return _TOOL_LINE_RE.sub(_compress_tool_line, t).strip()


def _turn_parts(t: str):
    parts = _TURN_SPLIT_RE.split(t or '')
    if len(parts) < 4:
        return [], (t or '')
    turns = [parts[i] + (parts[i + 1] if i + 1 < len(parts) else '') for i in range(1, len(parts), 2)]
    head = [parts[0]] if parts[0].strip() else []
    return head + turns[:-1], turns[-1]


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

    # ── Stream-style run_agent (overrides mixin default) ──────────────
    async def run_agent(self, chat_id, text, **ctx):
        state = {"running": True}
        self.user_tasks[chat_id] = state
        sent_turns = 0
        progress_cnt = 0
        last_send = 0.0
        MAX_PROGRESS = 9
        AGG_INTERVAL = 60.0  # 第 10 条起，至少隔这么久聚合推一次
        try:
            await self.send_text(chat_id, "思考中...", **ctx)
            dq = self.agent.put_task(f"{FILE_HINT}\n\n{text}", source=self.source)

            async def _push_progress(chunk: str) -> bool:
                nonlocal progress_cnt, last_send
                s = (chunk or '').strip()
                if not s:
                    return False
                now = time.time()
                if progress_cnt < MAX_PROGRESS:
                    # 前 9 条：6*n 秒节流
                    if progress_cnt and now - last_send < 6 * progress_cnt:
                        return False
                    await self.send_text(chat_id, s[:self.split_limit], **ctx)
                    progress_cnt += 1
                    last_send = now
                    return True
                # 超过上限：聚合心跳，每 AGG_INTERVAL 秒发一条合并简报
                if now - last_send < AGG_INTERVAL:
                    return False
                await self.send_text(chat_id, s[:self.split_limit], **ctx)
                last_send = now
                return True

            result = ''
            while state["running"]:
                try:
                    item = await asyncio.to_thread(dq.get, True, 3)
                except Q.Empty:
                    continue
                if 'done' in item:
                    result = item.get('done', '')
                    break
                raw = item.get('next', '')
                done_turns, _partial = _turn_parts(raw)
                if len(done_turns) > sent_turns:
                    merged = _clean_progress('\n\n'.join(done_turns[sent_turns:]))
                    if await _push_progress(merged):
                        sent_turns = len(done_turns)

            if not state["running"]:
                return await self.send_text(chat_id, "⏹️ 已停止", **ctx)

            # Flush any remaining completed turns as a compact progress message.
            done_turns, _partial = _turn_parts(result)
            if len(done_turns) > sent_turns:
                merged = _clean_progress('\n\n'.join(done_turns[sent_turns:]))
                if merged:
                    await _push_progress(merged)

            # Final body: strip all process markers, then let build_done_text
            # append [FILE:...] attachments.
            final_with_files = build_done_text(_clean_final(result))
            await self.send_text(chat_id, final_with_files or "...", **ctx)
        except Exception as e:
            import traceback
            print(f"[{self.label}] run_agent error: {e}")
            traceback.print_exc()
            await self.send_text(chat_id, f"❌ 错误: {e}", **ctx)
        finally:
            self.user_tasks.pop(chat_id, None)

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