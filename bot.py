#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import time
from typing import Set, Optional, Dict, List

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    filters,
)

from ollama_client import OllamaClient, OllamaError
from models import (
    init_db,
    Chat,
    ChatSession,
    Message,
    ensure_chat,
    get_active_session,
    start_new_session,
    list_history_messages,
    clear_chat_history,
)

# ========================= НАСТРОЙКИ / ОКРУЖЕНИЕ =========================

load_dotenv()  # читает .env из текущего каталога

TELEGRAM_API = os.getenv("TELEGRAM_API", "").strip()
if not TELEGRAM_API:
    raise RuntimeError("В .env отсутствует TELEGRAM_API")

CHAT_IDS_RAW = os.getenv("CHAT_IDS", "").strip()
if not CHAT_IDS_RAW:
    raise RuntimeError("В .env отсутствует CHAT_IDS (список chat_id через запятую)")

def parse_chat_ids(value: str) -> Set[int]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    ids: Set[int] = set()
    for it in items:
        try:
            ids.add(int(it))
        except ValueError:
            raise RuntimeError(f"Некорректный chat_id в CHAT_IDS: '{it}'")
    return ids

ALLOWED_CHAT_IDS: Set[int] = parse_chat_ids(CHAT_IDS_RAW)

# Ollama/модель по умолчанию и опции
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
DEFAULT_MODEL = os.getenv("MODEL_NAME", "dolphin-mixtral:8x7b").strip()

import json
def _parse_options(raw: str):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Не удалось распарсить OLLAMA_OPTIONS: {e}")

OLLAMA_OPTIONS = _parse_options(os.getenv("OLLAMA_OPTIONS", "").strip())

# Системная подсказка (можно оставить пустой)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "").strip() or None

# Ограничение истории, отправляемой в модель (по числу сообщений)
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "24"))

# Путь к БД
DB_PATH = os.getenv("DB_PATH", "bot.sqlite3")

# Алиасы моделей (команды и через /model ALIAS)
MODEL_ALIASES: Dict[str, str] = {
    "dolph": "dolphin-mixtral:8x7b",
    "oss20": "gpt-oss:20b",
    "qcod7": "qwen2.5-coder:7b",
    "qcod15": "qwen2.5-coder:1.5b",
    "qwen7": "qwen2.5:7b-instruct",
    "qwen3": "qwen2.5:3b-instruct",
    "qwen14": "qwen2.5:14b-instruct",
    "ll1": "llama3.2:1b",
    "mistral": "mistral:latest",
    # синоним
    "g20": "gpt-oss:20b",
}
REVERSE_ALIASES: Dict[str, str] = {v: k for k, v in MODEL_ALIASES.items()}

# Список команд-алиасов для регистрации
MODEL_ALIAS_COMMANDS = list(MODEL_ALIASES.keys())

# ========================= УТИЛИТЫ =========================

def chunk_text(text: str, limit: int = 4096) -> List[str]:
    """
    Делит длинный текст на куски <= limit символов, стараясь резать по абзацам/строкам.
    """
    if len(text) <= limit:
        return [text]

    parts: List[str] = []
    current: List[str] = []
    curr_len = 0

    for line in re.split(r"(\n+)", text):
        line_len = len(line)
        if curr_len + line_len > limit and current:
            parts.append("".join(current))
            current = [line]
            curr_len = line_len
        else:
            current.append(line)
            curr_len += line_len
    if current:
        tail = "".join(current)
        while len(tail) > limit:
            parts.append(tail[:limit])
            tail = tail[limit:]
        if tail:
            parts.append(tail)
    return parts

def _username(update: Update) -> str:
    u = update.effective_user
    if not u:
        return ""
    return u.username or (u.full_name or "").strip()

async def _ensure_ollama(app: Application) -> OllamaClient:
    client: OllamaClient | None = app.bot_data.get("ollama")  # type: ignore[assignment]
    if client is None:
        client = OllamaClient(base_url=OLLAMA_HOST, model=DEFAULT_MODEL, options=OLLAMA_OPTIONS)
        app.bot_data["ollama"] = client
    return client

def _model_alias(model_name: str) -> str:
    """Возвращает алиас модели или компактное имя по умолчанию."""
    alias = REVERSE_ALIASES.get(model_name)
    if alias:
        return alias
    # fallback: забрать базовую часть до ':' и цифры, слегка укоротить
    base = model_name.split(":")[0]
    base = base.replace("qwen2.5", "qwen").replace("dolphin-mixtral", "dolph")
    return base

def _format_duration_tag(elapsed_seconds: float, model_name: str) -> str:
    m = int(elapsed_seconds // 60)
    s = int(round(elapsed_seconds - m * 60))
    alias = _model_alias(model_name)
    return f"\n\n[{m}:{s:02d} {alias}]"

# ========================= ХЕНДЛЕРЫ КОМАНД =========================

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat:
        return
    chat_id = update.effective_chat.id
    if chat_id not in ALLOWED_CHAT_IDS:
        await update.effective_message.reply_text("⛔️ Доступ запрещён.")
        return

    chat = ensure_chat(chat_id=chat_id, username=_username(update), default_model=DEFAULT_MODEL)
    get_active_session(chat)

    await update.effective_message.reply_text(
        "Бот готов. История для этого чата сохраняется в БД.\n"
        f"Текущая модель: <code>{chat.current_model}</code>\n"
        "Команды: /new — новый диалог, /clear — очистить историю, "
        "/model <имя|алиас> — смена модели, /alias — список алиасов, "
        f"алиасы: {', '.join('/'+a for a in MODEL_ALIAS_COMMANDS)}",
        parse_mode=ParseMode.HTML,
    )

async def new_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat:
        return
    chat_id = update.effective_chat.id
    if chat_id not in ALLOWED_CHAT_IDS:
        await update.effective_message.reply_text("⛔️ Доступ запрещён.")
        return

    chat = ensure_chat(chat_id=chat_id, username=_username(update), default_model=DEFAULT_MODEL)
    start_new_session(chat)
    await update.effective_message.reply_text("✅ Открыт <b>новый чат</b>. История текущей сессии обнулена.", parse_mode=ParseMode.HTML)

async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /clear — удалить все сессии и сообщения для данного chat_id.
    После очистки создаётся новая пустая сессия.
    """
    if not update.effective_chat:
        return
    chat_id = update.effective_chat.id
    if chat_id not in ALLOWED_CHAT_IDS:
        await update.effective_message.reply_text("⛔️ Доступ запрещён.")
        return

    chat = ensure_chat(chat_id=chat_id, username=_username(update), default_model=DEFAULT_MODEL)
    msg_count, sess_count = clear_chat_history(chat)
    # создадим новую пустую сессию сразу
    start_new_session(chat)
    await update.effective_message.reply_text(
        f"🧹 История очищена: сообщений={msg_count}, сессий={sess_count}. Создан новый пустой чат.",
        parse_mode=ParseMode.HTML,
    )

async def alias_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /alias — показать список алиасов моделей.
    """
    if not update.effective_chat:
        return
    chat_id = update.effective_chat.id
    if chat_id not in ALLOWED_CHAT_IDS:
        await update.effective_message.reply_text("⛔️ Доступ запрещён.")
        return

    lines = []
    for k in sorted(MODEL_ALIASES.keys()):
        v = MODEL_ALIASES[k]
        lines.append(f"/{k} → <code>{v}</code>")
    txt = "Доступные алиасы моделей:\n" + "\n".join(lines)
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.HTML)

async def model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /model — показать/сменить модель: `/model qcod7` или `/model qwen2.5:7b-instruct`
    """
    if not update.effective_chat:
        return
    chat_id = update.effective_chat.id
    if chat_id not in ALLOWED_CHAT_IDS:
        await update.effective_message.reply_text("⛔️ Доступ запрещён.")
        return

    client = await _ensure_ollama(context.application)
    chat = ensure_chat(chat_id=chat_id, username=_username(update), default_model=DEFAULT_MODEL)

    args = update.effective_message.text.split(maxsplit=1)
    if len(args) == 1:
        await update.effective_message.reply_text(
            f"Текущая модель: <code>{chat.current_model}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    target = args[1].strip()
    model = MODEL_ALIASES.get(target, target)  # алиас или полное имя

    available = await client.ensure_model_available(model)
    if not available:
        await update.effective_message.reply_text(
            f"Модель <code>{model}</code> не найдена локально (по данным /api/tags). "
            f"Скачайте: <code>ollama pull {model}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    chat.current_model = model
    chat.save()
    client.model = model

    await update.effective_message.reply_text(
        f"OK. Для этого чата активна модель: <code>{model}</code>",
        parse_mode=ParseMode.HTML,
    )

async def alias_model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Хендлер для алиасов (например, /qcod7, /oss20, /mistral, ...).
    """
    if not update.effective_chat:
        return
    chat_id = update.effective_chat.id
    if chat_id not in ALLOWED_CHAT_IDS:
        await update.effective_message.reply_text("⛔️ Доступ запрещён.")
        return

    cmd = update.effective_message.text.split()[0].lstrip("/").split("@")[0]
    model = MODEL_ALIASES.get(cmd)
    if not model:
        await update.effective_message.reply_text("Неизвестный алиас модели.")
        return

    client = await _ensure_ollama(context.application)
    chat = ensure_chat(chat_id=chat_id, username=_username(update), default_model=DEFAULT_MODEL)

    available = await client.ensure_model_available(model)
    if not available:
        await update.effective_message.reply_text(
            f"Модель <code>{model}</code> не найдена локально. "
            f"Скачайте: <code>ollama pull {model}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    chat.current_model = model
    chat.save()
    client.model = model

    await update.effective_message.reply_text(
        f"OK. Для этого чата активна модель: <code>{model}</code>",
        parse_mode=ParseMode.HTML,
    )

# ========================= ОСНОВНОЙ ХЕНДЛЕР ТЕКСТА =========================

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message or update.effective_message.text is None:
        return

    chat_id = update.effective_chat.id
    text = update.effective_message.text

    if chat_id not in ALLOWED_CHAT_IDS:
        return

    client = await _ensure_ollama(context.application)

    # ORM: чат и активная сессия
    chat = ensure_chat(chat_id=chat_id, username=_username(update), default_model=DEFAULT_MODEL)
    sess = get_active_session(chat)

    # сохраняем входящее сообщение в БД
    Message.create(session_id=sess.id, role="user", content=text)

    # визуальная индикация
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    except Exception:
        pass

    # соберём историю для модели
    history = list_history_messages(sess.id, limit=HISTORY_LIMIT)
    messages_payload = [{"role": m.role, "content": m.content} for m in history]

    # запрос в Ollama с измерением времени
    t0 = time.perf_counter()
    try:
        reply = await client.chat(
            messages=messages_payload,
            model=chat.current_model,
            system=SYSTEM_PROMPT,
        )
    except OllamaError as e:
        err_msg = f"Ошибка Ollama: {e}"
        Message.create(session_id=sess.id, role="assistant", content=err_msg)
        await update.effective_message.reply_text(err_msg)
        return
    except Exception as e:
        err_msg = f"Непредвиденная ошибка: {e}"
        Message.create(session_id=sess.id, role="assistant", content=err_msg)
        await update.effective_message.reply_text(err_msg)
        return
    elapsed = time.perf_counter() - t0

    # сохраняем ответ ассистента
    Message.create(session_id=sess.id, role="assistant", content=reply)

    # формируем и добавляем метку [m:ss alias]
    tag = _format_duration_tag(elapsed, chat.current_model)

    parts = chunk_text(reply, limit=4096)
    if parts:
        # если последний кусок + тег помещаются, добавим метку в последний кусок
        if len(parts[-1]) + len(tag) <= 4096:
            parts[-1] = parts[-1] + tag
            for part in parts:
                await update.effective_message.reply_text(part)
        else:
            # иначе отправим как отдельное короткое сообщение
            for part in parts:
                await update.effective_message.reply_text(part)
            await update.effective_message.reply_text(tag.strip())
    else:
        await update.effective_message.reply_text(tag.strip())

# ========================= ХУКИ И СБОРКА ПРИЛОЖЕНИЯ =========================

async def _post_init(app: Application) -> None:
    app.bot_data["ollama"] = OllamaClient(base_url=OLLAMA_HOST, model=DEFAULT_MODEL, options=OLLAMA_OPTIONS)

async def _post_shutdown(app: Application) -> None:
    try:
        client: OllamaClient = app.bot_data.get("ollama")  # type: ignore[assignment]
        if client:
            await client.close()
    except Exception:
        pass

def build_app() -> Application:
    app = (
        Application.builder()
        .token(TELEGRAM_API)
        .concurrent_updates(True)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )

    # Команды
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("new", new_cmd))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(CommandHandler("alias", alias_cmd))
    app.add_handler(CommandHandler("model", model_cmd))
    # Алиасы моделей — одним хендлером на список команд
    app.add_handler(CommandHandler(MODEL_ALIAS_COMMANDS, alias_model_cmd))
    # Тексты
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))
    return app

# ========================= ТОЧКА ВХОДА =========================

if __name__ == "__main__":
    # Инициализация БД (после загрузки .env)
    init_db(DB_PATH)

    print(f"CHAT_IDS: {ALLOWED_CHAT_IDS}")
    print(f"DB_PATH: {DB_PATH}")
    application = build_app()
    application.run_polling(allowed_updates=Update.ALL_TYPES)
