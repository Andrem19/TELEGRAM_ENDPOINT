# TELEGRAM_ENDPOINT

Private Telegram endpoint for chatting with local Ollama models. The bot keeps a SQLite conversation history per allowed chat, supports model aliases, and exposes simple commands for starting new sessions, clearing history, and switching models.

## Features

- Telegram bot built with `python-telegram-bot`.
- Allowlist access control through `CHAT_IDS`.
- Local Ollama integration through an async `httpx` client.
- Per-chat SQLite history with sessions and messages.
- Lightweight in-repo ORM with automatic SQLite schema setup.
- Model aliases for quickly switching between local models.
- Long response chunking for Telegram message limits.

## Commands

- `/start` - initialize the chat and show current model.
- `/new` - start a new session.
- `/clear` - delete all saved history for the current chat.
- `/model` - show current model.
- `/model <name|alias>` - switch model.
- `/alias` - list configured aliases.
- `/<alias>` - shortcut to switch to a model alias.

## Configuration

Create a local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Required values:

```env
TELEGRAM_API=replace-with-telegram-bot-token
CHAT_IDS=123456789
```

Optional Ollama/runtime values:

```env
OLLAMA_HOST=http://127.0.0.1:11434
MODEL_NAME=dolphin-mixtral:8x7b
OLLAMA_OPTIONS={"temperature":0.2}
SYSTEM_PROMPT=
HISTORY_LIMIT=24
DB_PATH=bot.sqlite3
```

Telegram tokens and SQLite runtime databases must stay local. Do not commit `.env`, `bot.sqlite3`, logs, or generated SQLite WAL/SHM files.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the bot:

```bash
python bot.py
```

Ollama must be running and the selected model must be available locally.
