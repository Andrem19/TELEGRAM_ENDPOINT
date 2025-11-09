#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime as dt
from dataclasses import field
from typing import List, Tuple

from simple_orm import BaseModel, initialize as orm_initialize


# --------------------- модели ORM ---------------------

class Chat(BaseModel):
    # реальные типы (не строки, не Optional)
    chat_id: int
    username: str = ""
    current_model: str = "dolphin-mixtral:8x7b"
    current_session_id: int = None  # nullable
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.utcnow())
    updated_at: dt.datetime = field(default_factory=lambda: dt.datetime.utcnow())


class ChatSession(BaseModel):
    chat_id: int
    started_at: dt.datetime = field(default_factory=lambda: dt.datetime.utcnow())
    is_active: bool = True
    title: str = ""  # опционально


class Message(BaseModel):
    session_id: int
    role: str            # 'user' | 'assistant' | 'system'
    content: str
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.utcnow())


# --------------------- инициализация БД ---------------------

def init_db(db_path: str) -> None:
    """Инициализирует SQLite и применяет миграции для всех моделей."""
    orm_initialize(db_path)


# --------------------- вспомогательные утилиты ---------------------

def ensure_chat(chat_id: int, username: str, default_model: str) -> Chat:
    chat = Chat.get(chat_id=chat_id)
    if chat is None:
        chat = Chat.create(chat_id=chat_id, username=username or "", current_model=default_model)
    else:
        chat.username = username or chat.username
        chat.updated_at = dt.datetime.utcnow()
        chat.save()
    return chat


def get_active_session(chat: Chat) -> ChatSession:
    if chat.current_session_id is not None:
        sess = ChatSession.get(id=chat.current_session_id)
        if sess and sess.is_active:
            return sess

    # нет активной — создаём
    sess = ChatSession.create(chat_id=chat.chat_id, is_active=True)
    chat.current_session_id = sess.id
    chat.updated_at = dt.datetime.utcnow()
    chat.save()
    return sess


def start_new_session(chat: Chat) -> ChatSession:
    # помечаем старую как неактивную
    if chat.current_session_id is not None:
        old = ChatSession.get(id=chat.current_session_id)
        if old and old.is_active:
            old.is_active = False
            old.save()

    # создаём новую
    new_sess = ChatSession.create(chat_id=chat.chat_id, is_active=True)
    chat.current_session_id = new_sess.id
    chat.updated_at = dt.datetime.utcnow()
    chat.save()
    return new_sess


def list_history_messages(session_id: int, limit: int = None) -> List[Message]:
    msgs = sorted(Message.filter(session_id=session_id), key=lambda m: m.id or 0)
    if limit is not None and len(msgs) > limit:
        return msgs[-limit:]
    return msgs


def clear_chat_history(chat: Chat) -> Tuple[int, int]:
    """
    Полностью очищает историю бесед для данного chat: все сообщения и все сессии.
    Возвращает (count_messages, count_sessions).
    """
    # посчитаем
    q_msgs = ChatSession.raw_query(  # используем любой .raw_query — статический метод базового класса
        "SELECT COUNT(*) AS c FROM message WHERE session_id IN (SELECT id FROM chatsession WHERE chat_id=?)",
        (chat.chat_id,),
    )
    q_sess = ChatSession.raw_query("SELECT COUNT(*) AS c FROM chatsession WHERE chat_id=?", (chat.chat_id,))
    msg_count = int(q_msgs[0]["c"]) if q_msgs else 0
    sess_count = int(q_sess[0]["c"]) if q_sess else 0

    # удаляем в правильном порядке
    Message.raw_execute(
        "DELETE FROM message WHERE session_id IN (SELECT id FROM chatsession WHERE chat_id=?)",
        (chat.chat_id,),
    )
    ChatSession.raw_execute("DELETE FROM chatsession WHERE chat_id=?", (chat.chat_id,))

    # сброс текущей сессии
    chat.current_session_id = None
    chat.updated_at = dt.datetime.utcnow()
    chat.save()

    return (msg_count, sess_count)
