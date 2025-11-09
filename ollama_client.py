#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from typing import Optional, Dict, Any, List

import httpx


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """
    Асинхронный клиент к локальному Ollama API.
    Поддерживает /api/chat (нестримингово) с историей сообщений.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "dolphin-mixtral:8x7b",
        request_timeout: float = 300.0,
        connect_timeout: float = 5.0,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.options = options or {"temperature": 0.2}
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(request_timeout, connect=connect_timeout),
        )

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------- внутренние -------------------------

    async def _get(self, path: str) -> httpx.Response:
        resp = await self._client.get(path)
        resp.raise_for_status()
        return resp

    async def _post(self, path: str, payload: Dict[str, Any]) -> httpx.Response:
        resp = await self._client.post(path, json=payload)
        resp.raise_for_status()
        return resp

    # ------------------------- публичные --------------------------

    async def ensure_model_available(self, model: Optional[str] = None) -> bool:
        """
        Проверяет наличие модели по /api/tags. Возвращает True/False.
        """
        mdl = model or self.model
        try:
            resp = await self._get("/api/tags")
            data = resp.json()
            for it in data.get("models", []):
                if it.get("name") == mdl:
                    return True
        except Exception:
            return False
        return False

    async def chat_once(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Одноходовый запрос без истории.
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages=messages, model=model, system=system, extra_options=extra_options)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        system: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Запрос с историей сообщений (формат OpenAI-like).
        Пример messages: [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
        """
        effective_model = model or self.model

        if system:
            messages = [{"role": "system", "content": system}] + messages

        payload: Dict[str, Any] = {
            "model": effective_model,
            "stream": False,
            "messages": messages,
        }

        options = dict(self.options or {})
        if extra_options:
            options.update(extra_options)
        if options:
            payload["options"] = options

        try:
            resp = await self._post("/api/chat", payload)
        except httpx.HTTPStatusError as e:
            text = getattr(e.response, "text", "")
            raise OllamaError(f"Ollama returned HTTP {e.response.status_code}: {text}") from e
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            raise OllamaError(f"Не удалось подключиться к Ollama: {e}") from e

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise OllamaError(f"Некорректный JSON от Ollama: {e}") from e

        msg = data.get("message") or {}
        content = msg.get("content") or data.get("response")
        if not content:
            raise OllamaError(f"Пустой ответ от модели '{effective_model}': {data}")

        return content.strip()
