from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import requests
from dotenv import load_dotenv


class Notifier(Protocol):
    @property
    def enabled(self) -> bool: ...

    def send(self, text: str, *, blocks: list[dict] | None = None) -> None: ...


class NullNotifier:
    enabled = False

    def send(self, text: str, *, blocks: list[dict] | None = None) -> None:
        return None


@dataclass(frozen=True)
class SlackConfig:
    enabled: bool = False
    webhook_url: str = ""
    bot_token: str = ""
    channel: str = ""
    timeout: float = 10.0

    @classmethod
    def from_env(cls) -> "SlackConfig":
        load_dotenv()
        api_key = os.getenv("SLACK_API_KEY", "").strip()
        webhook_url = os.getenv("SLACK_WEBHOOK_URL", "").strip()
        if not webhook_url and api_key.startswith("https://hooks.slack.com/"):
            webhook_url = api_key
        return cls(
            enabled=os.getenv("SLACK_NOTIFICATIONS_ENABLED", "false").strip().lower()
            in {"1", "true", "yes", "on"},
            webhook_url=webhook_url,
            bot_token=(
                os.getenv("SLACK_BOT_TOKEN", "").strip()
                or (api_key if not webhook_url else "")
            ),
            channel=os.getenv("SLACK_CHANNEL", "").strip(),
            timeout=float(os.getenv("SLACK_TIMEOUT_SECONDS", "10")),
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.webhook_url:
            if not self.webhook_url.startswith("https://hooks.slack.com/"):
                raise ValueError("SLACK_WEBHOOK_URL must use https://hooks.slack.com/")
            return
        if not self.bot_token or not self.channel:
            raise ValueError(
                "Set SLACK_WEBHOOK_URL or both SLACK_BOT_TOKEN and SLACK_CHANNEL"
            )


class SlackNotifier:
    def __init__(
        self,
        config: SlackConfig,
        session: requests.Session | None = None,
    ):
        config.validate()
        self.config = config
        self.session = session or requests.Session()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def send(self, text: str, *, blocks: list[dict] | None = None) -> None:
        if not self.enabled:
            return
        payload: dict = {"text": text}
        if blocks:
            payload["blocks"] = blocks
        if self.config.webhook_url:
            response = self.session.post(
                self.config.webhook_url,
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            if response.text.strip().lower() != "ok":
                raise RuntimeError(f"Slack webhook rejected message: {response.text[:200]}")
            return

        payload["channel"] = self.config.channel
        response = self.session.post(
            "https://slack.com/api/chat.postMessage",
            headers={
                "Authorization": f"Bearer {self.config.bot_token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("ok"):
            raise RuntimeError(f"Slack API error: {result.get('error', 'unknown_error')}")

    def upload_file(
        self, path: Path, *, title: str, initial_comment: str = ""
    ) -> str:
        """Upload a report with Slack's external upload flow and return its permalink."""
        if not self.enabled or not self.config.bot_token:
            return ""
        content = path.read_bytes()
        headers = {"Authorization": f"Bearer {self.config.bot_token}"}
        ticket_response = self.session.get(
            "https://slack.com/api/files.getUploadURLExternal",
            headers=headers,
            params={"filename": path.name, "length": len(content)},
            timeout=self.config.timeout,
        )
        ticket_response.raise_for_status()
        ticket = ticket_response.json()
        if not ticket.get("ok"):
            raise RuntimeError(f"Slack file ticket error: {ticket.get('error', 'unknown_error')}")
        upload = self.session.post(
            ticket["upload_url"], data=content,
            headers={
                "Content-Type": (
                    "application/pdf"
                    if path.suffix.lower() == ".pdf"
                    else (
                        "text/html; charset=utf-8"
                        if path.suffix.lower() == ".html"
                        else "application/octet-stream"
                    )
                )
            },
            timeout=self.config.timeout,
        )
        upload.raise_for_status()
        complete_response = self.session.post(
            "https://slack.com/api/files.completeUploadExternal",
            headers={**headers, "Content-Type": "application/json; charset=utf-8"},
            json={
                "files": [{"id": ticket["file_id"], "title": title}],
                "channel_id": self.config.channel,
                "initial_comment": initial_comment,
            },
            timeout=self.config.timeout,
        )
        complete_response.raise_for_status()
        completed = complete_response.json()
        if not completed.get("ok"):
            raise RuntimeError(
                f"Slack file completion error: {completed.get('error', 'unknown_error')}"
            )
        info_response = self.session.get(
            "https://slack.com/api/files.info",
            headers=headers, params={"file": ticket["file_id"]},
            timeout=self.config.timeout,
        )
        info_response.raise_for_status()
        info = info_response.json()
        if not info.get("ok"):
            raise RuntimeError(f"Slack file info error: {info.get('error', 'unknown_error')}")
        return str(info.get("file", {}).get("permalink") or "")
