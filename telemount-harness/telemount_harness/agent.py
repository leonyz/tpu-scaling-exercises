"""Claude agent loop that plays Telemount through TelemountBrowser."""

from __future__ import annotations

import asyncio
import base64
import datetime
import json
from pathlib import Path

import anthropic

from .browser import TelemountBrowser, image_digest
from .prompts import SKIP_LEVEL_TOOL, SYSTEM_PROMPT, TOOLS

FALLBACK_CAPABLE_PREFIXES = ("claude-opus-5", "claude-fable-5")


class PlaySession:
    def __init__(
        self,
        browser: TelemountBrowser,
        run_dir: Path,
        model: str = "claude-opus-5",
        effort: str = "high",
        max_turns: int = 40,
        keep_images: int = 3,
        allow_skip: bool = False,
        enable_fallbacks: bool = True,
        goal: str | None = None,
    ):
        self.browser = browser
        self.run_dir = run_dir
        self.model = model
        self.effort = effort
        self.max_turns = max_turns
        self.keep_images = keep_images
        self.tools = TOOLS + ([SKIP_LEVEL_TOOL] if allow_skip else [])
        self.enable_fallbacks = enable_fallbacks and model.startswith(FALLBACK_CAPABLE_PREFIXES)
        self.goal = goal
        self.client = anthropic.Anthropic()
        self.messages: list = []
        self.shot_count = 0
        self.last_digest: str | None = None
        (self.run_dir / "shots").mkdir(parents=True, exist_ok=True)

    # ---------- logging ----------

    def _log(self, event: dict) -> None:
        event["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(self.run_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(event) + "\n")

    def _save_shot(self, png: bytes, label: str) -> Path:
        self.shot_count += 1
        path = self.run_dir / "shots" / f"{self.shot_count:04d}_{label}.png"
        path.write_bytes(png)
        return path

    def _append_note(self, text: str) -> None:
        with open(self.run_dir / "notes.md", "a") as f:
            f.write(f"- {text}\n")

    # ---------- screenshots as content blocks ----------

    async def _screenshot_blocks(self, label: str) -> tuple[list, str]:
        png = await self.browser.screenshot()
        digest = image_digest(png)
        changed = "yes" if digest != self.last_digest else "NO (screen identical to previous screenshot)"
        self.last_digest = digest
        path = self._save_shot(png, label)
        self._log({"event": "screenshot", "path": str(path), "digest": digest})
        status = (
            f"Screenshot after '{label}'. Screen changed since last screenshot: {changed}. "
            f"Total keys pressed so far: {self.browser.keys_pressed}."
        )
        blocks = [
            {"type": "text", "text": status},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(png).decode(),
                },
            },
        ]
        return blocks, digest

    def _trim_old_images(self) -> None:
        """Replace image blocks in all but the most recent tool results with a
        placeholder, so context stays bounded on long runs."""
        seen = 0
        for message in reversed(self.messages):
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for item in content:
                if not (isinstance(item, dict) and item.get("type") == "tool_result"):
                    continue
                inner = item.get("content")
                if not isinstance(inner, list):
                    continue
                has_image = any(b.get("type") == "image" for b in inner if isinstance(b, dict))
                if not has_image:
                    continue
                seen += 1
                if seen > self.keep_images:
                    item["content"] = [
                        b for b in inner if not (isinstance(b, dict) and b.get("type") == "image")
                    ] + [{"type": "text", "text": "[screenshot removed to save context]"}]

    # ---------- tool dispatch ----------

    async def _run_tool(self, name: str, tool_input: dict) -> list:
        if name == "press_keys":
            keys = tool_input["keys"]
            await self.browser.press(keys)
            self._log({"event": "press_keys", "keys": keys})
            blocks, _ = await self._screenshot_blocks("press_" + "-".join(keys)[:40])
            return blocks
        if name == "observe":
            await asyncio.sleep(float(tool_input["seconds"]))
            self._log({"event": "observe", "seconds": tool_input["seconds"]})
            blocks, _ = await self._screenshot_blocks("observe")
            return blocks
        if name == "note":
            self._append_note(tool_input["text"])
            self._log({"event": "note", "text": tool_input["text"]})
            return [{"type": "text", "text": "Note recorded."}]
        if name == "reload_game":
            await self.browser.reload()
            self.browser.keys_pressed = 0
            self._log({"event": "reload_game"})
            blocks, _ = await self._screenshot_blocks("reload")
            return blocks
        if name == "skip_level":
            await self.browser.skip_level(tool_input["direction"])
            self._log({"event": "skip_level", "direction": tool_input["direction"]})
            blocks, _ = await self._screenshot_blocks(f"skip_{tool_input['direction']}")
            return blocks
        raise ValueError(f"unknown tool {name!r}")

    # ---------- model calls ----------

    def _create_message(self):
        kwargs = dict(
            model=self.model,
            max_tokens=16000,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=self.tools,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            messages=self.messages,
        )
        if self.enable_fallbacks:
            return self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                **kwargs,
            )
        return self.client.messages.create(**kwargs)

    # ---------- main loop ----------

    async def run(self) -> None:
        first_blocks, _ = await self._screenshot_blocks("initial")
        opening = (
            "The game has loaded. Here is the initial screen. "
            "Start playing — remember to experiment to learn the mechanics."
        )
        if self.goal:
            opening += f"\nAdditional instructions for this run: {self.goal}"
        self.messages.append(
            {"role": "user", "content": [{"type": "text", "text": opening}, *first_blocks]}
        )

        for turn in range(1, self.max_turns + 1):
            self._trim_old_images()
            response = self._create_message()
            self._log(
                {
                    "event": "model_response",
                    "turn": turn,
                    "stop_reason": response.stop_reason,
                    "usage": response.usage.to_dict(),
                    "request_id": response._request_id,
                }
            )
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"[turn {turn}] {block.text.strip()}")
                    self._log({"event": "agent_text", "turn": turn, "text": block.text})

            if response.stop_reason == "refusal":
                details = response.stop_details
                print(f"Model refused (category={details.category if details else None}); stopping.")
                self._log({"event": "refusal", "turn": turn})
                return

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                print(f"Agent ended the session after {turn} turns.")
                self._log({"event": "session_end", "turn": turn, "reason": "agent_done"})
                return

            self.messages.append({"role": "assistant", "content": response.content})
            results = []
            for tool_use in tool_uses:
                try:
                    content = await self._run_tool(tool_use.name, tool_use.input)
                    results.append(
                        {"type": "tool_result", "tool_use_id": tool_use.id, "content": content}
                    )
                except Exception as exc:  # report tool failures to the model
                    self._log({"event": "tool_error", "tool": tool_use.name, "error": str(exc)})
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": f"Tool error: {exc}",
                            "is_error": True,
                        }
                    )
            self.messages.append({"role": "user", "content": results})

        print(f"Turn budget of {self.max_turns} exhausted; stopping.")
        self._log({"event": "session_end", "reason": "max_turns"})
