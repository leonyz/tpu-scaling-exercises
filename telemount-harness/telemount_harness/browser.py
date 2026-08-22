"""Playwright driver for the Telemount HTML5 game.

Telemount (https://hempuli.itch.io/telemount) is a Clickteam Fusion HTML5
export served by itch.io. The playable page is a bare index.html with a
single 640x640 <canvas id="MMFCanvas"> element; all input is keyboard.

Two quirks matter for automation:

* The Clickteam runtime polls keyboard state once per frame, so a
  zero-duration keydown/keyup pair is usually missed. Every key press must
  be held for ~100-200ms (see DEFAULT_KEY_HOLD_MS).
* Some corporate/TLS-intercepting proxies cannot parse Chromium's TLS 1.3
  post-quantum ClientHello and reset the connection. When a proxy is used,
  we cap Chromium at TLS 1.2 (`--ssl-version-max=tls1.2`); certificate
  verification stays fully enabled.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass, field

from playwright.async_api import async_playwright, Browser, Page, Playwright

DEFAULT_GAME_URL = "https://html-classic.itch.zone/html/18353394/index.html?v=1784182371"
DEFAULT_KEY_HOLD_MS = 150
DEFAULT_INTER_KEY_DELAY_S = 0.20
DEFAULT_SETTLE_DELAY_S = 0.6
DEFAULT_LOAD_WAIT_S = 15.0

# Friendly key names the agent uses -> Playwright key identifiers.
KEY_MAP = {
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "z": "z",
    "undo": "z",
    "r": "r",
    "restart": "r",
    "space": " ",
    "x": "x",
    "enter": "Enter",
    "escape": "Escape",
}


@dataclass
class BrowserConfig:
    game_url: str = DEFAULT_GAME_URL
    headless: bool = True
    executable_path: str | None = None  # None -> Playwright-managed Chromium
    key_hold_ms: int = DEFAULT_KEY_HOLD_MS
    inter_key_delay_s: float = DEFAULT_INTER_KEY_DELAY_S
    settle_delay_s: float = DEFAULT_SETTLE_DELAY_S
    load_wait_s: float = DEFAULT_LOAD_WAIT_S
    extra_args: list[str] = field(default_factory=list)

    @staticmethod
    def _detect_executable() -> str | None:
        for candidate in (os.environ.get("TELEMOUNT_CHROMIUM"), "/opt/pw-browsers/chromium"):
            if candidate and os.path.exists(candidate):
                return candidate
        return None

    def resolved_executable(self) -> str | None:
        return self.executable_path or self._detect_executable()


class TelemountBrowser:
    """Owns the browser session and exposes screenshot/keypress primitives."""

    def __init__(self, config: BrowserConfig | None = None):
        self.config = config or BrowserConfig()
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None
        self.keys_pressed = 0

    async def start(self) -> None:
        self._playwright = await async_playwright().start()
        launch_kwargs: dict = {"headless": self.config.headless}
        executable = self.config.resolved_executable()
        if executable:
            launch_kwargs["executable_path"] = executable
        args = list(self.config.extra_args)
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy:
            launch_kwargs["proxy"] = {"server": proxy}
            # TLS-intercepting proxies often reset Chromium's TLS 1.3
            # post-quantum handshake; capping the version fixes that while
            # keeping certificate verification on.
            args.append("--ssl-version-max=tls1.2")
        if args:
            launch_kwargs["args"] = args
        self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        self._page = await self._browser.new_page(viewport={"width": 800, "height": 800})
        await self._load_game()

    async def _load_game(self) -> None:
        assert self._page is not None
        await self._page.goto(self.config.game_url, wait_until="load")
        canvas = self._page.locator("#MMFCanvas")
        await canvas.wait_for(state="visible", timeout=60_000)
        await asyncio.sleep(self.config.load_wait_s)
        await canvas.click()  # focus the page so keyboard input is delivered

    async def reload(self) -> None:
        """Reload the game page (recovers from a wedged runtime; loses all progress)."""
        await self._load_game()

    async def screenshot(self) -> bytes:
        assert self._page is not None
        return await self._page.locator("#MMFCanvas").screenshot()

    async def press(self, keys: list[str]) -> None:
        """Press a sequence of friendly-named keys, holding each one."""
        assert self._page is not None
        for name in keys:
            key = KEY_MAP.get(name.lower())
            if key is None:
                raise ValueError(f"unknown key {name!r}; valid keys: {sorted(KEY_MAP)}")
            await self._page.keyboard.press(key, delay=self.config.key_hold_ms)
            self.keys_pressed += 1
            await asyncio.sleep(self.config.inter_key_delay_s)
        await asyncio.sleep(self.config.settle_delay_s)

    async def skip_level(self, direction: str) -> None:
        """Ctrl+Q skips to the next level, Ctrl+E returns to the previous one."""
        assert self._page is not None
        key = {"next": "q", "previous": "e"}.get(direction)
        if key is None:
            raise ValueError("direction must be 'next' or 'previous'")
        await self._page.keyboard.down("Control")
        try:
            await self._page.keyboard.press(key, delay=self.config.key_hold_ms)
        finally:
            await self._page.keyboard.up("Control")
        await asyncio.sleep(self.config.settle_delay_s)

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()


def image_digest(png_bytes: bytes) -> str:
    return hashlib.md5(png_bytes).hexdigest()[:10]
