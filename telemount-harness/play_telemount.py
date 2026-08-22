#!/usr/bin/env python3
"""CLI for running a Claude agent against the Telemount browser game.

Examples:
    # Let Claude play for up to 40 turns (requires ANTHROPIC_API_KEY or an
    # `ant auth login` profile):
    python play_telemount.py

    # Watch it play in a visible browser window:
    python play_telemount.py --headed

    # Verify the browser plumbing without any API calls:
    python play_telemount.py --demo
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
from pathlib import Path

from telemount_harness.browser import BrowserConfig, TelemountBrowser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Claude plays Telemount")
    parser.add_argument("--model", default="claude-opus-5", help="Model ID (default: claude-opus-5)")
    parser.add_argument(
        "--effort",
        default="high",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort (default: high)",
    )
    parser.add_argument("--max-turns", type=int, default=40, help="Agent turn budget (default: 40)")
    parser.add_argument(
        "--keep-images",
        type=int,
        default=3,
        help="How many recent screenshots stay in context (default: 3)",
    )
    parser.add_argument("--goal", default=None, help="Extra instructions for this run")
    parser.add_argument(
        "--allow-skip",
        action="store_true",
        help="Expose the Ctrl+Q/Ctrl+E level-skip cheat as a tool",
    )
    parser.add_argument(
        "--no-fallbacks",
        action="store_true",
        help="Disable server-side refusal fallbacks (enabled by default on Opus 5 / Fable 5)",
    )
    parser.add_argument("--headed", action="store_true", help="Show the browser window")
    parser.add_argument(
        "--start-level",
        type=int,
        default=1,
        help="Skip ahead to this level with Ctrl+Q before handing over (default: 1)",
    )
    parser.add_argument("--run-dir", default=None, help="Output directory (default: runs/<timestamp>)")
    parser.add_argument(
        "--key-hold-ms",
        type=int,
        default=150,
        help="How long each key is held; the game misses taps shorter than ~100ms (default: 150)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="No API calls: load the game, play a fixed key sequence, save screenshots",
    )
    return parser.parse_args()


async def demo(browser: TelemountBrowser, run_dir: Path) -> None:
    """Scripted smoke test of the browser layer — no model involved."""
    (run_dir / "shots").mkdir(parents=True, exist_ok=True)
    (run_dir / "shots" / "0001_initial.png").write_bytes(await browser.screenshot())
    await browser.press(["right", "right", "up"])
    (run_dir / "shots" / "0002_after_moves.png").write_bytes(await browser.screenshot())
    await browser.press(["z", "z", "z"])
    (run_dir / "shots" / "0003_after_undo.png").write_bytes(await browser.screenshot())
    print(f"Demo complete. Screenshots in {run_dir / 'shots'}")


async def main() -> None:
    args = parse_args()
    run_dir = Path(
        args.run_dir
        or Path("runs") / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    browser = TelemountBrowser(
        BrowserConfig(headless=not args.headed, key_hold_ms=args.key_hold_ms)
    )
    await browser.start()
    try:
        for _ in range(args.start_level - 1):
            await browser.skip_level("next")
        if args.demo:
            await demo(browser, run_dir)
            return
        from telemount_harness.agent import PlaySession

        session = PlaySession(
            browser=browser,
            run_dir=run_dir,
            model=args.model,
            effort=args.effort,
            max_turns=args.max_turns,
            keep_images=args.keep_images,
            allow_skip=args.allow_skip,
            enable_fallbacks=not args.no_fallbacks,
            goal=args.goal,
        )
        print(f"Run directory: {run_dir}")
        await session.run()
    finally:
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
