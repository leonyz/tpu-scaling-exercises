# Telemount harness

A harness that lets Claude models play [Telemount](https://hempuli.itch.io/telemount),
the browser puzzle game by Arvi "Hempuli" Teikari (creator of Baba Is You).

## How it works

Telemount is a Clickteam Fusion HTML5 export hosted by itch.io — a single
640×640 `<canvas>` driven entirely by the keyboard, with no game-state API.
The harness therefore plays it the way a person does:

1. **Playwright** drives headless Chromium at the game's direct embed URL
   (`html-classic.itch.zone/html/18353394/index.html`, skipping the itch.io
   page chrome entirely).
2. A **Claude agent loop** (manual tool-use loop over the Messages API)
   receives canvas screenshots and issues keypresses through tools:
   - `press_keys` — a batch of moves (`up/down/left/right`, `z` undo,
     `r` restart, plus `space`/`x`/`enter`); returns a fresh screenshot
     and whether the screen actually changed (a blocked-move signal).
   - `observe` — wait and re-screenshot (for animations).
   - `note` — durable memory. Old screenshots are trimmed from the context
     to bound token usage, so the agent records proven mechanics, level
     numbers, and plans as text notes that survive.
   - `reload_game` — last-resort page reload.
   - `skip_level` (opt-in via `--allow-skip`) — the game's Ctrl+Q/Ctrl+E
     level-skip cheat.
3. Every screenshot, keypress, note, and model response is logged under
   `runs/<timestamp>/` (`log.jsonl`, `shots/*.png`, `notes.md`), so a run
   can be replayed and audited afterwards.

Two non-obvious mechanics the harness handles, found empirically:

- **Keys must be held.** The Clickteam runtime polls the keyboard once per
  frame, so an instantaneous keydown/keyup is silently missed. Every press
  is held for ~150 ms (`--key-hold-ms`).
- **TLS-intercepting proxies.** If `HTTPS_PROXY` is set, Chromium is routed
  through it with `--ssl-version-max=tls1.2`, because some MITM proxies
  reset Chromium's TLS 1.3 post-quantum handshake. Certificate verification
  stays fully enabled (the proxy CA must be in Chromium's NSS store,
  `~/.pki/nssdb`).

## Setup

```bash
pip install -r requirements.txt
playwright install chromium   # or set TELEMOUNT_CHROMIUM to an existing binary
export ANTHROPIC_API_KEY=...  # or use an `ant auth login` profile
```

## Usage

```bash
# Verify the browser plumbing without any API calls
python play_telemount.py --demo

# Let Claude play (default: claude-opus-5, 40 turns)
python play_telemount.py

# Watch it play, with the level-skip cheat available and a bigger budget
python play_telemount.py --headed --allow-skip --max-turns 100

# Cheaper/faster configurations
python play_telemount.py --model claude-sonnet-5 --effort medium
```

Useful flags: `--max-turns`, `--effort low|medium|high|xhigh|max`,
`--keep-images N` (screenshots kept in context), `--start-level N`
(skip ahead before handing over), `--goal "..."` (extra instructions),
`--run-dir PATH`, `--no-fallbacks`.

On Claude Opus 5 / Fable 5 the harness enables server-side refusal
fallbacks by default (`fallbacks: "default"`), so a rare safety decline
re-routes to a fallback model inside the same request instead of killing
the run; disable with `--no-fallbacks`.

## Cost notes

Each turn sends up to `--keep-images` 640×640 screenshots (~750 tokens
each) plus the conversation text. The system prompt carries a cache
breakpoint. A 40-turn Opus 5 run typically lands in the low single-digit
dollars; use `--model claude-sonnet-5 --effort medium` for cheap runs.
