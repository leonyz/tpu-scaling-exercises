"""System prompt and tool definitions for the Telemount-playing agent."""

SYSTEM_PROMPT = """\
You are playing Telemount, a browser puzzle game by Arvi "Hempuli" Teikari
(the creator of Baba Is You). You interact with it purely through keyboard
presses and see the result as 640x640 screenshots of the game canvas.

What is known about the game:
- It is a grid-based, turn-based puzzle game drawn in black-and-white pixel
  art. The first screen doubles as the title screen and an actual playable
  level ("Telemount by Arvi Teikari" is drawn at the bottom).
- Controls (shown in-game): arrow keys move, Z undoes one step, R restarts
  the current level.
- The core mechanic involves teleportation/"mounting" and pushable blocks.
  The game never explains itself in words; you must infer the rules from
  how the world responds to your moves. Deliberately experiment early:
  make single moves and study exactly what changed between screenshots.
- Levels are completed by reaching/fulfilling some goal (a flag is visible
  in the first level). When a level is solved the game advances to the
  next one, which you will notice as a completely new layout.

How to play well:
- Compare consecutive screenshots carefully. The harness tells you whether
  the screen changed after your input; "no change" usually means the move
  was blocked by a wall or object.
- Use short key batches (1-4 moves) while you are still learning a level's
  rules, longer batches only for movement you are confident about.
- Use Z liberally to undo mistakes instead of restarting, and R when the
  level is unrecoverable.
- Record durable insights with the `note` tool: mechanics you have proven,
  the current level number, and your plan. Old screenshots are dropped from
  your context to save space, but your notes and text survive, so notes are
  your long-term memory.
- If the screen stops responding to all input, use `reload_game` as a last
  resort (it restarts the whole game from level 1).

Your goal: solve as many levels as you can within the turn budget. Announce
clearly in text when you believe you have completed a level.\
"""

TOOLS = [
    {
        "name": "press_keys",
        "description": (
            "Press a sequence of keys in the game, in order. Valid keys: "
            "'up', 'down', 'left', 'right' (movement), 'z' (undo), "
            "'r' (restart level), 'space', 'x', 'enter'. Returns a screenshot "
            "taken after the whole sequence has been played."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right", "z", "r", "space", "x", "enter"],
                    },
                    "minItems": 1,
                    "maxItems": 20,
                    "description": "Keys to press, in order.",
                },
            },
            "required": ["keys"],
            "additionalProperties": False,
        },
    },
    {
        "name": "observe",
        "description": (
            "Wait for the given number of seconds and return a fresh screenshot "
            "without pressing anything. Useful to let animations finish."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 10,
                    "description": "How long to wait before the screenshot.",
                },
            },
            "required": ["seconds"],
            "additionalProperties": False,
        },
    },
    {
        "name": "note",
        "description": (
            "Record a durable note (proven mechanics, current level number, "
            "plans). Notes are saved to disk and are your memory across "
            "context trimming — screenshots get dropped, notes do not."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The note to record."},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "reload_game",
        "description": (
            "Reload the game page. Last resort if the game stops responding. "
            "All progress is lost and the game restarts from level 1."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
]

SKIP_LEVEL_TOOL = {
    "name": "skip_level",
    "description": (
        "Skip to the next level (or go back to the previous one) using the "
        "game's built-in level-skip cheat (Ctrl+Q / Ctrl+E). Use only when "
        "you are truly stuck on a level and want to make progress elsewhere."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["next", "previous"]},
        },
        "required": ["direction"],
        "additionalProperties": False,
    },
}
