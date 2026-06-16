#!/usr/bin/env python3
"""
ask.py — Ask the AI one question about the methane plume model, from the command line.

This is the simple, reliable way to ask questions. Unlike an interactive chat (which
needs a real terminal and kept failing when launched indirectly), this takes your
question as a command-line argument, prints the answer, and exits. It works anywhere
you have internet and an OPENAI_API_KEY in the .env file.

USAGE
─────
    python3 ask.py "what does the green dashed line on the graph mean?"
    python3 ask.py "why does the plume get narrower in stable air?"
    python3 ask.py "what is clamp_to_table and why does the inversion need it?"

The answer is grounded in this project's actual physics, tests, guard rails, and the
current graph's numbers (peak ppm, detection reach, etc.). No API key or internet →
it tells you exactly what's missing instead of failing silently.
"""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ui.explain import ask_once, scenario_facts

console = Console(highlight=False)


def main() -> int:
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        console.print(Panel(
            Text(
                'Ask a question as an argument, in quotes. For example:\n\n'
                '  python3 ask.py "what does the green dashed line mean?"\n'
                '  python3 ask.py "why does wind speed change the peak reading?"',
                style="dim white",
            ),
            title="[bold bright_cyan] How to ask [/bold bright_cyan]",
            border_style="steel_blue1",
            padding=(1, 2),
        ))
        return 0

    # Echo the question, then compute the current graph's numbers (best-effort) so
    # the answer can reference them. A compute hiccup must not block the question.
    console.print()
    console.print(Panel(
        Text(question, style="bright_cyan"),
        title="[bold] Your question [/bold]",
        border_style="grey30", padding=(0, 2),
    ))

    try:
        facts = scenario_facts()
    except Exception:
        facts = None   # ask without graph grounding rather than fail

    with console.status("[dim]Thinking…[/dim]", spinner="dots"):
        answer = ask_once(question, facts)

    console.print(Panel(
        Text(answer, style="white"),
        title="[bold bright_white] Answer [/bold bright_white]",
        border_style="steel_blue1", padding=(1, 2),
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
