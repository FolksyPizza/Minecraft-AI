#!/usr/bin/env python3
"""Build concrete SPECS-aligned pairs from existing sources plus curated Paper/Folia examples."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TITLE_RE = re.compile(r"Title:\s*([^\n\r]+?)\s+(?:Addon:|Syntax Pattern:|Description:|$)", re.IGNORECASE)
ADDON_RE = re.compile(r"Addon:\s*([^\n\r]+?)\s+(?:Syntax|Description:|$)", re.IGNORECASE)
SYNTAX_RE = re.compile(r"(?:Syntax Pattern:|Syntax:)\s*([^\n\r]+?)(?:\s+Description:|\s+Return only|\s+Format exactly|$)", re.IGNORECASE)


ADDON_ALIASES = {
    "skbee": ["skbee"],
    "skript-reflect": ["skript-reflect", "skript reflect", "skriptmirror"],
    "skript-gui": ["skript-gui", "skript gui"],
    "skript-yaml": ["skript-yaml", "skript yaml"],
    "poask": ["poask", "poa sk"],
    "hippo": ["hippo"],
}


PLACEHOLDER_DEFAULTS = [
    (re.compile(r"%player%"), "player"),
    (re.compile(r"%offlineplayer%"), "player"),
    (re.compile(r"%number%"), "1"),
    (re.compile(r"%integer%"), "1"),
    (re.compile(r"%string%"), '"example"'),
    (re.compile(r"%text%"), '"example"'),
    (re.compile(r"%boolean%"), "true"),
    (re.compile(r"%world%"), "world"),
    (re.compile(r"%item(stack)?%"), "stone"),
    (re.compile(r"%location%"), "player's location"),
    (re.compile(r"%entity%"), "player"),
    (re.compile(r"%block%"), "stone"),
]


def read_pairs(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            p = obj.get("prompt")
            c = obj.get("completion")
            if isinstance(p, str) and p.strip() and isinstance(c, str) and c.strip():
                rows.append({"prompt": p.strip(), "completion": c.strip()})
    return rows


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (" ".join(row["prompt"].split()), " ".join(row["completion"].split()))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def map_addon_name(raw: str) -> str | None:
    n = normalize(raw)
    for addon, aliases in ADDON_ALIASES.items():
        if any(alias in n for alias in aliases):
            return addon
    return None


def remove_optional_brackets(text: str) -> str:
    out = text
    while True:
        nxt = re.sub(r"\[[^\[\]]*\]", "", out)
        if nxt == out:
            break
        out = nxt
    return out


def choose_first_alternative(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        inner = match.group(1)
        return inner.split("|", 1)[0].strip()

    out = text
    while True:
        nxt = re.sub(r"\(([^()]*)\)", repl, out)
        if nxt == out:
            break
        out = nxt
    return out


def substitute_placeholders(text: str) -> str:
    out = text
    for pat, replacement in PLACEHOLDER_DEFAULTS:
        out = pat.sub(replacement, out)
    out = re.sub(r"%[^%]+%", "value", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def concretize_pattern(pattern: str) -> str:
    first = pattern.splitlines()[0].strip()
    out = remove_optional_brackets(first)
    out = choose_first_alternative(out)
    out = substitute_placeholders(out)
    return out.strip()


def template_like(text: str) -> bool:
    return (
        "%" in text
        or ("[" in text and "]" in text)
        or ("(" in text and ")" in text and "|" in text)
    )


def clean_concrete_line(text: str) -> str:
    line = re.sub(r"\s+", " ", text.strip())
    line = line.strip("`")
    return line


def valid_line(text: str) -> bool:
    if not text:
        return False
    if template_like(text):
        return False
    if text.endswith(":"):
        return False
    lowered = text.lower()
    banned_markers = (
        "checks if ",
        "check if ",
        "returns ",
        "whether ",
        "description:",
        "example:",
        "format exactly",
        "return only",
    )
    if any(marker in lowered for marker in banned_markers):
        return False
    if text.endswith("."):
        return False
    w = len(text.split())
    return 2 <= w <= 30


def extract_from_skripthub(raw_rows: list[dict[str, str]], max_per_addon: int) -> tuple[list[dict[str, str]], dict[str, int]]:
    counts = {k: 0 for k in ADDON_ALIASES}
    out: list[dict[str, str]] = []

    for row in raw_rows:
        prompt = row["prompt"]
        m_addon = ADDON_RE.search(prompt)
        if not m_addon:
            continue
        addon = map_addon_name(m_addon.group(1))
        if not addon:
            continue
        if counts[addon] >= max_per_addon:
            continue

        m_title = TITLE_RE.search(prompt)
        title = m_title.group(1).strip() if m_title else "server automation"
        m_syntax = SYNTAX_RE.search(prompt)
        syntax = m_syntax.group(1).strip() if m_syntax else row["completion"]
        concrete = clean_concrete_line(concretize_pattern(syntax))
        if not valid_line(concrete):
            continue

        out.append(
            {
                "prompt": f"Write one final executable Skript line for this {addon} task: {title}",
                "completion": concrete,
            }
        )
        out.append(
            {
                "prompt": f"Return one runnable {addon} Skript line for: {title}",
                "completion": concrete,
            }
        )
        counts[addon] += 1
    return out, counts


def extract_from_github(github_rows: list[dict[str, str]], max_rows: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in github_rows:
        if len(out) >= max_rows:
            break
        c = clean_concrete_line(row["completion"])
        if not valid_line(c):
            continue
        out.append({"prompt": row["prompt"], "completion": c})
        out.append(
            {
                "prompt": (
                    "Write one final executable Skript line matching this usage context:\n"
                    f"{row['prompt']}"
                ),
                "completion": c,
            }
        )
    return out


def extract_core_skripthub(raw_rows: list[dict[str, str]], max_rows: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in raw_rows:
        if len(out) >= max_rows:
            break
        prompt = row["prompt"]
        m_syntax = SYNTAX_RE.search(prompt)
        if not m_syntax:
            continue
        m_addon = ADDON_RE.search(prompt)
        if m_addon and map_addon_name(m_addon.group(1)):
            continue
        m_title = TITLE_RE.search(prompt)
        title = m_title.group(1).strip() if m_title else "minecraft automation"
        concrete = clean_concrete_line(concretize_pattern(m_syntax.group(1).strip()))
        if not valid_line(concrete):
            continue
        out.append(
            {
                "prompt": f"Write one final executable Skript line for this task: {title}",
                "completion": concrete,
            }
        )
        out.append(
            {
                "prompt": f"Return one runnable Skript line for: {title}",
                "completion": concrete,
            }
        )
    return out


def curated_code_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    rows.append(
        {
            "prompt": "Write a complete Java Paper command executor that heals the command target by 8 health points.",
            "completion": """public final class HealCommand implements CommandExecutor {\n    @Override\n    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {\n        if (args.length != 1) return false;\n        Player target = Bukkit.getPlayerExact(args[0]);\n        if (target == null) {\n            sender.sendMessage(Component.text(\"Player not found.\"));\n            return true;\n        }\n        double next = Math.min(target.getAttribute(Attribute.GENERIC_MAX_HEALTH).getValue(), target.getHealth() + 8.0D);\n        target.setHealth(next);\n        sender.sendMessage(Component.text(\"Healed \" + target.getName()));\n        return true;\n    }\n}""",
        }
    )
    rows.append(
        {
            "prompt": "Write Java Paper plugin code to register the heal command in onEnable.",
            "completion": """@Override\npublic void onEnable() {\n    PluginCommand cmd = getCommand(\"healplayer\");\n    if (cmd != null) {\n        cmd.setExecutor(new HealCommand());\n    }\n}""",
        }
    )
    rows.append(
        {
            "prompt": "Write a Folia-safe Java snippet that teleports a player using their region scheduler.",
            "completion": """Location destination = new Location(player.getWorld(), 0.5, 80.0, 0.5);\nplayer.getScheduler().run(this, task -> player.teleportAsync(destination), null);""",
        }
    )
    rows.append(
        {
            "prompt": "Write a Folia-safe Java snippet that sends a delayed message to a player after 40 ticks.",
            "completion": """player.getScheduler().runDelayed(this, task -> player.sendMessage(Component.text(\"2 seconds passed\")), null, 40L);""",
        }
    )
    rows.append(
        {
            "prompt": "Write Kotlin Paper listener code that cancels block break for non-op players.",
            "completion": """class BlockGuard : Listener {\n    @EventHandler\n    fun onBreak(event: BlockBreakEvent) {\n        if (!event.player.isOp) {\n            event.isCancelled = true\n            event.player.sendMessage(Component.text(\"You cannot break blocks here.\"))\n        }\n    }\n}""",
        }
    )
    rows.append(
        {
            "prompt": "Write Kotlin code for a simple Paper plugin onEnable log message.",
            "completion": """override fun onEnable() {\n    logger.info(\"Plugin enabled\")\n}""",
        }
    )
    rows.append(
        {
            "prompt": "Write a concrete Skript command that heals the executing player by 4 hearts.",
            "completion": "heal player by 4 hearts",
        }
    )
    rows.append(
        {
            "prompt": "Write a concrete Skript line that teleports player to spawn.",
            "completion": "teleport player to spawn point of world \"world\"",
        }
    )
    rows.append(
        {
            "prompt": "Write a complete Skript command that clears rain in the world.",
            "completion": """command /sun:\n    trigger:\n        set weather in world of player to clear\n        send \"&eWeather set to clear.\" to player""",
        }
    )
    rows.append(
        {
            "prompt": "Write a complete Skript event that gives speed 2 for 10 seconds when a player joins.",
            "completion": """on join:\n    apply speed 2 to player for 10 seconds\n    send \"&aYou received a speed boost.\" to player""",
        }
    )
    rows.append(
        {
            "prompt": "Write a Skript line that stores a player's balance variable.",
            "completion": "set {balance::%uuid of player%} to 500",
        }
    )
    rows.append(
        {
            "prompt": "Write a Java method that prints hello world to console.",
            "completion": """public static void printHello() {\n    System.out.println(\"hello world\");\n}""",
        }
    )
    rows.append(
        {
            "prompt": "Write Python code that prints hello world.",
            "completion": "print(\"hello world\")",
        }
    )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skripthub-in", required=True)
    ap.add_argument("--github-in", required=False, default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-addon", type=int, default=1200)
    ap.add_argument("--max-github-rows", type=int, default=5000)
    ap.add_argument("--max-core-rows", type=int, default=5000)
    args = ap.parse_args()

    skripthub_rows = read_pairs(Path(args.skripthub_in).resolve())
    github_rows = read_pairs(Path(args.github_in).resolve()) if args.github_in else []

    addon_rows, addon_counts = extract_from_skripthub(skripthub_rows, max_per_addon=args.max_per_addon)
    github_concrete = extract_from_github(github_rows, max_rows=args.max_github_rows)
    core_rows = extract_core_skripthub(skripthub_rows, max_rows=args.max_core_rows)
    curated = curated_code_rows()

    merged = dedupe(addon_rows + core_rows + github_concrete + curated)
    out_path = Path(args.out).resolve()
    write_jsonl(out_path, merged)

    report = {
        "skripthub_rows": len(skripthub_rows),
        "github_rows": len(github_rows),
        "rows_from_skripthub_addons": len(addon_rows),
        "rows_from_skripthub_core": len(core_rows),
        "rows_from_github_concrete": len(github_concrete),
        "rows_curated": len(curated),
        "rows_out": len(merged),
        "addon_counts": addon_counts,
        "out": str(out_path),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
