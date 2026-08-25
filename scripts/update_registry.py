#!/usr/bin/env python3
"""Update the DSH community plugin registry (registry/plugins.json).

Reads a plugin's `package.json` + `cordis.patch.yml` (+ client source for slot
names) and upserts an entry into `registry/plugins.json`. Shared by:

  - `.github/workflows/plugin-publish.yml`  (after npm publish)
  - `dsh-plugin registry update`            (local maintenance)

Usage:
    python scripts/update_registry.py                            # dry run, default plugin dir
    python scripts/update_registry.py --write                    # write registry/plugins.json
    python scripts/update_registry.py --plugin-dir dsh-plugin/foo --registry registry/plugins.json

Exit code: 0 on success (including "no change"), 1 on error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # PyYAML is a guarftrain core dependency; CI installs it in a venv.
    yaml = None

DEFAULT_PLUGIN_DIR = "dsh-plugin/dsh-client-ui-training-guardian"
DEFAULT_REGISTRY = "registry/plugins.json"

SLOT_INJECT_RE = re.compile(r"ctx\.slots\.inject\(\s*['\"]([^'\"]+)['\"]")
SLOT_MAP_RE = re.compile(r"^    ['\"]([^'\"]+)['\"]:\s*\{", re.MULTILINE)


def fail(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


def load_yaml_file(path: Path) -> Any:
    if yaml is None:
        raise RuntimeError("PyYAML is required (pip install pyyaml)")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_slots(pkg_dir: Path) -> list[str]:
    """Collect slot names: `ctx.slots.inject()` call sites + SlotMap keys."""
    slots: list[str] = []
    seen: set[str] = set()

    src_root = pkg_dir / "src"
    if not src_root.is_dir():
        return slots

    for path in sorted(src_root.rglob("*.ts")) + sorted(src_root.rglob("*.tsx")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in SLOT_INJECT_RE.finditer(text):
            if m.group(1) not in seen:
                seen.add(m.group(1))
                slots.append(m.group(1))

    # Fall back to the SlotMap type declaration (slots-augment.ts).
    if not slots:
        for path in sorted(src_root.rglob("slots-augment.ts")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for m in SLOT_MAP_RE.finditer(text):
                if m.group(1) not in seen:
                    seen.add(m.group(1))
                    slots.append(m.group(1))
    return slots


def build_entry(pkg_dir: Path, source_url: str | None) -> dict[str, Any]:
    """Build a registry entry from package.json + cordis.patch.yml."""
    manifest_path = pkg_dir / "package.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} not found")
    pkg = json.loads(manifest_path.read_text(encoding="utf-8"))

    dsh = pkg.get("dsh") or {}
    bundle = dsh.get("bundle") or {}
    client = dsh.get("client") or {}

    patch_path = pkg_dir / Path(str(bundle.get("patch", "./cordis.patch.yml")))
    if not patch_path.is_file():
        raise FileNotFoundError(f"cordis.patch.yml not found at {patch_path}")

    patch = load_yaml_file(patch_path)
    meta: dict[str, Any] = {}
    insert_name: str | None = None

    for item in patch if isinstance(patch, list) else [patch]:
        if not isinstance(item, dict):
            continue
        inserts = item.get("insert") or []
        for entry in inserts if isinstance(inserts, list) else []:
            if isinstance(entry, dict) and entry.get("id"):
                insert_name = str(entry.get("name", ""))
        if isinstance(item.get("meta"), dict):
            meta = item["meta"]

    if not meta:
        raise ValueError("cordis.patch.yml missing meta block")

    meta_version = str(meta.get("version", ""))
    pkg_version = str(pkg.get("version", ""))
    if meta_version != pkg_version:
        print(
            f"warning: meta.version {meta_version!r} != package.json version "
            f"{pkg_version!r} — using package.json value"
        )

    skills: list[dict[str, str]] = []
    for skill in meta.get("skills") or []:
        if not isinstance(skill, dict) or not skill.get("id"):
            continue
        when = skill.get("whenToUse")
        if isinstance(when, str):
            when = " ".join(when.split())
        skills.append(
            {
                "id": str(skill["id"]),
                "name": str(skill.get("name", skill["id"])),
                "description": " ".join(str(skill.get("description", "")).split()),
                "whenToUse": str(when or ""),
            }
        )

    keywords = pkg.get("keywords") or meta.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]

    entry: dict[str, Any] = {
        "name": pkg.get("name", insert_name or ""),
        "version": pkg_version,
        "description": pkg.get("description") or str(meta.get("description", "")),
        "author": pkg.get("author") or str(meta.get("author", "")),
        "license": pkg.get("license") or str(meta.get("license", "")),
        "homepage": pkg.get("homepage") or str(meta.get("homepage", "")),
        "npm": pkg.get("name", ""),
        "keywords": [str(k) for k in keywords],
        "dsh": dsh.get("engines", {}).get("dsh", ">=0.1.0"),
        "platform": client.get("platform", "web"),
        "slots": extract_slots(pkg_dir),
        "skills": skills,
        "updated_at": date.today().isoformat(),
    }
    # `source` is hand-curated (tree URL, mirror, ...) — only override when
    # explicitly requested; the merge in update_registry() preserves the rest.
    if source_url:
        entry["source"] = source_url

    return entry


def update_registry(
    plugin_dir: Path,
    registry_path: Path,
    *,
    write: bool = False,
    source_url: str | None = None,
    registry_url: str | None = None,
) -> int:
    """Upsert the plugin entry into the registry file."""
    entry = build_entry(plugin_dir, source_url)

    if registry_path.is_file():
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        data = {"version": 1, "plugins": []}

    data.setdefault("version", 1)
    if registry_url:
        data["registry_url"] = registry_url

    plugins = list(data.get("plugins") or [])
    existing = next((p for p in plugins if p.get("name") == entry["name"]), None)

    # Preserve fields the manifest cannot provide (e.g. hand-curated source).
    if existing:
        entry = {**existing, **entry}
        entry.setdefault("source", entry.get("homepage") or "")

    updated = [
        entry if p.get("name") == entry["name"] else p
        for p in plugins
    ]
    if all(p.get("name") != entry["name"] for p in updated):
        updated.append(entry)
    updated.sort(key=lambda p: str(p.get("name", "")))

    new_data = {**data, "plugins": updated}
    new_data["updated_at"] = date.today().isoformat()

    changed = json.dumps(new_data, sort_keys=True, ensure_ascii=False) != json.dumps(
        data, sort_keys=True, ensure_ascii=False
    )
    if changed:
        new_data["version"] = int(data.get("version", 1)) + 1
    else:
        new_data["version"] = data.get("version", 1)
        new_data["updated_at"] = data.get("updated_at", new_data["updated_at"])

    if write:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        with registry_path.open("w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    verb = "updated" if changed else "unchanged"
    print(
        f"{'[write]' if write else '[dry-run]'} {registry_path}: {verb} "
        f"(v{new_data['version']}, {len(new_data['plugins'])} plugin(s))"
    )
    print(
        f"  entry: {entry['name']}@{entry['version']} "
        f"slots={entry.get('slots')} skills={[s['id'] for s in entry.get('skills', [])]}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="update_registry.py",
        description="Upsert a DSH plugin entry into registry/plugins.json",
    )
    parser.add_argument("--plugin-dir", default=DEFAULT_PLUGIN_DIR, help="Plugin package directory")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY, help="Path to registry/plugins.json")
    parser.add_argument("--write", action="store_true", help="Write the registry file (default: dry run)")
    parser.add_argument("--source-url", help="Override the entry's source URL")
    parser.add_argument("--registry-url", help="Set/update the top-level registry_url")
    args = parser.parse_args(argv)

    try:
        return update_registry(
            Path(args.plugin_dir),
            Path(args.registry),
            write=args.write,
            source_url=args.source_url,
            registry_url=args.registry_url,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        return fail(str(exc))


if __name__ == "__main__":
    sys.exit(main())
