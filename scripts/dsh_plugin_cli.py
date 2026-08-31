#!/usr/bin/env python3
"""
dsh-plugin — DSH Community Plugin CLI

Wraps `dsh plugin --profile <name>` with community-friendly commands:
  list    — list installed plugins with metadata
  search  — search the community registry
  info    — show detailed info about a plugin
  add     — install a plugin (delegates to pnpm via dsh plugin)
  remove  — uninstall a plugin

Usage:
    dsh-plugin list --profile web
    dsh-plugin search training
    dsh-plugin info @rrrelink/dsh-client-ui-training-guardian
    dsh-plugin add @rrrelink/dsh-client-ui-training-guardian --profile web
    dsh-plugin remove @rrrelink/dsh-client-ui-training-guardian --profile web
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from urllib.request import urlopen, Request
    from urllib.error import URLError
except ImportError:
    urlopen = None  # type: ignore[assignment]

REGISTRY_URL = (
    "https://raw.githubusercontent.com/Washington5533/Guid-traince/main/registry/plugins.json"
)
LOCAL_REGISTRY_CACHE = Path.home() / ".dsh" / "plugins" / "registry.json"
PROFILE_DIRS = {
    "web": Path.home() / ".dsh" / "profiles" / "web",
    "tui": Path.home() / ".dsh" / "profiles" / "tui",
    "headless": Path.home() / ".dsh" / "profiles" / "headless",
}


def resolve_profile_dir(profile: str) -> Path:
    """Resolve the dsh-wsl profile directory for a given profile name."""
    if profile in PROFILE_DIRS:
        return PROFILE_DIRS[profile]
    # WSL-style path: ~/.dsh/profiles/<profile>
    return Path.home() / ".dsh" / "profiles" / profile


def read_profile_packages(profile_dir: Path) -> dict[str, dict[str, Any]]:
    """Read installed package metadata from a profile's node_modules."""
    packages: dict[str, dict[str, Any]] = {}
    node_modules = profile_dir / "node_modules"
    if not node_modules.is_dir():
        return packages

    for pkg_dir in node_modules.iterdir():
        if pkg_dir.name.startswith("."):
            continue
        manifest = pkg_dir / "package.json"
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "dsh" in data or ".pnpm" in pkg_dir.name:
            packages[pkg_dir.name] = data
    return packages


def load_cordis_patch(pkg_dir: Path) -> dict[str, Any] | None:
    """Load and return the cordis.patch.yml content for a package."""
    patch = pkg_dir / "cordis.patch.yml"
    if not patch.is_file():
        return None
    # Simple YAML parsing for our known structure
    # We only need to extract the meta block
    text = patch.read_text(encoding="utf-8")
    meta: dict[str, Any] = {}
    in_meta = False
    meta_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "meta:":
            in_meta = True
            continue
        if in_meta:
            if stripped and not stripped.startswith(" ") and not stripped.startswith("-"):
                break
            meta_lines.append(stripped)
    if meta_lines:
        # Parse simple key: value pairs
        for ml in meta_lines:
            m = re.match(r'^(\w[\w]*)\s*:\s*(.+)$', ml)
            if m:
                key = m.group(1)
                val = m.group(2).strip().strip('"').strip("'")
                meta[key] = val
    return meta if meta else None


# ---------------------------------------------------------------------------
# Repo/plugin layout helpers (shared by validate / scaffold / release)
# ---------------------------------------------------------------------------

DEFAULT_PLUGIN_DIR = "dsh-plugin/dsh-client-ui-training-guardian"
PLUGIN_NAME_RE = re.compile(r"^(@[a-z0-9-_.]+/)?[a-z0-9-_.]+$")
KEBAB_ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
SLOT_INJECT_RE = re.compile(r"ctx\.slots\.inject\(\s*['\"]([^'\"]+)['\"]")
SLOT_MAP_KEY_RE = re.compile(r"^    ['\"]([^'\"]+)['\"]:\s*\{", re.MULTILINE)
MODULE_LOADER_RE = re.compile(r"__ModuleLoader__\.load")


def find_repo_root(start: Path | None = None) -> Path | None:
    """Walk up from `start` looking for a git repo or registry/plugins.json."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() or (candidate / "registry" / "plugins.json").is_file():
            return candidate
    return None


def resolve_plugin_dir(args: argparse.Namespace) -> Path:
    """Resolve --plugin-dir, defaulting to the repo's reference plugin."""
    if getattr(args, "plugin_dir", None):
        return Path(args.plugin_dir).resolve()
    repo = find_repo_root()
    if repo and (repo / DEFAULT_PLUGIN_DIR).is_dir():
        return (repo / DEFAULT_PLUGIN_DIR).resolve()
    return Path(DEFAULT_PLUGIN_DIR).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bump_version(version: str, bump: str) -> str:
    parts = [int(p) for p in version.split(".") if p.isdigit()]
    if len(parts) < 3:
        raise ValueError(f"version {version!r} is not semver x.y.z")
    major, minor, patch = parts[:3]
    if bump == "major":
        major, minor, patch = major + 1, 0, 0
    elif bump == "minor":
        minor, patch = minor + 1, 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


def _update_patch_version(patch_path: Path, old: str, new: str) -> None:
    """Rewrite the `version:` line inside cordis.patch.yml's meta block."""
    text = patch_path.read_text(encoding="utf-8")
    pattern = rf'^(?P<indent>\s*)version:\s*["\']?{re.escape(old)}["\']?\s*$'
    updated, count = re.subn(pattern, rf'\g<indent>version: "{new}"', text, count=1, flags=re.MULTILINE)
    if count == 0:
        raise ValueError(f"version {old!r} not found in {patch_path}")
    patch_path.write_text(updated, encoding="utf-8")


def _run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def _load_meta(pkg_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (package.json data, cordis.patch.yml meta block)."""
    pkg = load_json(pkg_dir / "package.json")
    patch = pkg_dir / "cordis.patch.yml"
    meta: dict[str, Any] = {}
    if patch.is_file():
        import yaml as _yaml  # pyyaml is a guarftrain dependency

        data = _yaml.safe_load(patch.read_text(encoding="utf-8"))
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict) and isinstance(item.get("meta"), dict):
                meta = item["meta"]
    return pkg, meta


# ---------------------------------------------------------------------------
# validate — run the PLUGIN_STANDARDS checklist against a plugin directory
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a plugin package against docs/PLUGIN_STANDARDS.md."""
    pkg_dir = resolve_plugin_dir(args)
    print(f"Validating plugin at {pkg_dir}\n")

    results: list[tuple[str, str]] = []  # (status, message); ok / warn / fail

    def check(ok: bool, msg: str, hard: bool = True) -> None:
        results.append(("ok" if ok else ("fail" if hard else "warn"), msg))

    if not pkg_dir.is_dir():
        print(f"error: plugin dir not found: {pkg_dir}")
        return 1

    manifest = pkg_dir / "package.json"
    check(manifest.is_file(), "package.json exists")

    pkg: dict[str, Any] = {}
    if manifest.is_file():
        pkg = load_json(manifest)

    dsh = pkg.get("dsh") or {}
    engines = dsh.get("engines") or {}
    bundle = dsh.get("bundle") or {}
    client = dsh.get("client") or {}
    exports = pkg.get("exports") or {}

    check(bool(engines.get("dsh")), "package.json has dsh.engines.dsh")
    check(bool(bundle.get("patch")), "package.json has dsh.bundle.patch")

    patch_path = pkg_dir / str(bundle.get("patch", "cordis.patch.yml"))
    check(patch_path.is_file(), f"bundle patch exists ({patch_path.name})")

    meta: dict[str, Any] = {}
    if patch_path.is_file():
        try:
            _, meta = _load_meta(pkg_dir)
            check(bool(meta), "cordis.patch.yml has meta block")

            import yaml as _yaml

            patch = _yaml.safe_load(patch_path.read_text(encoding="utf-8"))
            insert_entries: list[dict[str, Any]] = []
            for item in patch if isinstance(patch, list) else [patch]:
                if isinstance(item, dict):
                    inserts = item.get("insert") or []
                    if isinstance(inserts, list):
                        insert_entries.extend(e for e in inserts if isinstance(e, dict))

            check(bool(insert_entries), "cordis.patch.yml has - insert: block")
            if insert_entries:
                insert_id = str(insert_entries[0].get("id", ""))
                insert_name = str(insert_entries[0].get("name", ""))
                check(bool(KEBAB_ID_RE.match(insert_id)), f"insert id is kebab-case ({insert_id!r})")
                check(
                    insert_name == pkg.get("name"),
                    f"insert name matches package.json name ({insert_name!r})",
                )

            meta_version = str(meta.get("version", ""))
            check(
                meta_version == str(pkg.get("version", "")),
                f"meta.version == package.json version ({meta_version!r})",
            )
            desc = str(meta.get("description", ""))
            check(len(desc) <= 120, f"meta.description <= 120 chars (currently {len(desc)})")
            check(bool(meta.get("author")), "meta.author set")
            check(bool(meta.get("license")), "meta.license set (SPDX)")
        except Exception as exc:  # noqa: BLE001 — report as a failed check
            check(False, f"cordis.patch.yml parse failed: {exc}")

    skills = meta.get("skills") or []
    check(bool(skills), "meta.skills declares at least one skill (recommended)", hard=False)
    for skill in skills:
        if isinstance(skill, dict):
            check(
                bool(skill.get("id") and skill.get("name") and skill.get("description")),
                f"skill '{skill.get('id', '?')}' has id/name/description",
            )

    platform = client.get("platform", "web")
    check(platform in ("web", "tui"), f"dsh.client.platform is 'web' or 'tui' ({platform!r})")

    check((pkg_dir / "README.md").is_file(), "README.md exists")
    check((pkg_dir / "README.zh.md").is_file(), "README.zh.md exists (recommended)", hard=False)
    check((pkg_dir / "tsconfig.json").is_file(), "tsconfig.json exists")
    check((pkg_dir / "tsdown.config.ts").is_file(), "tsdown.config.ts exists (dual-target)")

    test_files = list((pkg_dir / "tests").glob("*.test.ts")) if (pkg_dir / "tests").is_dir() else []
    check(bool(test_files), f"tests/ has vitest tests ({len(test_files)} found)")

    client_export = exports.get("./client") or {}
    check(bool(client_export.get("default")), "package.json exports './client' bundle path")

    dist_mjs = pkg_dir / "dist" / "index.mjs"
    dist_client = pkg_dir / "dist-client" / "client" / "index.js"
    check(dist_mjs.is_file(), "dist/index.mjs exists", hard=False)
    check(dist_client.is_file(), "dist-client/client/index.js exists", hard=False)
    if dist_client.is_file():
        check(
            bool(MODULE_LOADER_RE.search(dist_client.read_text(encoding="utf-8"))),
            "client bundle is wrapped in __ModuleLoader__.load",
            hard=False,
        )

    # Slots declared in slots-augment.ts must cover every inject() call site.
    src_root = pkg_dir / "src" / "client"
    used_slots: set[str] = set()
    if src_root.is_dir():
        for path in sorted(src_root.rglob("*.ts")) + sorted(src_root.rglob("*.tsx")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            used_slots.update(SLOT_INJECT_RE.findall(text))
    if used_slots:
        augment = pkg_dir / "src" / "client" / "slots-augment.ts"
        check(augment.is_file(), "slots-augment.ts exists (slots are injected)")
        if augment.is_file():
            declared = set(
                SLOT_MAP_KEY_RE.findall(augment.read_text(encoding="utf-8"))
            )
            missing = used_slots - declared
            check(
                not missing,
                f"slots-augment.ts declares all injected slots (missing: {sorted(missing)})",
            )
    else:
        results.append(("warn", "no ctx.slots.inject() call sites found"))

    # Repo-level requirements (only when the plugin lives in a repo).
    repo = find_repo_root(pkg_dir)
    if repo:
        check(
            (repo / ".github" / "workflows" / "plugin-publish.yml").is_file(),
            "repo has .github/workflows/plugin-publish.yml",
        )
        registry = repo / "registry" / "plugins.json"
        if registry.is_file():
            data = load_json(registry)
            names = [p.get("name") for p in data.get("plugins", [])]
            check(
                pkg.get("name") in names,
                f"registry/plugins.json contains {pkg.get('name')}",
                hard=False,
            )
        else:
            results.append(("warn", "registry/plugins.json missing in repo root"))

    n_fail = sum(1 for status, _ in results if status == "fail")
    n_warn = sum(1 for status, _ in results if status == "warn")
    for status, msg in results:
        icon = {"ok": "✓", "warn": "⚠", "fail": "✗"}[status]
        print(f"  {icon} {msg}")
    print(f"\n{len(results)} checks: {len(results) - n_fail - n_warn} ok, {n_warn} warn, {n_fail} fail")
    return 1 if n_fail else 0


def cmd_list(args: argparse.Namespace) -> int:
    """List installed plugins in a profile."""
    profile_dir = resolve_profile_dir(args.profile)
    packages = read_profile_packages(profile_dir)

    if not packages:
        print(f"No plugins found in profile '{args.profile}'.")
        return 0

    print(f"Plugins in profile '{args.profile}':\n")
    for name, data in sorted(packages.items()):
        dsh = data.get("dsh", {})
        bundle = dsh.get("bundle", {})
        is_plugin = bundle.get("patch") is not None
        status = "plugin" if is_plugin else "dependency"
        version = data.get("version", "?")
        desc = data.get("description", "")
        print(f"  {name}@{version}  [{status}]")
        if desc:
            print(f"    {desc[:80]}")

        if is_plugin:
            pkg_dir = profile_dir / "node_modules" / name
            meta = load_cordis_patch(pkg_dir)
            if meta:
                skills = meta.get("skills", "")
                if skills:
                    print(f"    skills: {skills[:60]}")

    return 0


def _fetch_registry() -> list[dict[str, Any]] | None:
    """Fetch the community plugin registry from GitHub."""
    if urlopen is None:
        return None
    try:
        req = Request(REGISTRY_URL, headers={"User-Agent": "dsh-plugin-cli/1.0"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("plugins", [])
    except (URLError, json.JSONDecodeError, OSError):
        return None


def cmd_search(args: argparse.Namespace) -> int:
    """Search the community registry for plugins."""
    plugins = _fetch_registry()
    if plugins is None:
        print("Could not fetch registry. Try again later.")
        return 1

    query = args.query.lower()
    results = [
        p for p in plugins
        if query in p.get("name", "").lower()
        or query in p.get("description", "").lower()
        or any(query in kw.lower() for kw in p.get("keywords", []))
    ]

    if not results:
        print(f"No plugins found matching '{args.query}'.")
        return 0

    print(f"Found {len(results)} plugin(s):\n")
    for p in results:
        print(f"  {p['name']}@{p.get('version', '?')}")
        print(f"    {p.get('description', '')[:80]}")
        print(f"    npm: {p.get('npm', '?')}  |  source: {p.get('source', '?')}")
        skills = p.get("skills", [])
        if skills:
            skill_names = [s.get("id", "?") for s in skills]
            print(f"    skills: {', '.join(skill_names)}")
        print()

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed info about an installed plugin."""
    profile_dir = resolve_profile_dir(args.profile)
    packages = read_profile_packages(profile_dir)

    name = args.plugin
    if name not in packages:
        print(f"Plugin '{name}' not found in profile '{args.profile}'.")
        print("Run 'dsh-plugin list --profile <name>' to see installed plugins.")
        return 1

    data = packages[name]
    pkg_dir = profile_dir / "node_modules" / name
    meta = load_cordis_patch(pkg_dir)

    print(f"Name:        {name}")
    print(f"Version:     {data.get('version', '?')}")
    print(f"Description: {data.get('description', 'N/A')}")
    print(f"Author:      {data.get('author', 'N/A')}")
    print(f"License:     {data.get('license', 'N/A')}")
    print(f"Homepage:    {data.get('homepage', 'N/A')}")

    dsh = data.get("dsh", {})
    engines = dsh.get("engines", {})
    print(f"DSH engine:  {engines.get('dsh', 'N/A')}")

    client = dsh.get("client", {})
    inject = client.get("inject", [])
    if inject:
        print(f"Injects:     {', '.join(inject)}")

    if meta:
        skills = meta.get("skills", "")
        if skills:
            print(f"Skills:      {skills}")

    # Check build artifacts
    dist_ok = (pkg_dir / "dist" / "index.mjs").is_file()
    dist_client_ok = (pkg_dir / "dist-client" / "client" / "index.js").is_file()
    print(f"\nBuild artifacts:")
    print(f"  dist/index.mjs:       {'✓' if dist_ok else '✗ missing'}")
    print(f"  dist-client/client/index.js: {'✓' if dist_client_ok else '✗ missing'}")

    return 0


def cmd_add(args: argparse.Namespace) -> int:
    """Install a plugin — delegates to pnpm via dsh plugin."""
    profile = args.profile
    source = args.source

    # Build pnpm args
    pnpm_args = ["add", source]
    if args.save_dev:
        pnpm_args.insert(1, "-D")
    if args.save_exact:
        pnpm_args.insert(1, "-E")

    print(f"Installing {source} into profile '{profile}'...")
    result = subprocess.run(
        ["dsh", "plugin", "--profile", profile] + pnpm_args,
        shell=(os.name == "nt"),
    )
    return result.returncode


def cmd_remove(args: argparse.Namespace) -> int:
    """Uninstall a plugin — delegates to pnpm via dsh plugin."""
    profile = args.profile
    name = args.plugin

    print(f"Removing {name} from profile '{profile}'...")
    result = subprocess.run(
        ["dsh", "plugin", "--profile", profile, "remove", name],
        shell=(os.name == "nt"),
    )
    return result.returncode


# ---------------------------------------------------------------------------
# scaffold — generate a complete, standards-compliant DSH plugin
# ---------------------------------------------------------------------------

# Templates use __SENTINEL__ placeholders substituted via str.replace()
# (str.format is unsafe here: TS/JSX templates contain literal braces).

_TEMPLATE_PACKAGE_JSON = """{
  "name": "__NAME__",
  "description": "__DESCRIPTION__",
  "version": "0.1.0",
  "type": "module",
  "packageManager": "pnpm@11.22.0",
  "engines": {
    "node": "^22.19.0 || >=24.0.0"
  },
  "scripts": {
    "build": "tsc --noEmit && tsdown",
    "typecheck": "tsc --noEmit",
    "test": "vitest run",
    "test:watch": "vitest",
    "prepare": "pnpm build"
  },
  "main": "dist/index.mjs",
  "types": "dist/index.d.mts",
  "exports": {
    ".": {
      "types": "./dist/index.d.mts",
      "default": "./dist/index.mjs"
    },
    "./client": {
      "types": "./dist-client/client/index.d.ts",
      "default": "./dist-client/client/index.js"
    },
    "./package.json": "./package.json"
  },
  "files": [
    "dist/**/*.js",
    "dist/**/*.js.map",
    "dist/**/*.d.ts",
    "dist/**/*.d.ts.map",
    "dist/**/*.d.mts",
    "dist/**/*.d.mts.map",
    "dist-client/**/*.js",
    "dist-client/**/*.js.map",
    "dist-client/**/*.d.ts",
    "dist-client/**/*.d.ts.map",
    "src",
    "cordis.patch.yml"
  ],
  "license": "__LICENSE__",
  "author": "__AUTHOR__",
  "dsh": {
    "engines": {
      "dsh": ">=0.1.1-rc.1"
    },
    "bundle": {
      "patch": "./cordis.patch.yml"
    },
    "client": {
      "inject": [
        "@deepseek-ai/dsh-client-runtime",
        "@deepseek-ai/dsh-client-connection",
        "@deepseek-ai/dsh-client-locale",
        "@deepseek-ai/dsh-client-ui-settings",
        "@deepseek-ai/dsh-client-ui-slots"
      ],
      "platform": "__PLATFORM__"
    }
  },
  "peerDependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "~18.2.49",
    "@types/react-dom": "~18.2.22",
    "jsdom": "^26.0.0",
    "tsdown": "^0.22.2",
    "typescript": "^6.0.3",
    "vitest": "^4.1.8"
  },
  "repository": {
    "type": "git",
    "url": "__REPOSITORY_URL__",
    "directory": "__PLUGIN_REL_DIR__"
  }
}
"""

_TEMPLATE_PATCH_YML = """# __PLUGIN_TITLE__ plugin for DSH
# Mounts the browser-side panel into DSH slots and registers settings.

- insert:
    - id: __PLUGIN_ID__
      name: '__NAME__'
  meta:
    version: "0.1.0"
    description: "__DESCRIPTION__"
    author: "__AUTHOR__"
    license: "__LICENSE__"
    homepage: "__HOMEPAGE__"
    keywords: [__KEYWORDS__]
    skills:
      - id: __PLUGIN_ID__
        name: "__PLUGIN_TITLE__"
        description: "__DESCRIPTION__"
        whenToUse: >
          Use when the user asks about __PLUGIN_ID__ functionality.
          Opens the __PLUGIN_TITLE__ panel in the current session.
"""

_TEMPLATE_TSCONFIG = """{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "jsx": "react-jsx",
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src", "tests"]
}
"""

_TEMPLATE_TSDOWN = """/**
 * Minimal two-config tsdown setup for DSH plugin.
 *
 * Target A (host): ESM for Node/cordis -> dist/
 * Target B (client): CJS loader-factory for DSH browser -> dist-client/
 */

import { defineConfig } from 'tsdown'

const CLIENT_ID = '__NAME__'

export default defineConfig([
  // Host half: plain ESM
  {
    entry: { index: './src/index.ts' },
    outDir: 'dist',
    format: 'esm',
    platform: 'node',
    target: 'node22',
    dts: true,
    sourcemap: true,
  },

  // Client half: CJS + __ModuleLoader__.load wrapper
  {
    entry: { 'client/index': './src/client/index.ts' },
    outDir: 'dist-client',
    format: 'cjs',
    platform: 'browser',
    target: 'es2022',
    sourcemap: true,
    external: ['react', 'react/jsx-runtime', 'react-dom'],
    outputOptions: {
      // Force `.js` (not `.cjs`): package.json `type: module` would otherwise
      // make tsdown emit `index.cjs`, but DSH's loader resolves the bundle via
      // the `./client` export which points at `dist-client/client/index.js`.
      entryFileNames: '[name].js',
      banner: [
        `window.__ModuleLoader__.load({`,
        `  id: ${JSON.stringify(CLIENT_ID)},`,
        `  factory: (require) => {`,
        `    var module = { exports: {} };`,
        `    var exports = module.exports;`,
      ].join('\\n'),
      footer: '    return module.exports;\\n  }\\n});',
    },
  },
])
"""

_TEMPLATE_VITEST = """import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
  },
})
"""

_TEMPLATE_HOST_INDEX = """/**
 * DSH host-side entry for __PLUGIN_TITLE__.
 *
 * The host half is a cordis no-op: the DSH loader reads package.json
 * `dsh.bundle.patch` + the `./client` export and mounts the browser half.
 */

export const name = '__PLUGIN_ID__'

export function apply(): void {
  // Host-side lifecycle (optional). Most plugins are client-only.
}
"""

_TEMPLATE_CLIENT_INDEX = """/**
 * DSH browser-side entry for __PLUGIN_TITLE__.
 *
 * Registers i18n, settings, slots and the plugin skill on the DSH fiber ctx.
 * Every service touched here must be listed in package.json `dsh.client.inject`.
 */

import type { Context } from '@deepseek-ai/dsh-client-runtime'
import { __PLUGIN_CAMEL__ } from './panel/__PLUGIN_CAMEL__'
import { SettingsCard, DEFAULT_SETTINGS } from './settings/SettingsCard'
import { zh, en } from './locales'

export const inject = ['slots', 'locale', 'settingsScope']

export function apply(ctx: Context): void {
  const ns = '__PLUGIN_ID__'

  // ---------- i18n ----------
  ctx.locale.register(ns, { zh, en })

  const t = (key: string): string => ctx.locale.t(`${ns}:${key}`)

  // ---------- settings ----------
  const settingsBinder = (ctx.settingsScope ?? ctx) as {
    bind<S>(spec: { default: S; namespace: string }): { getSnapshot(): S; setValue(partial: Partial<S>): void }
  }

  let settingsScope: { getSnapshot(): typeof DEFAULT_SETTINGS; setValue(partial: Partial<typeof DEFAULT_SETTINGS>): void } | null = null
  try {
    settingsScope = settingsBinder.bind({ default: DEFAULT_SETTINGS, namespace: ns })
  } catch {
    // Settings scope not available in this runtime version.
  }

  const snapshot = (): typeof DEFAULT_SETTINGS => settingsScope?.getSnapshot() ?? { ...DEFAULT_SETTINGS }

  // ---------- inject slots ----------
  try {
    ctx.slots.inject('conversation.session.header.actions', () =>
      ctx.slots.register({
        name: 'conversation.session.header.actions',
        id: '__PLUGIN_ID__',
        order: 10,
        locale: ns,
      }, __PLUGIN_CAMEL__))
    console.log('[__PLUGIN_ID__] slot registration succeeded')
  } catch (e) {
    console.error('[__PLUGIN_ID__] slot registration failed:', e)
  }

  try {
    ctx.slots.inject('settings.plugin.item', () =>
      ctx.slots.register({
        name: 'settings.plugin.item',
        key: '__PLUGIN_ID__',
        locale: ns,
        inject: (): Record<string, unknown> => ({
          controller: { getSnapshot: snapshot, setValue: (v: Partial<typeof DEFAULT_SETTINGS>) => settingsScope?.setValue(v) },
          t,
        }),
      }, SettingsCard))
    console.log('[__PLUGIN_ID__] settings slot registration succeeded')
  } catch (e) {
    console.error('[__PLUGIN_ID__] settings slot registration failed:', e)
  }

  // ---------- skill registration ----------
  try {
    const skills = (ctx as unknown as { skills?: { register(id: string, def: Record<string, unknown>): void } }).skills
    skills?.register('__PLUGIN_ID__', {
      id: '__PLUGIN_ID__',
      name: '__PLUGIN_TITLE__',
      description: '__DESCRIPTION__',
      whenToUse: 'Use when the user asks about __PLUGIN_ID__ functionality',
      modelInvocable: true,
      userInvocable: true,
      invoke: () => {
        const btn = document.querySelector(
          '[data-slot-id="__PLUGIN_ID__"]'
        )
        ;(btn as HTMLButtonElement | null)?.click()
      },
    })
    console.log('[__PLUGIN_ID__] skill registration succeeded')
  } catch (e) {
    console.error('[__PLUGIN_ID__] skill registration failed:', e)
  }
}
"""

_TEMPLATE_LOCALES = """/** i18n dictionaries for __PLUGIN_TITLE__ (namespace __PLUGIN_ID__). */

const zhDict = {
  'panel.title': '__PLUGIN_TITLE__',
  'panel.empty': '暂无数据',
  'settings.label': '__PLUGIN_TITLE__ 设置',
} as const

const enDict: { [K in keyof typeof zhDict]: string } = {
  'panel.title': '__PLUGIN_TITLE__',
  'panel.empty': 'No data yet',
  'settings.label': '__PLUGIN_TITLE__ settings',
}

export const zh = zhDict
export const en = enDict
"""

_TEMPLATE_SLOTS_AUGMENT = """/**
 * Module augmentation for DSH SDK slot and locale declarations.
 *
 * Augments the DSH SlotMap so that ctx.slots.inject() / ctx.slots.register()
 * type-check for our custom seats.
 */

declare module '@deepseek-ai/dsh-client-ui-slots' {
  interface SlotMap {
    /**
     * Session-header actions list. Each registered entry renders as a
     * button in the conversation session header bar.
     */
    'conversation.session.header.actions': {
      kind: 'list'
      scope: 'session'
      owner: never
    }

    /**
     * Plugin item card in the settings -> plugins tab.
     * Keyed by the plugin's settings namespace.
     */
    'settings.plugin.item': {
      kind: 'keyed'
      scope: 'root'
      owner: never
    }
  }

  interface LocaleNamespaceMap {
    /** __PLUGIN_TITLE__ UI strings. */
    '__PLUGIN_ID__': {
      'panel.title': string
      'panel.empty': string
      'settings.label': string
    }
  }
}
"""

_TEMPLATE_PANEL = """import { createElement } from 'react'

export interface __PLUGIN_CAMEL__Props {
  /** Framework-wired translator (locale: '__PLUGIN_ID__'). */
  t: (key: string) => string
}

export function __PLUGIN_CAMEL__({ t }: __PLUGIN_CAMEL__Props) {
  return createElement(
    'div',
    { 'data-plugin': '__PLUGIN_ID__', style: { padding: 12 } },
    createElement('h3', null, t('panel.title')),
    createElement('p', null, t('panel.empty')),
  )
}
"""

_TEMPLATE_SETTINGS = """import { createElement } from 'react'

export interface TgSettings {
  serverUrl: string
  authToken: string
}

export const DEFAULT_SETTINGS: TgSettings = {
  serverUrl: 'http://127.0.0.1:8765',
  authToken: '',
}

export interface SettingsCardProps {
  controller: {
    getSnapshot(): TgSettings
    setValue(partial: Partial<TgSettings>): void
  }
  t: (key: string) => string
}

export function SettingsCard({ controller, t }: SettingsCardProps) {
  const { serverUrl, authToken } = controller.getSnapshot()

  return createElement(
    'div',
    { 'data-plugin-settings': '__PLUGIN_ID__' },
    createElement('h4', null, t('settings.label')),
    createElement(
      'label',
      null,
      t('settings.label'),
      createElement('input', {
        type: 'text',
        value: serverUrl,
        onChange: (e: Event) =>
          controller.setValue({ serverUrl: (e.target as HTMLInputElement).value }),
      }),
    ),
    createElement('input', {
      type: 'password',
      placeholder: 'Auth Token',
      value: authToken,
      onChange: (e: Event) =>
        controller.setValue({ authToken: (e.target as HTMLInputElement).value }),
    }),
  )
}
"""

_TEMPLATE_SMOKE_TEST = """import { describe, expect, it } from 'vitest'
import { apply } from '../src/client/index'

describe('client apply()', () => {
  it('applies to a mock DSH ctx without throwing', () => {
    const slots = {
      inject: (_name: string, factory: () => unknown) => factory(),
      register: (_opts: unknown, component: unknown) => component,
    }
    const ctx = {
      locale: {
        register: (_ns: string, dict: unknown) => dict,
        t: (key: string) => key,
      },
      settingsScope: {
        bind: (_spec: unknown) => ({
          getSnapshot: () => ({ serverUrl: '', authToken: '' }),
          setValue: () => {},
        }),
      },
      slots,
      skills: { register: (_id: string, def: unknown) => def },
    }
    expect(() => apply(ctx as never)).not.toThrow()
  })
})
"""

_TEMPLATE_README = """# __PLUGIN_TITLE__

__DESCRIPTION__

## Install

```bash
dsh plugin --profile web add __NAME__
# or
dsh-plugin add __NAME__ --profile web
```

## Usage

Open a DSH session and click the **__PLUGIN_TITLE__** button in the session
header. Configure the plugin under Settings → Plugins.

## Development

```bash
pnpm install
pnpm build
pnpm test
```

Release: `dsh-plugin release --bump patch --push` — the CI workflow publishes
to npm, updates `registry/plugins.json` and creates a GitHub release.

See docs/PLUGIN_STANDARDS.md (canonical spec) and `.claude/skills/dsh-plugin/`.
"""

_TEMPLATE_README_ZH = """# __PLUGIN_TITLE__

__DESCRIPTION__

## 安装

```bash
dsh plugin --profile web add __NAME__
# 或
dsh-plugin add __NAME__ --profile web
```

## 使用

打开 DSH 会话，点击会话头部的 **__PLUGIN_TITLE__** 按钮。
在「设置 → 插件」中配置本插件。

## 开发

```bash
pnpm install
pnpm build
pnpm test
```

发布：`dsh-plugin release --bump patch --push` — CI 会自动发布到 npm、
更新 `registry/plugins.json` 并创建 GitHub Release。

规范见 docs/PLUGIN_STANDARDS.md，开发流程见 `.claude/skills/dsh-plugin/`。
"""

_TEMPLATE_GITIGNORE = """node_modules/
dist/
dist-client/
lib/
*.tsbuildinfo
"""

_TEMPLATE_WORKFLOW = """name: Publish Plugin

on:
  push:
    tags: ['v*']
  workflow_dispatch:

permissions:
  contents: write
  id-token: write

env:
  PLUGIN_DIR: __PLUGIN_REL_DIR__

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: pnpm/action-setup@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: pnpm

      - name: Install dependencies
        run: |
          cd "$PLUGIN_DIR"
          pnpm install

      - name: Validate against PLUGIN_STANDARDS
        run: python3 scripts/dsh_plugin_cli.py validate --plugin-dir "$PLUGIN_DIR"

  test:
    runs-on: ubuntu-latest
    needs: validate
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: pnpm
      - run: |
          cd "$PLUGIN_DIR"
          pnpm install
      - run: |
          cd "$PLUGIN_DIR"
          pnpm typecheck
      - run: |
          cd "$PLUGIN_DIR"
          pnpm test

  build:
    runs-on: ubuntu-latest
    needs: validate
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: pnpm
      - run: |
          cd "$PLUGIN_DIR"
          pnpm install
      - run: |
          cd "$PLUGIN_DIR"
          pnpm build
      - name: Verify build artifacts
        run: |
          cd "$PLUGIN_DIR"
          test -f dist/index.mjs || { echo "ERROR: dist/index.mjs missing"; exit 1; }
          test -f dist-client/client/index.js || { echo "ERROR: dist-client/client/index.js missing"; exit 1; }
          grep -q "__ModuleLoader__.load" dist-client/client/index.js \\
            || { echo "ERROR: client bundle missing __ModuleLoader__ wrapper"; exit 1; }

  publish-npm:
    runs-on: ubuntu-latest
    needs: [validate, test, build]
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: pnpm
          registry-url: 'https://registry.npmjs.org'
      - run: |
          cd "$PLUGIN_DIR"
          pnpm install
      - name: Tag/version consistency check
        run: |
          cd "$PLUGIN_DIR"
          TAG_VERSION="${GITHUB_REF#refs/tags/v}"
          PKG_VERSION=$(node -p "require('./package.json').version")
          if [ "$TAG_VERSION" != "$PKG_VERSION" ]; then
            echo "ERROR: tag v$TAG_VERSION != package.json $PKG_VERSION"
            exit 1
          fi
      - run: |
          cd "$PLUGIN_DIR"
          pnpm build
      - name: Publish to npm
        run: |
          cd "$PLUGIN_DIR"
          npm publish --access public --provenance
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}

  update-registry:
    runs-on: ubuntu-latest
    needs: publish-npm
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
      - uses: actions/checkout@v4

      - name: Update registry
        run: |
          python3 -m venv /tmp/venv
          /tmp/venv/bin/pip install --quiet pyyaml
          /tmp/venv/bin/python scripts/update_registry.py \\
            --plugin-dir "$PLUGIN_DIR" \\
            --registry registry/plugins.json \\
            --registry-url "https://raw.githubusercontent.com/${{ github.repository }}/${{ github.event.repository.default_branch }}/registry/plugins.json" \\
            --source-url "${{ github.server_url }}/${{ github.repository }}/tree/${{ github.event.repository.default_branch }}/${{ env.PLUGIN_DIR }}" \\
            --write

      - name: Commit updated registry
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add registry/plugins.json
          git diff --cached --quiet || git commit -m "chore(registry): update plugin list [skip ci]"
          git push origin "HEAD:${{ github.event.repository.default_branch }}"

  github-release:
    runs-on: ubuntu-latest
    needs: publish-npm
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          generate_release_notes: true
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
"""

_TEMPLATE_REGISTRY_JSON = """{
  "version": 0,
  "updated_at": "",
  "plugins": []
}
"""


def _camel_case(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.replace("_", "-").split("-"))


def _render(template: str, values: dict[str, str]) -> str:
    text = template
    for key, val in values.items():
        text = text.replace(f"__{key}__", val)
    leftover = re.findall(r"__[A-Z_]+__", text)
    if leftover:
        raise ValueError(f"unsubstituted template placeholders: {sorted(set(leftover))}")
    return text


def _guess_repository_url(repo: Path) -> str:
    try:
        remote = _run_git(repo, "remote", "get-url", "origin").stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    remote = remote.removesuffix(".git")
    if remote.startswith("git@"):  # git@github.com:owner/repo
        remote = remote.split("@", 1)[1].replace(":", "/")
    if "://" in remote:
        remote = remote.split("://", 1)[1]
    host, _, path = remote.partition("/")
    if host in ("github.com", "gitlab.com"):
        return f"https://{host}/{path}"
    return ""


def cmd_scaffold(args: argparse.Namespace) -> int:
    """Generate a complete, standards-compliant DSH plugin skeleton."""
    name: str = args.name
    if not PLUGIN_NAME_RE.match(name):
        print(f"error: invalid package name {name!r} (expected e.g. 'my-plugin' or '@scope/my-plugin')")
        return 1
    slug = name.rsplit("/", 1)[-1]
    if not KEBAB_ID_RE.match(slug):
        print(f"error: plugin id must be kebab-case, got {slug!r}")
        return 1

    title = " ".join(part.capitalize() for part in slug.replace("_", "-").split("-"))
    camel = _camel_case(slug)
    keywords = [k.strip() for k in (args.keywords or "dsh-plugin").split(",") if k.strip()]
    repo = find_repo_root()
    target = Path(args.dir).resolve() if args.dir else None

    if target is None:
        root = repo or Path.cwd()
        plugin_dir = root / "dsh-plugin" / slug
    elif repo and target.is_relative_to(repo) and target != repo:
        # Inside an existing repo: --dir names the plugin directory itself.
        plugin_dir, root = target, repo
    else:
        # Fresh repo layout at --dir (also covers --dir <repo-root>).
        plugin_dir, root = target / "dsh-plugin" / slug, target

    if plugin_dir.exists() and any(plugin_dir.iterdir()):
        print(f"error: {plugin_dir} already exists and is not empty")
        return 1

    rel_dir = plugin_dir.relative_to(root).as_posix() if plugin_dir.is_relative_to(root) else "."
    repository_url = _guess_repository_url(root) if (root / ".git").exists() else args.homepage
    values = {
        "NAME": name,
        "PLUGIN_ID": slug,
        "PLUGIN_TITLE": title,
        "PLUGIN_CAMEL": camel + "Panel",
        "DESCRIPTION": args.description or f"{title} panel for DSH",
        "AUTHOR": args.author or "",
        "LICENSE": args.license,
        "PLATFORM": args.platform,
        "KEYWORDS": ", ".join(json.dumps(k) for k in keywords),
        "HOMEPAGE": args.homepage or repository_url or "",
        "REPOSITORY_URL": repository_url or "https://github.com/your-org/your-repo.git",
        "PLUGIN_REL_DIR": rel_dir,
    }

    files: dict[str, str] = {
        "package.json": _TEMPLATE_PACKAGE_JSON,
        "cordis.patch.yml": _TEMPLATE_PATCH_YML,
        "tsconfig.json": _TEMPLATE_TSCONFIG,
        "tsdown.config.ts": _TEMPLATE_TSDOWN,
        "vitest.config.ts": _TEMPLATE_VITEST,
        "src/index.ts": _TEMPLATE_HOST_INDEX,
        "src/client/index.ts": _TEMPLATE_CLIENT_INDEX,
        "src/client/locales.ts": _TEMPLATE_LOCALES,
        "src/client/slots-augment.ts": _TEMPLATE_SLOTS_AUGMENT,
        f"src/client/panel/{camel}Panel.tsx": _TEMPLATE_PANEL,
        "src/client/settings/SettingsCard.tsx": _TEMPLATE_SETTINGS,
        "tests/smoke.test.ts": _TEMPLATE_SMOKE_TEST,
        "README.md": _TEMPLATE_README,
        "README.zh.md": _TEMPLATE_README_ZH,
        ".gitignore": _TEMPLATE_GITIGNORE,
    }

    try:
        for rel, template in files.items():
            path = plugin_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_render(template, values), encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: scaffold failed: {exc}")
        return 1

    # Repo-level publish workflow — create only when absent, never clobber.
    workflow_dst = root / ".github" / "workflows" / "plugin-publish.yml"
    if not workflow_dst.is_file():
        try:
            workflow_dst.parent.mkdir(parents=True, exist_ok=True)
            workflow_dst.write_text(_render(_TEMPLATE_WORKFLOW, values), encoding="utf-8")
        except (OSError, ValueError) as exc:
            print(f"warning: workflow generation failed: {exc}")

    # Repo-level tooling: copy the CLI + registry updater so CI and local
    # workflows work in a fresh repo too.
    scripts_src = Path(__file__).resolve().parent
    scripts_dst = root / "scripts"
    try:
        scripts_dst.mkdir(parents=True, exist_ok=True)
        for tool in ("dsh_plugin_cli.py", "update_registry.py", "__init__.py"):
            src, dst = scripts_src / tool, scripts_dst / tool
            if src.is_file() and not dst.exists():
                shutil.copy2(src, dst)
    except OSError as exc:
        print(f"warning: could not copy tooling scripts: {exc}")

    # Registry bootstrap: create an empty registry, then upsert this plugin.
    registry = root / "registry" / "plugins.json"
    try:
        from scripts.update_registry import update_registry  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(scripts_src))
        from update_registry import update_registry  # noqa: F401
    if not registry.is_file():
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(_TEMPLATE_REGISTRY_JSON, encoding="utf-8")
    try:
        update_registry(plugin_dir, registry, write=True, source_url=args.homepage or None)
    except Exception as exc:  # noqa: BLE001 — registry is best-effort during scaffold
        print(f"warning: registry update skipped: {exc}")

    print(f"\nScaffolded plugin '{name}' at {plugin_dir}")
    print("Next steps:")
    print(f"  1. cd {plugin_dir} && pnpm install")
    print("  2. pnpm build && pnpm test")
    print(f"  3. python {root / 'scripts' / 'dsh_plugin_cli.py'} validate --plugin-dir {plugin_dir}")
    print("  4. dsh-plugin release --bump patch --push   # CI publishes npm + registry + release")
    return 0


# ---------------------------------------------------------------------------
# release — bump version, sync patch meta, commit and tag (CI does the rest)
# ---------------------------------------------------------------------------


def cmd_release(args: argparse.Namespace) -> int:
    pkg_dir = resolve_plugin_dir(args)
    if not pkg_dir.is_dir():
        print(f"error: plugin dir not found: {pkg_dir}")
        return 1

    pkg, meta = _load_meta(pkg_dir)
    old = str(pkg.get("version", ""))
    try:
        new = args.version or _bump_version(old, args.bump or "patch")
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    if new == old:
        print(f"error: --version must differ from current version ({old})")
        return 1

    repo = find_repo_root(pkg_dir)

    if repo and not args.no_git and not args.force:
        status = _run_git(repo, "status", "--porcelain").stdout.strip()
        if status:
            print("error: working tree is dirty — commit or stash first (or use --force)")
            print(status)
            return 1

    manifest = pkg_dir / "package.json"
    patch_path = pkg_dir / "cordis.patch.yml"
    pkg["version"] = new
    manifest.write_text(json.dumps(pkg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        _update_patch_version(patch_path, old, new)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(f"{pkg.get('name')}: {old} -> {new}")

    # Sync the local registry entry so version metadata stays consistent.
    if repo and (repo / "registry" / "plugins.json").is_file():
        try:
            from scripts.update_registry import update_registry
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from update_registry import update_registry
        try:
            update_registry(pkg_dir, repo / "registry" / "plugins.json", write=True)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: local registry sync skipped: {exc}")

    if repo and not args.no_git:
        rel = pkg_dir.relative_to(repo).as_posix()
        try:
            _run_git(repo, "add", f"{rel}/package.json", f"{rel}/cordis.patch.yml", "registry/plugins.json")
            _run_git(repo, "commit", "-m", f"chore(release): {pkg.get('name')} v{new}")
            _run_git(repo, "tag", f"v{new}")
        except subprocess.CalledProcessError as exc:
            print(f"error: git failed: {exc.stderr.strip() or exc}")
            return 1
        if args.push:
            try:
                _run_git(repo, "push", "origin", "HEAD")
                _run_git(repo, "push", "origin", f"v{new}")
            except subprocess.CalledProcessError as exc:
                print(f"error: push failed: {exc.stderr.strip() or exc}")
                return 1
            print(f"\nPushed tag v{new} — CI will publish npm, update the registry and create a release.")
        else:
            print(f"\nCommitted + tagged v{new}. Run `git push origin HEAD --tags` (or --push) to trigger CI.")

    return 0


# ---------------------------------------------------------------------------
# registry update — upsert a plugin entry into registry/plugins.json
# ---------------------------------------------------------------------------


def cmd_registry(args: argparse.Namespace) -> int:
    if args.registry_command != "update":
        print("error: unknown registry subcommand (try 'registry update')")
        return 1

    pkg_dir = resolve_plugin_dir(args)
    repo = find_repo_root(pkg_dir)
    registry = Path(args.registry) if args.registry else (
        repo / "registry" / "plugins.json" if repo else Path(DEFAULT_REGISTRY)
    )

    try:
        from scripts.update_registry import update_registry
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from update_registry import update_registry

    try:
        return update_registry(
            pkg_dir,
            registry,
            write=args.write,
            source_url=args.source_url,
            registry_url=args.registry_url,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 1


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to GBK/cp936 — allow ✓/⚠/✗ output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="dsh-plugin",
        description="DSH Community Plugin CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="List installed plugins")
    p_list.add_argument("--profile", required=True, help="DSH profile name (web/tui/headless)")

    # search
    p_search = sub.add_parser("search", help="Search community registry")
    p_search.add_argument("query", help="Search query")

    # info
    p_info = sub.add_parser("info", help="Show plugin details")
    p_info.add_argument("plugin", help="Plugin package name")
    p_info.add_argument("--profile", required=True, help="DSH profile name")

    # add
    p_add = sub.add_parser("add", help="Install a plugin")
    p_add.add_argument("source", help="npm package, git URL, or local path")
    p_add.add_argument("--profile", required=True, help="DSH profile name")
    p_add.add_argument("-D", "--save-dev", action="store_true")
    p_add.add_argument("-E", "--save-exact", action="store_true")

    # remove
    p_remove = sub.add_parser("remove", help="Uninstall a plugin")
    p_remove.add_argument("plugin", help="Plugin package name")
    p_remove.add_argument("--profile", required=True, help="DSH profile name")

    # validate
    p_validate = sub.add_parser("validate", help="Validate a plugin against PLUGIN_STANDARDS")
    p_validate.add_argument("--plugin-dir", help="Plugin directory (default: repo reference plugin)")

    # scaffold
    p_scaffold = sub.add_parser("scaffold", help="Scaffold a new DSH community plugin")
    p_scaffold.add_argument("name", help="npm package name (e.g. 'my-plugin' or '@scope/my-plugin')")
    p_scaffold.add_argument("--dir", help="Plugin directory (inside a repo) or new repo root")
    p_scaffold.add_argument("--author", default="", help="meta.author value")
    p_scaffold.add_argument("--description", default="", help="meta.description value")
    p_scaffold.add_argument("--homepage", default="", help="meta.homepage value")
    p_scaffold.add_argument("--license", default="Apache-2.0", help="SPDX license id")
    p_scaffold.add_argument("--platform", choices=["web", "tui"], default="web")
    p_scaffold.add_argument("--keywords", default="dsh-plugin", help="comma-separated keywords")

    # release
    p_release = sub.add_parser("release", help="Bump version, commit and tag (CI publishes)")
    p_release.add_argument("--plugin-dir", help="Plugin directory (default: repo reference plugin)")
    p_release.add_argument("--bump", choices=["patch", "minor", "major"], default="patch")
    p_release.add_argument("--version", help="Explicit version (overrides --bump)")
    p_release.add_argument("--push", action="store_true", help="Push commit + tag to origin")
    p_release.add_argument("--force", action="store_true", help="Skip the dirty-tree check")
    p_release.add_argument("--no-git", action="store_true", help="Only bump files; no commit/tag")

    # registry update
    p_registry = sub.add_parser("registry", help="Community registry maintenance")
    p_reg_sub = p_registry.add_subparsers(dest="registry_command", required=True)
    p_reg_update = p_reg_sub.add_parser("update", help="Upsert a plugin entry into registry/plugins.json")
    p_reg_update.add_argument("--plugin-dir", help="Plugin directory (default: repo reference plugin)")
    p_reg_update.add_argument("--registry", help="Path to registry/plugins.json")
    p_reg_update.add_argument("--write", action="store_true", help="Write the registry (default: dry run)")
    p_reg_update.add_argument("--source-url", help="Override the entry's source URL")
    p_reg_update.add_argument("--registry-url", help="Set/update the top-level registry_url")

    args = parser.parse_args(argv)

    commands = {
        "list": cmd_list,
        "search": cmd_search,
        "info": cmd_info,
        "add": cmd_add,
        "remove": cmd_remove,
        "validate": cmd_validate,
        "scaffold": cmd_scaffold,
        "release": cmd_release,
        "registry": cmd_registry,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
