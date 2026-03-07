#!/usr/bin/env python3
"""Sync the canonical local evaluation overview into GitHub Project V2.

Default mode is dry-run. Use `--live` to write to the remote project.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = REPO_ROOT / 'scripts' / 'sync_benchmark_project_v2.py'
OVERVIEW_SCRIPT = REPO_ROOT / 'scripts' / 'render_project_eval_overview.py'


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _build_args(entry: Dict[str, Any], args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        project_org=args.project_org,
        project_number=args.project_number,
        github_token_env=args.github_token_env,
        run_id=None,
        line=entry.get('line'),
        pack=entry.get('pack'),
        mode=entry.get('mode'),
        noise=entry.get('noise'),
        dropout=entry.get('dropout'),
        seed=entry.get('seed'),
        status=entry.get('status'),
        gate_status_override=entry.get('gate_status_override'),
        item_type='Run',
        parent_epic=entry.get('parent_epic'),
        priority='P1',
        owner='MassimoQu',
        artifact_link=str(Path(entry['summary_json']).parent.relative_to(REPO_ROOT)),
        repo_link=None,
        config_path=None,
        item_title=entry.get('label'),
        config_link=None,
        status_note=entry.get('status_note'),
        next_action=entry.get('next_action'),
        curation_tier=entry.get('curation_tier'),
        result=entry.get('result'),
        repo_name='vggt_series_4_coop',
        branch='lantu_A800',
        dry_run=not args.live,
        manifest_out=None,
        summary_json=Path(entry['summary_json']),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-org', default='coopVGGT')
    ap.add_argument('--project-number', type=int, default=2)
    ap.add_argument('--github-token-env', default='GH_PROJECT_TOKEN')
    ap.add_argument('--live', action='store_true', help='Write to GitHub Project V2')
    ap.add_argument('--only', action='append', default=[], help='Restrict to run_id substring(s)')
    ap.add_argument('--manifest-dir', type=Path, default=None, help='Optional directory for emitted manifests')
    args = ap.parse_args()

    sync_mod = _load_module('syncproj_live', SYNC_SCRIPT)
    overview_mod = _load_module('overviewproj_live', OVERVIEW_SCRIPT)
    entries: List[Dict[str, Any]] = list(overview_mod.CANONICAL_RUNS)

    if args.only:
        needles = [x.lower() for x in args.only]
        entries = [e for e in entries if any(n in str(Path(e['summary_json']).parent.name).lower() for n in needles)]

    if not entries:
        raise SystemExit('No canonical runs matched --only filters')

    for entry in entries:
        summary_path = Path(entry['summary_json']).expanduser().resolve()
        doc = sync_mod._read_json(summary_path)
        entry_args = _build_args(entry, args)
        manifest = sync_mod._build_manifest(entry_args, summary_path, doc)
        if args.manifest_dir:
            args.manifest_dir.mkdir(parents=True, exist_ok=True)
            out = args.manifest_dir / f"{manifest['run_id']}.manifest.json"
            out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f"[RUN] {manifest['run_id']} live={args.live} gate={manifest.get('gate_status')} status={manifest.get('status')}")
        if args.live:
            sync_mod._sync_to_project(entry_args, manifest)


if __name__ == '__main__':
    main()
