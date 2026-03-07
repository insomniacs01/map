#!/usr/bin/env python3
"""Build a curated local unified-eval overview and optional Project manifests/sync.

The goal is to turn a small curated registry into:
- per-run manifests generated through `sync_benchmark_project_v2.py`
- one local `overview.md` page that links the canonical contracts, curated runs,
  and existing local evidence bundles
- optional Project V2 sync once the local board looks good
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = REPO_ROOT / 'scripts' / 'sync_benchmark_project_v2.py'
DEFAULT_REGISTRY = REPO_ROOT / 'configs' / 'unified_eval_project_registry.json'


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_registry_path(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    repo_candidate = (REPO_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate
    parent_candidate = (REPO_ROOT.parent / candidate).resolve()
    if parent_candidate.exists():
        return parent_candidate
    return repo_candidate


def _fmt(value: Any) -> str:
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def _metric(manifest: Dict[str, Any], section: str, key: str) -> Any:
    block = manifest.get(section) or {}
    if isinstance(block, dict):
        return block.get(key)
    return None


def _build_sync_cmd(args: argparse.Namespace, registry: Dict[str, Any], entry: Dict[str, Any], manifest_out: Path) -> List[str]:
    project = registry['project']
    fields = dict(entry.get('project_fields') or {})
    run_id = entry.get('run_id') or Path(entry['summary_json']).parent.name

    cmd: List[str] = [
        sys.executable,
        str(SYNC_SCRIPT),
        '--summary_json',
        str(_resolve_registry_path(entry['summary_json'])),
        '--manifest_out',
        str(manifest_out.resolve()),
        '--run_id',
        str(run_id),
    ]

    item_title = entry.get('item_title')
    if item_title:
        cmd += ['--item_title', str(item_title)]

    artifact_link = entry.get('artifact_link') or str(Path(entry['summary_json']).parent)
    status_note = entry.get('status_note')
    next_action = entry.get('next_action')

    literal_args = {
        'artifact_link': artifact_link,
        'status_note': status_note,
        'next_action': next_action,
        **fields,
    }
    for key, value in literal_args.items():
        if value is None:
            continue
        cmd += [f'--{key}', str(value)]

    if args.sync_project:
        cmd += [
            '--project_org', str(project['org']),
            '--project_number', str(project['number']),
            '--github_token_env', args.github_token_env,
        ]
    else:
        cmd += ['--dry_run']

    return cmd


def _write_overview(out_dir: Path, registry: Dict[str, Any], manifests: List[Dict[str, Any]], sync_mode: str) -> None:
    lines: List[str] = []
    project = registry['project']
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    lines.append('# Unified Eval Board')
    lines.append('')
    lines.append(f'- Generated: `{now}`')
    lines.append(f'- Project target: `{project["org"]}/{project["number"]}` — `{project["title"]}`')
    lines.append(f'- Sync mode: `{sync_mode}`')
    lines.append(f'- Registry: `{_rel(DEFAULT_REGISTRY)}`')
    lines.append('')
    lines.append('## Canonical Contracts')
    lines.append('')
    for key, contract in registry.get('contracts', {}).items():
        lines.append(f'### {contract.get("label", key)}')
        lines.append('')
        lines.append(f'- Key: `{key}`')
        frames_json = contract.get('frames_json')
        if frames_json:
            lines.append(f'- Frames: `{frames_json}`')
        protocol = contract.get('protocol') or {}
        if protocol:
            lines.append(f'- Protocol: `{json.dumps(protocol, ensure_ascii=False, sort_keys=True)}`')
        fairness = contract.get('fairness') or []
        if fairness:
            lines.append('- Fairness locks: `' + '; '.join(fairness) + '`')
        gates = contract.get('gates') or {}
        if gates:
            lines.append(f'- Gates: `{json.dumps(gates, ensure_ascii=False, sort_keys=True)}`')
        lines.append('')

    lines.append('## Curated Runs')
    lines.append('')
    lines.append('| Slug | Title | Gate | Scale | CrossTrans | DepthRel | Det AP | Repo@Commit | Summary | Manifest |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---|---|---|')
    for item in manifests:
        manifest = item['manifest']
        lines.append(
            '| {slug} | {title} | {gate} | {scale} | {cross} | {depth} | {det} | `{repo_commit}` | `{summary}` | `{manifest_path}` |'.format(
                slug=item['slug'],
                title=item['item_title'],
                gate=_fmt(manifest.get('gate_status')),
                scale=_fmt(_metric(manifest, 'metrics_geometry_core', 'scale_to_gt_mult_err_mean')),
                cross=_fmt(_metric(manifest, 'metrics_geometry_core', 'cross_agent_pose_trans_mean')),
                depth=_fmt(_metric(manifest, 'metrics_geometry_core', 'depth_rel_mean')),
                det=_fmt(_metric(manifest, 'metrics_detection', 'det_ap_iou')),
                repo_commit=_fmt(manifest.get('repo@commit')),
                summary=item['summary_rel'],
                manifest_path=item['manifest_rel'],
            )
        )
    lines.append('')

    lines.append('## Existing Evidence Bundles')
    lines.append('')
    for link in registry.get('overview_links', []):
        lines.append(f'- {link["label"]}: `{link["path"]}`')
    lines.append('')

    lines.append('## Next Review Questions')
    lines.append('')
    lines.append('- Which curated runs are clean enough to sync to Project immediately?')
    lines.append('- Which local HTML / compare pages should be linked from the first Project items?')
    lines.append('- Does any missing narrative still require a new protocol-locked rerun?')
    lines.append('')

    (out_dir / 'overview.md').write_text('\n'.join(lines), encoding='utf-8')


def _load_existing_rows(out_dir: Path, registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    manifests_dir = out_dir / 'manifests'
    logs_dir = out_dir / 'logs'
    rows: Dict[str, Dict[str, Any]] = {}
    for entry in registry.get('curated_runs', []):
        slug = entry['slug']
        manifest_out = manifests_dir / f'{slug}.json'
        if not manifest_out.exists():
            continue
        log_out = logs_dir / f'{slug}.log'
        rows[slug] = {
            'slug': slug,
            'item_title': entry.get('item_title') or slug,
            'manifest': _read_json(manifest_out),
            'summary_rel': entry['summary_json'],
            'manifest_rel': _rel(manifest_out),
            'log_rel': _rel(log_out),
        }
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description='Build a curated unified-eval board from the registry.')
    ap.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument('--out_dir', type=Path, default=None)
    ap.add_argument('--only', nargs='*', default=None, help='Optional curated run slugs to include')
    ap.add_argument('--sync-project', action='store_true', help='Write curated runs into GitHub Project V2')
    ap.add_argument('--github-token-env', default='GH_PROJECT_TOKEN')
    args = ap.parse_args()

    registry = _read_json(args.registry.expanduser().resolve())
    stamp = dt.datetime.now().strftime('%Y%m%d')
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else (REPO_ROOT / 'eval_runs' / f'unified_eval_board_{stamp}')
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = out_dir / 'manifests'
    logs_dir = out_dir / 'logs'
    manifests_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    wanted = set(args.only or [])
    sync_mode = 'project-sync' if args.sync_project else 'dry-run'
    row_map: Dict[str, Dict[str, Any]] = _load_existing_rows(out_dir, registry) if wanted else {}

    for entry in registry.get('curated_runs', []):
        slug = entry['slug']
        if wanted and slug not in wanted:
            continue

        manifest_out = manifests_dir / f'{slug}.json'
        log_out = logs_dir / f'{slug}.log'
        cmd = _build_sync_cmd(args, registry, entry, manifest_out)
        with log_out.open('w', encoding='utf-8') as fh:
            subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, stdout=fh, stderr=subprocess.STDOUT)

        manifest = _read_json(manifest_out)
        row_map[slug] = {
            'slug': slug,
            'item_title': entry.get('item_title') or slug,
            'manifest': manifest,
            'summary_rel': entry['summary_json'],
            'manifest_rel': _rel(manifest_out),
            'log_rel': _rel(log_out),
        }

    manifest_rows: List[Dict[str, Any]] = [
        row_map[entry['slug']]
        for entry in registry.get('curated_runs', [])
        if entry['slug'] in row_map
    ]

    snapshot = {
        'generated_at': dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        'sync_mode': sync_mode,
        'registry': _rel(args.registry.expanduser().resolve()),
        'selected_slugs': sorted(wanted),
        'row_count': len(manifest_rows),
        'rows': manifest_rows,
    }
    (out_dir / 'registry_snapshot.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
    _write_overview(out_dir, registry, manifest_rows, sync_mode)
    print(str(out_dir))


if __name__ == '__main__':
    main()
