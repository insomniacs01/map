#!/usr/bin/env python3
"""Render a Project-ready local overview from canonical benchmark summaries.

This script is read-only with respect to model artifacts: it reuses existing
`summary_test.json` files and the manifest logic in `sync_benchmark_project_v2.py`
to produce a compact markdown overview for local review before GitHub Project sync.
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

CANONICAL_RUNS: List[Dict[str, Any]] = [
    {
        'section': 'Geometry',
        'label': 'Baseline (deploy protocol reference)',
        'summary_json': REPO_ROOT / 'eval_runs' / 'geom_generalize_test500_20260223_c20260210_sfix_direct_calibrated_coop' / 'summary_test.json',
        'line': 'LA', 'pack': 'report500', 'mode': 'M0', 'noise': 'N0', 'dropout': 'D0', 'seed': '42',
        'parent_epic': 'Geometry Base Models', 'curation_tier': 'Reference',
        'result': 'Baseline',
        'status_note': 'Historical deploy-protocol baseline',
        'next_action': 'Keep as geometry baseline reference',
    },
    {
        'section': 'Geometry',
        'label': 'Promoted startpoint (nm4_scaleonly_w02)',
        'summary_json': REPO_ROOT / 'eval_runs' / 'geom_recheck_test500_20260223_nm4_scaleonly_w02_calibrated' / 'summary_test.json',
        'line': 'LA', 'pack': 'report500', 'mode': 'M0', 'noise': 'N0', 'dropout': 'D0', 'seed': '42',
        'parent_epic': 'Geometry Base Models', 'curation_tier': 'Key',
        'result': 'PromotedStartpoint',
        'status_note': 'Current promoted geometry startpoint under deploy protocol',
        'next_action': 'Compare against promoted_v2 and decide next validation',
    },
    {
        'section': 'Geometry',
        'label': 'Promoted_v2 (FairLock / Plan1 two-stage)',
        'summary_json': REPO_ROOT / 'eval_runs' / 'geom_fairlock_test500_20260226_promoted_v2' / 'summary_test.json',
        'line': 'LA', 'pack': 'report500', 'mode': 'M0', 'noise': 'N0', 'dropout': 'D0', 'seed': '42',
        'parent_epic': 'Geometry Base Models', 'curation_tier': 'Key',
        'result': 'PromotedV2',
        'status_note': 'Current fairlock promoted_v2 geometry result',
        'next_action': 'Present as current promoted geometry candidate',
    },
    {
        'section': 'Detection',
        'label': 'Pinhole det baseline (single)',
        'summary_json': REPO_ROOT / 'eval_runs' / 'det_e2e_v5_eval_t005' / 'summary_test.json',
        'line': 'LA', 'pack': 'report500', 'mode': 'M1', 'noise': 'N0', 'dropout': 'D0', 'seed': '42',
        'parent_epic': 'Detection', 'curation_tier': 'Reference',
        'status': 'Done',
        'gate_status_override': 'none',
        'result': 'SingleBaseline',
        'status_note': 'Single-view detection baseline on comparable test50',
        'next_action': 'Compare directly against coop counterpart',
    },
    {
        'section': 'Detection',
        'label': 'Pinhole det baseline (coop)',
        'summary_json': REPO_ROOT / 'eval_runs' / 'det_e2e_v5_coop_eval_t005' / 'summary_test.json',
        'line': 'LA', 'pack': 'report500', 'mode': 'M1', 'noise': 'N0', 'dropout': 'D0', 'seed': '42',
        'parent_epic': 'Detection', 'curation_tier': 'Reference',
        'result': 'CoopBaseline',
        'status_note': 'Coop-view detection baseline on comparable test50',
        'next_action': 'Highlight single-vs-coop tradeoff in Project',
    },
]


def _load_sync_module():
    spec = importlib.util.spec_from_file_location('syncproj', SYNC_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _build_args(entry: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        project_org=None,
        project_number=None,
        github_token_env='GITHUB_TOKEN',
        run_id=None,
        line=entry.get('line'),
        pack=entry.get('pack'),
        mode=entry.get('mode'),
        noise=entry.get('noise'),
        dropout=entry.get('dropout'),
        seed=entry.get('seed'),
        status=entry.get('status'),
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
        gate_status_override=entry.get('gate_status_override'),
        dry_run=True,
        manifest_out=None,
        summary_json=Path(entry['summary_json']),
    )


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return 'NA'
    if isinstance(x, (int, float)):
        return f'{x:.{nd}f}'
    return str(x)


def _render_markdown(manifests: List[Dict[str, Any]], entries: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append('# Project-ready Evaluation Overview')
    lines.append('')
    lines.append('Generated from local `summary_test.json` artifacts using `sync_benchmark_project_v2.py` manifest logic.')
    lines.append('')
    lines.append('## Contract')
    lines.append('')
    lines.append('- Geometry: deploy protocol / fixed Test500 / coop-first evidence chain.')
    lines.append('- Detection: fairness-checked comparable test50 evidence chain.')
    lines.append('- Purpose: prepare readable local review before GitHub Project V2 sync.')
    lines.append('')

    for section in ['Geometry', 'Detection']:
        lines.append(f'## {section}')
        lines.append('')
        lines.append('| Label | run_id | gate_status | scale_mult | scale_ratio | pose_abs | depth_rel | det_ap_iou | artifact | curation |')
        lines.append('| --- | --- | --- | ---:| ---:| ---:| ---:| ---:| --- | --- |')
        for entry, manifest in zip(entries, manifests):
            if entry['section'] != section:
                continue
            geo = manifest.get('metrics_geometry_core', {}) or {}
            det = manifest.get('metrics_detection', {}) or {}
            artifact = manifest.get('artifact_link') or ''
            lines.append(
                '| ' + ' | '.join([
                    entry['label'],
                    str(manifest.get('run_id')),
                    str(manifest.get('gate_status') or 'NA'),
                    _fmt(manifest.get('gate_scale_to_gt_mult_err_mean')),
                    _fmt(geo.get('scale_to_gt_ratio_mean')),
                    _fmt(geo.get('pose_abs_mean')),
                    _fmt(geo.get('depth_rel_mean')),
                    _fmt(det.get('det_ap_iou')),
                    f'`{artifact}`',
                    str(entry.get('curation_tier')),
                ]) + ' |'
            )
        lines.append('')

    lines.append('## Suggested Project Metadata Defaults')
    lines.append('')
    lines.append('| Label | line | pack | mode | parent_epic | priority | owner |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- |')
    for entry in entries:
        lines.append('| ' + ' | '.join([
            entry['label'], entry['line'], entry['pack'], entry['mode'], entry['parent_epic'], 'P1', 'MassimoQu'
        ]) + ' |')
    lines.append('')

    lines.append('## Evidence')
    lines.append('')
    for entry, manifest in zip(entries, manifests):
        lines.append(f"- `{manifest['run_id']}` → `{manifest['summary_json']}`")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-md', type=Path, default=None, help='Optional markdown output path')
    ap.add_argument('--out-json', type=Path, default=None, help='Optional JSON output path for manifests')
    args = ap.parse_args()

    mod = _load_sync_module()
    manifests: List[Dict[str, Any]] = []
    for entry in CANONICAL_RUNS:
        summary_path = Path(entry['summary_json']).expanduser().resolve()
        doc = mod._read_json(summary_path)
        manifest = mod._build_manifest(_build_args(entry), summary_path, doc)
        manifests.append(manifest)

    markdown = _render_markdown(manifests, CANONICAL_RUNS)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown, encoding='utf-8')
    else:
        print(markdown, end='')

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(manifests, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
