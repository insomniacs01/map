#!/usr/bin/env python3
"""Sync local benchmark summary_test.json to GitHub Project V2.

This script standardizes local benchmark artifacts into one manifest and can upsert
that manifest into a GitHub Project V2 item keyed by `run_id`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import hashlib
import json
import math
import os
import ssl
import time
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


GRAPHQL_ENDPOINT = "https://api.github.com/graphql"

# Field aliases let us adapt to slightly different Project V2 field naming.
FIELD_ALIASES: Dict[str, List[str]] = {
    "run_id": ["run_id", "Run ID", "run id", "RunId"],
    "line": ["line", "Line"],
    "pack": ["pack", "Pack"],
    "mode": ["mode", "Mode"],
    "noise": ["noise", "Noise"],
    "dropout": ["dropout", "Dropout"],
    "seed": ["seed", "Seed"],
    "status": ["status", "Status"],
    "gate_status": ["gate_status", "Gate Status", "Gate"],
    "item_type": ["item_type", "Item Type", "Type"],
    "parent_epic": ["parent_epic", "Parent Epic", "Epic"],
    "priority": ["priority", "Priority"],
    "owner": ["owner", "Owner"],
    "artifact_link": ["artifact_link", "Artifact Link", "Artifact"],
    "repo_link": ["repo_link", "Repo Link", "Repository"],
    "repo@commit": ["repo@commit", "repo_commit", "Repo Commit"],
    "config_path": ["config_path", "Config Path"],
    "config_link": ["config_link", "Config Link"],
    "frames_json_sha256": ["frames_json_sha256", "Frames SHA256", "Frames Contract SHA256"],
    "metrics_geometry_core": ["metrics_geometry_core", "Metrics Geometry Core"],
    "metrics_detection": ["metrics_detection", "Metrics Detection"],
    "metrics_robustness": ["metrics_robustness", "Metrics Robustness"],
    "updated_at": ["updated_at", "Updated At"],
    "status_note": ["status_note", "Status Note"],
    "next_action": ["next_action", "Next Action"],
    "run_intent": ["run_intent", "Run Intent", "protocol", "protocol_tag"],
    "curation_tier": ["curation_tier", "Curation Tier"],
    "result": ["result", "Result"],
    "repo_name": ["repo_name", "Repo Name"],
    "branch": ["branch", "Branch"],
}

GEOMETRY_KEYS = [
    "pose_abs_mean",
    "pose_rot_mean",
    "pose_ate_mean",
    "pose_ate_rot_mean",
    "cross_agent_pose_trans_mean",
    "cross_agent_pose_rot_mean",
    "rig_pose_trans_mean",
    "rig_pose_rot_mean",
    "depth_rel_mean",
    "depth_mae_mean",
    "depth_rmse_mean",
    "chamfer_pred_to_gt_mean",
    "chamfer_gt_to_pred_mean",
    "bev_iou_raw_mean",
    "bev_iou_filtered_mean",
    "scale_to_gt_mult_err_mean",
    "scale_to_gt_ratio_mean",
    "scale_to_gt_err_mean",
    "scale_to_gt_log_err_mean",
    "scale_to_gt_eq_rel_err_mean",
]

DETECTION_KEYS = [
    "det_ap_iou",
    "det_precision_iou",
    "det_recall_iou",
    "det_mean_iou",
]

ROBUSTNESS_KEYS = [
    "scale_to_gt_ratio_p90_mean",
    "scale_to_gt_rows_used",
    "scale_to_gt_rows_skipped_ambiguous",
    "scale_ratio_p90_mean",
    "frames",
]


@dataclass
class MetricLeaf:
    model: str
    mode: str
    metrics: Dict[str, Any]


@dataclass
class ProjectField:
    field_id: str
    name: str
    data_type: str
    options: Dict[str, str]


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _norm_field_name(name: str) -> str:
    chars: List[str] = []
    for ch in name.lower().strip():
        if ch.isalnum():
            chars.append(ch)
        elif ch in (" ", "-", ".", "/", "@"):
            chars.append("_")
    out = "".join(chars)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _first_nonempty(*vals: Any) -> Optional[Any]:
    for val in vals:
        if val is None:
            continue
        if isinstance(val, str) and not val.strip():
            continue
        return val
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def _run_git(cwd: Path, args: List[str]) -> Optional[str]:
    try:
        cp = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    text = cp.stdout.strip()
    return text if text else None


def _detect_repo_commit(summary_path: Path, repo_link_hint: Optional[str]) -> Tuple[str, Optional[str], Optional[str]]:
    """Return (repo@commit, repo_link, commit_sha)."""
    search_roots: List[Path] = [summary_path.parent, Path.cwd(), Path(__file__).resolve().parent]
    commit_sha: Optional[str] = None
    repo_link: Optional[str] = None

    for root in search_roots:
        git_root = _run_git(root, ["rev-parse", "--show-toplevel"])
        if not git_root:
            continue
        git_root_path = Path(git_root)
        commit_sha = _run_git(git_root_path, ["rev-parse", "HEAD"])
        repo_link = _run_git(git_root_path, ["remote", "get-url", "origin"])
        break

    repo_link = _first_nonempty(repo_link_hint, repo_link)
    if repo_link and commit_sha:
        return f"{repo_link}@{commit_sha}", repo_link, commit_sha
    if commit_sha:
        return f"unknown_repo@{commit_sha}", repo_link, commit_sha
    if repo_link:
        return f"{repo_link}@unknown_commit", repo_link, commit_sha
    return "unknown_repo@unknown_commit", repo_link, commit_sha


def _git_current_branch(summary_path: Path) -> Optional[str]:
    search_roots: List[Path] = [summary_path.parent, Path.cwd(), Path(__file__).resolve().parent]
    for root in search_roots:
        git_root = _run_git(root, ["rev-parse", "--show-toplevel"])
        if not git_root:
            continue
        branch = _run_git(Path(git_root), ["rev-parse", "--abbrev-ref", "HEAD"])
        if branch:
            return branch
    return None


def _resolve_candidate_paths(summary_path: Path, candidate: str) -> List[Path]:
    p = Path(candidate)
    out: List[Path] = []
    if p.is_absolute():
        out.append(p)
    else:
        out.append((summary_path.parent / p).resolve())
        out.append((Path.cwd() / p).resolve())
    return out


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _compute_frames_json_sha256(summary_path: Path, doc: Dict[str, Any]) -> Tuple[Optional[str], str]:
    meta = doc.get("meta", {}) if isinstance(doc.get("meta"), dict) else {}
    run_info = doc.get("run_info", {}) if isinstance(doc.get("run_info"), dict) else {}
    frame_json_candidates = [
        meta.get("frames_json"),
        meta.get("gt_scale_contract_json"),
        run_info.get("frames_json"),
        run_info.get("gt_scale_contract_json"),
    ]

    for candidate in frame_json_candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        for path in _resolve_candidate_paths(summary_path, candidate):
            if path.is_file():
                return _sha256_bytes(path.read_bytes()), f"file:{path}"

    frames = doc.get("frames")
    if isinstance(frames, list):
        canonical = json.dumps(_json_safe(frames), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return _sha256_bytes(canonical.encode("utf-8")), "summary.frames"

    known_hash = _first_nonempty(meta.get("frames_hash_md5_v2"), meta.get("frames_hash_md5"), run_info.get("frames_hash_md5"))
    if known_hash:
        _warn("无法读取 frames_json 文件，退化使用已有 MD5 字段推导 sha256。")
        text = str(known_hash)
        return _sha256_bytes(text.encode("utf-8")), "summary.frames_hash_md5"

    return None, "missing"


def _iter_metric_leaves(doc: Dict[str, Any]) -> List[MetricLeaf]:
    metrics = doc.get("metrics")
    if not isinstance(metrics, dict):
        return []

    leaves: List[MetricLeaf] = []
    for model_name, by_mode in metrics.items():
        if not isinstance(by_mode, dict):
            continue
        nested = False
        for mode_name, payload in by_mode.items():
            if isinstance(payload, dict):
                leaves.append(MetricLeaf(model=str(model_name), mode=str(mode_name), metrics=payload))
                nested = True
        if not nested:
            leaves.append(MetricLeaf(model=str(model_name), mode="default", metrics=by_mode))
    return leaves


def _select_primary_leaf(leaves: List[MetricLeaf], mode_hint: Optional[str]) -> Optional[MetricLeaf]:
    if not leaves:
        return None

    if mode_hint:
        hint = mode_hint.strip().lower()
        for leaf in reversed(leaves):
            if leaf.mode.lower() == hint:
                return leaf

    for preferred_mode in ("coop", "single"):
        for leaf in reversed(leaves):
            if leaf.mode.lower() == preferred_mode:
                return leaf

    return leaves[-1]


def _select_coop_leaf_for_gate(leaves: List[MetricLeaf]) -> Optional[MetricLeaf]:
    for leaf in reversed(leaves):
        if leaf.mode.lower() == "coop":
            return leaf
    return None


def _pick_metric_values(metrics: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in keys:
        if key in metrics:
            out[key] = _json_safe(metrics.get(key))
    return out


def _pick_robustness_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out = _pick_metric_values(metrics, ROBUSTNESS_KEYS)
    for key, value in metrics.items():
        if key in out:
            continue
        key_l = str(key).lower()
        if any(token in key_l for token in ("p90", "p95", "p99", "tail", "worst", "max", "std", "rows_")):
            out[key] = _json_safe(value)
    return out


def _infer_gate_status(coop_leaf: Optional[MetricLeaf]) -> Tuple[str, Optional[float], str]:
    """Gate rule: pass iff coop scale_to_gt_mult_err_mean <= 1.3, else Blocked."""
    if coop_leaf is None:
        return "Blocked", None, "missing coop metrics"

    metric = _to_float(coop_leaf.metrics.get("scale_to_gt_mult_err_mean"))
    if metric is None:
        return "Blocked", None, "missing coop.scale_to_gt_mult_err_mean"

    if metric <= 1.3:
        return "GatePassed", metric, f"{coop_leaf.model}/{coop_leaf.mode}.scale_to_gt_mult_err_mean<=1.3"
    return "Blocked", metric, f"{coop_leaf.model}/{coop_leaf.mode}.scale_to_gt_mult_err_mean>1.3"


def _apply_gate_status_override(
    gate_status: Optional[str],
    gate_metric: Optional[float],
    gate_basis: str,
    override: Optional[str],
) -> Tuple[Optional[str], Optional[float], str]:
    if override is None:
        return gate_status, gate_metric, gate_basis
    token = str(override).strip()
    if not token:
        return gate_status, gate_metric, gate_basis
    if token.lower() in {"none", "na", "n/a", "skip"}:
        return None, None, f"override:{token.lower()}"
    return token, gate_metric, f"override:{token}"


def _extract_seed(doc: Dict[str, Any], seed_arg: Optional[str]) -> Optional[Any]:
    if seed_arg is not None:
        return seed_arg
    meta = doc.get("meta", {}) if isinstance(doc.get("meta"), dict) else {}
    run_info = doc.get("run_info", {}) if isinstance(doc.get("run_info"), dict) else {}
    return _first_nonempty(meta.get("seed"), run_info.get("seed"))


def _as_text_json(obj: Any) -> str:
    return json.dumps(_json_safe(obj), ensure_ascii=False, sort_keys=True)


def _build_manifest(args: argparse.Namespace, summary_path: Path, doc: Dict[str, Any]) -> Dict[str, Any]:
    run_id = args.run_id if args.run_id else summary_path.parent.name
    leaves = _iter_metric_leaves(doc)
    primary_leaf = _select_primary_leaf(leaves, args.mode)
    coop_leaf = _select_coop_leaf_for_gate(leaves)

    gate_status, gate_metric, gate_basis = _infer_gate_status(coop_leaf)
    gate_status, gate_metric, gate_basis = _apply_gate_status_override(
        gate_status, gate_metric, gate_basis, getattr(args, "gate_status_override", None)
    )

    primary_metrics = primary_leaf.metrics if primary_leaf else {}
    metrics_geometry_core = _pick_metric_values(primary_metrics, GEOMETRY_KEYS)
    metrics_detection = _pick_metric_values(primary_metrics, DETECTION_KEYS)
    metrics_robustness = _pick_robustness_metrics(primary_metrics)

    frames_sha256, frames_sha_source = _compute_frames_json_sha256(summary_path, doc)
    repo_at_commit, inferred_repo_link, commit_sha = _detect_repo_commit(summary_path, args.repo_link)

    generated_at = None
    meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
    if isinstance(meta, dict):
        generated_at = _first_nonempty(meta.get("generated_at_utc"), meta.get("generated_at"))

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "summary_json": str(summary_path),
        "repo@commit": repo_at_commit,
        "repo_link": _first_nonempty(args.repo_link, inferred_repo_link),
        "commit": commit_sha,
        "frames_json_sha256": frames_sha256,
        "frames_json_sha256_source": frames_sha_source,
        "gate_status": gate_status,
        "gate_scale_to_gt_mult_err_mean": gate_metric,
        "gate_basis": gate_basis,
        "metrics_source": {
            "model": primary_leaf.model if primary_leaf else None,
            "mode": primary_leaf.mode if primary_leaf else None,
        },
        "metrics_geometry_core": metrics_geometry_core,
        "metrics_detection": metrics_detection,
        "metrics_robustness": metrics_robustness,
        "updated_at": _now_utc_iso(),
        "summary_generated_at": generated_at,
        "artifact_link": _first_nonempty(args.artifact_link, str(summary_path.parent)),
        "status_note": _first_nonempty(args.status_note, ""),
        "next_action": _first_nonempty(args.next_action, ""),
        "run_intent": args.run_intent,
        "line": args.line,
        "pack": args.pack,
        "mode": args.mode,
        "noise": args.noise,
        "dropout": args.dropout,
        "seed": _extract_seed(doc, args.seed),
        "status": _first_nonempty(args.status, "Done" if gate_status in (None, "GatePassed") else "Blocked"),
        "item_type": args.item_type,
        "parent_epic": args.parent_epic,
        "priority": args.priority,
        "owner": args.owner,
        "config_path": args.config_path,
        "config_link": args.config_link,
        "curation_tier": args.curation_tier,
        "result": args.result,
        "repo_name": _first_nonempty(args.repo_name, summary_path.parents[2].name if len(summary_path.parents) >= 3 else None),
        "branch": _first_nonempty(args.branch, _git_current_branch(summary_path)),
    }

    return _json_safe(manifest)


def _write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _info(f"manifest written: {path}")


def _graphql(token: str, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    req = urllib.request.Request(
        GRAPHQL_ENDPOINT,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
        },
        method="POST",
    )

    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("errors"):
                raise RuntimeError(f"GitHub GraphQL errors: {json.dumps(data['errors'], ensure_ascii=False)}")
            if "data" not in data:
                raise RuntimeError(f"GitHub GraphQL malformed response: {data}")
            return data["data"]
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code < 500 or attempt == 3:
                raise RuntimeError(f"GitHub GraphQL HTTP {exc.code}: {body}") from exc
            _warn(f"GitHub GraphQL transient HTTP {exc.code}; retry {attempt}/3")
            time.sleep(2 * attempt)
        except (urllib.error.URLError, TimeoutError, ssl.SSLError, http.client.IncompleteRead) as exc:
            if attempt == 3:
                raise RuntimeError(f"GitHub GraphQL transport error: {exc}") from exc
            _warn(f"GitHub GraphQL transient transport error; retry {attempt}/3: {exc}")
            time.sleep(2 * attempt)

    raise RuntimeError("GitHub GraphQL unreachable after retries")


def _load_project_fields(token: str, org: str, number: int) -> Tuple[str, str, Dict[str, ProjectField]]:
    query = """
    query($org: String!, $number: Int!) {
      organization(login: $org) {
        projectV2(number: $number) {
          id
          title
          fields(first: 100) {
            nodes {
              __typename
              ... on ProjectV2FieldCommon {
                id
                name
                dataType
              }
              ... on ProjectV2SingleSelectField {
                options {
                  id
                  name
                }
              }
            }
          }
        }
      }
    }
    """
    data = _graphql(token, query, {"org": org, "number": number})
    proj = (data.get("organization") or {}).get("projectV2")
    if not proj:
        raise RuntimeError(f"Project V2 not found: org={org}, number={number}")

    fields_by_norm: Dict[str, ProjectField] = {}
    for node in ((proj.get("fields") or {}).get("nodes") or []):
        if not isinstance(node, dict):
            continue
        field_id = node.get("id")
        name = node.get("name")
        data_type = node.get("dataType")
        if not (field_id and name and data_type):
            continue
        options_map: Dict[str, str] = {}
        for opt in node.get("options") or []:
            if not isinstance(opt, dict):
                continue
            opt_name = opt.get("name")
            opt_id = opt.get("id")
            if opt_name and opt_id:
                options_map[_norm_field_name(str(opt_name))] = str(opt_id)

        pf = ProjectField(
            field_id=str(field_id),
            name=str(name),
            data_type=str(data_type),
            options=options_map,
        )
        fields_by_norm[_norm_field_name(pf.name)] = pf

    return str(proj["id"]), str(proj.get("title") or ""), fields_by_norm


def _lookup_field(fields_by_norm: Dict[str, ProjectField], logical_key: str) -> Optional[ProjectField]:
    aliases = FIELD_ALIASES.get(logical_key, [logical_key])
    for alias in aliases:
        hit = fields_by_norm.get(_norm_field_name(alias))
        if hit is not None:
            return hit
    return None


def _parse_item_field_value(node: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (field_id, field_name, string_value)."""
    field = node.get("field") if isinstance(node, dict) else None
    if not isinstance(field, dict):
        return None, None, None
    field_id = field.get("id")
    field_name = field.get("name")
    if not field_id or not field_name:
        return None, None, None

    tname = node.get("__typename")
    if tname == "ProjectV2ItemFieldTextValue":
        return str(field_id), str(field_name), node.get("text")
    if tname == "ProjectV2ItemFieldNumberValue":
        num = node.get("number")
        return str(field_id), str(field_name), None if num is None else str(num)
    if tname == "ProjectV2ItemFieldDateValue":
        return str(field_id), str(field_name), node.get("date")
    if tname == "ProjectV2ItemFieldSingleSelectValue":
        return str(field_id), str(field_name), node.get("name")
    return None, None, None


def _iter_project_items(token: str, org: str, number: int) -> Iterable[Dict[str, Any]]:
    query = """
    query($org: String!, $number: Int!, $after: String) {
      organization(login: $org) {
        projectV2(number: $number) {
          items(first: 100, after: $after) {
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              id
              type
              content {
                __typename
                ... on DraftIssue { id title }
                ... on Issue { title number url }
                ... on PullRequest { title number url }
              }
              fieldValues(first: 100) {
                nodes {
                  __typename
                  ... on ProjectV2ItemFieldTextValue {
                    text
                    field { ... on ProjectV2FieldCommon { id name } }
                  }
                  ... on ProjectV2ItemFieldNumberValue {
                    number
                    field { ... on ProjectV2FieldCommon { id name } }
                  }
                  ... on ProjectV2ItemFieldDateValue {
                    date
                    field { ... on ProjectV2FieldCommon { id name } }
                  }
                  ... on ProjectV2ItemFieldSingleSelectValue {
                    name
                    optionId
                    field { ... on ProjectV2FieldCommon { id name } }
                  }
                }
              }
            }
          }
        }
      }
    }
    """

    after: Optional[str] = None
    while True:
        data = _graphql(token, query, {"org": org, "number": number, "after": after})
        project = (data.get("organization") or {}).get("projectV2")
        if not project:
            return
        items = (project.get("items") or {}).get("nodes") or []
        for item in items:
            if isinstance(item, dict):
                yield item

        page_info = (project.get("items") or {}).get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            return
        after = page_info.get("endCursor")


def _find_item_by_run_id(
    token: str,
    org: str,
    number: int,
    run_id: str,
    run_id_field: Optional[ProjectField],
) -> Optional[Dict[str, Any]]:
    run_id_norm = str(run_id).strip()
    fallback_titles = {run_id_norm, f"Benchmark: {run_id_norm}", f"benchmark: {run_id_norm}"}

    for item in _iter_project_items(token, org, number):
        if run_id_field is not None:
            for fv in ((item.get("fieldValues") or {}).get("nodes") or []):
                if not isinstance(fv, dict):
                    continue
                field_id, _field_name, value = _parse_item_field_value(fv)
                if field_id != run_id_field.field_id:
                    continue
                if value is not None and str(value).strip() == run_id_norm:
                    return item

        content = item.get("content")
        if isinstance(content, dict):
            title = content.get("title")
            if isinstance(title, str) and title.strip() in fallback_titles:
                return item

    return None


def _create_draft_issue_item(token: str, project_id: str, run_id: str, manifest: Dict[str, Any], item_title: Optional[str] = None) -> str:
    mutation = """
    mutation($projectId: ID!, $title: String!, $body: String!) {
      addProjectV2DraftIssue(input: {projectId: $projectId, title: $title, body: $body}) {
        projectItem { id }
      }
    }
    """
    display_title = str(item_title).strip() if item_title else run_id
    body_lines = [
        f"Auto synced benchmark run `{run_id}`.",
        "",
        f"- result: {manifest.get('result')}",
        f"- gate_status: {manifest.get('gate_status')}",
        f"- summary_json: {manifest.get('summary_json')}",
        f"- updated_at: {manifest.get('updated_at')}",
    ]
    status_note = manifest.get("status_note")
    if status_note:
        body_lines.extend(["", f"- status_note: {status_note}"])
    body = "\n".join(body_lines)
    data = _graphql(token, mutation, {"projectId": project_id, "title": display_title, "body": body})
    node = ((data.get("addProjectV2DraftIssue") or {}).get("projectItem")) or {}
    item_id = node.get("id")
    if not item_id:
        raise RuntimeError("Failed to create Project V2 draft issue item.")
    return str(item_id)


def _update_draft_issue_title(token: str, draft_issue_id: str, title: str) -> None:
    mutation = """
    mutation($draftIssueId: ID!, $title: String!) {
      updateProjectV2DraftIssue(input: {draftIssueId: $draftIssueId, title: $title}) {
        draftIssue { id title }
      }
    }
    """
    _graphql(token, mutation, {"draftIssueId": draft_issue_id, "title": title})


def _coerce_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return None


def _build_field_value_payload(field: ProjectField, value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None

    if field.data_type == "TEXT":
        text = str(value)
        return {"text": text}

    if field.data_type == "NUMBER":
        num = _to_float(value)
        if num is None:
            return None
        return {"number": num}

    if field.data_type == "DATE":
        date_text = _coerce_date(value)
        if not date_text:
            return None
        return {"date": date_text}

    if field.data_type == "SINGLE_SELECT":
        target = _norm_field_name(str(value))
        option_id = field.options.get(target)
        if not option_id:
            return None
        return {"singleSelectOptionId": option_id}

    return None


def _update_item_field(
    token: str,
    project_id: str,
    item_id: str,
    field_id: str,
    value_payload: Dict[str, Any],
) -> None:
    mutation = """
    mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $value: ProjectV2FieldValue!) {
      updateProjectV2ItemFieldValue(
        input: {
          projectId: $projectId,
          itemId: $itemId,
          fieldId: $fieldId,
          value: $value
        }
      ) {
        projectV2Item { id }
      }
    }
    """
    _graphql(
        token,
        mutation,
        {
            "projectId": project_id,
            "itemId": item_id,
            "fieldId": field_id,
            "value": value_payload,
        },
    )


def _clear_item_field(token: str, project_id: str, item_id: str, field_id: str) -> None:
    mutation = """
    mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!) {
      clearProjectV2ItemFieldValue(
        input: {
          projectId: $projectId,
          itemId: $itemId,
          fieldId: $fieldId
        }
      ) {
        projectV2Item { id }
      }
    }
    """
    _graphql(
        token,
        mutation,
        {
            "projectId": project_id,
            "itemId": item_id,
            "fieldId": field_id,
        },
    )


def _build_project_payload(args: argparse.Namespace, manifest: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "run_id": manifest.get("run_id"),
        "line": manifest.get("line"),
        "pack": manifest.get("pack"),
        "mode": manifest.get("mode"),
        "noise": manifest.get("noise"),
        "dropout": manifest.get("dropout"),
        "seed": manifest.get("seed"),
        "status": manifest.get("status"),
        "gate_status": "__CLEAR__" if (manifest.get("gate_status") is None and str(getattr(args, "gate_status_override", "")).strip().lower() in {"none", "na", "n/a", "skip"}) else manifest.get("gate_status"),
        "item_type": manifest.get("item_type"),
        "parent_epic": manifest.get("parent_epic"),
        "priority": manifest.get("priority"),
        "owner": manifest.get("owner"),
        "artifact_link": manifest.get("artifact_link"),
        "repo_link": _first_nonempty(args.repo_link, manifest.get("repo_link")),
        "repo@commit": manifest.get("repo@commit"),
        "config_path": manifest.get("config_path"),
        "config_link": manifest.get("config_link"),
        "frames_json_sha256": manifest.get("frames_json_sha256"),
        "metrics_geometry_core": _as_text_json(manifest.get("metrics_geometry_core", {})),
        "metrics_detection": _as_text_json(manifest.get("metrics_detection", {})),
        "metrics_robustness": _as_text_json(manifest.get("metrics_robustness", {})),
        "updated_at": manifest.get("updated_at"),
        "status_note": manifest.get("status_note"),
        "next_action": manifest.get("next_action"),
        "run_intent": manifest.get("run_intent"),
        "curation_tier": manifest.get("curation_tier"),
        "result": manifest.get("result"),
        "repo_name": manifest.get("repo_name"),
        "branch": manifest.get("branch"),
    }
    return payload


def _sync_to_project(args: argparse.Namespace, manifest: Dict[str, Any]) -> None:
    if not args.project_org or args.project_number is None:
        raise RuntimeError("project_org / project_number are required when not using --dry_run")

    token = os.getenv(args.github_token_env)
    if not token:
        raise RuntimeError(f"Missing GitHub token env: {args.github_token_env}")

    project_id, project_title, fields_by_norm = _load_project_fields(token, args.project_org, args.project_number)
    _info(f"target project: {args.project_org}/{args.project_number} ({project_title})")

    run_id = str(manifest.get("run_id"))
    run_id_field = _lookup_field(fields_by_norm, "run_id")
    if run_id_field is None:
        _warn("Project 缺少 run_id 字段，将回退为按标题匹配，可能产生重复 item。")

    existing_item = _find_item_by_run_id(token, args.project_org, args.project_number, run_id, run_id_field)
    if existing_item:
        item_id = str(existing_item.get("id"))
        _info(f"found existing item for run_id={run_id}: {item_id}")
        content = existing_item.get("content") if isinstance(existing_item, dict) else None
        desired_title = str(args.item_title).strip() if args.item_title else ""
        if desired_title and isinstance(content, dict) and str(content.get("__typename") or "") == "DraftIssue":
            current_title = str(content.get("title") or "").strip()
            draft_issue_id = content.get("id")
            if draft_issue_id and current_title != desired_title:
                _update_draft_issue_title(token, str(draft_issue_id), desired_title)
                _info(f"updated existing draft title: {current_title or run_id} -> {desired_title}")
    else:
        item_id = _create_draft_issue_item(token, project_id, run_id, manifest, args.item_title)
        _info(f"created draft issue item: {item_id}")

    payload = _build_project_payload(args, manifest)
    updated = 0
    skipped = 0
    for logical_key, value in payload.items():
        if value is None:
            continue
        field = _lookup_field(fields_by_norm, logical_key)
        if field is None:
            _warn(f"field missing in project (skip): {logical_key}")
            skipped += 1
            continue

        if value == "__CLEAR__":
            _clear_item_field(token, project_id, item_id, field.field_id)
            updated += 1
            continue

        val_payload = _build_field_value_payload(field, value)
        if val_payload is None:
            _warn(f"field value incompatible (skip): {logical_key} -> {field.name} ({field.data_type})")
            skipped += 1
            continue

        _update_item_field(token, project_id, item_id, field.field_id, val_payload)
        updated += 1

    _info(f"sync finished: updated={updated}, skipped={skipped}, item_id={item_id}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sync benchmark manifest to GitHub Project V2.")
    ap.add_argument("--summary_json", type=Path, required=True, help="Path to summary_test.json")
    ap.add_argument("--project_org", type=str, default=None)
    ap.add_argument("--project_number", type=int, default=None)
    ap.add_argument("--github_token_env", type=str, default="GITHUB_TOKEN")
    ap.add_argument("--run_id", type=str, default=None, help="Default: summary parent directory name")
    ap.add_argument("--item_title", type=str, default=None, help="Optional friendly Project item title; run_id remains the id key")

    ap.add_argument("--line", type=str, default=None)
    ap.add_argument("--pack", type=str, default=None)
    ap.add_argument("--mode", type=str, default=None)
    ap.add_argument("--noise", type=str, default=None)
    ap.add_argument("--dropout", type=str, default=None)
    ap.add_argument("--seed", type=str, default=None)
    ap.add_argument("--status", type=str, default=None)
    ap.add_argument("--gate_status_override", type=str, default=None, help="Override derived gate status; use none/skip to clear gate_status on the remote Project item for reference runs")
    ap.add_argument("--item_type", type=str, default=None)
    ap.add_argument("--parent_epic", type=str, default=None)
    ap.add_argument("--priority", type=str, default=None)
    ap.add_argument("--owner", type=str, default=None)
    ap.add_argument("--artifact_link", type=str, default=None)
    ap.add_argument("--repo_link", type=str, default=None)
    ap.add_argument("--config_path", type=str, default=None)
    ap.add_argument("--config_link", type=str, default=None)
    ap.add_argument("--status_note", type=str, default=None)
    ap.add_argument("--next_action", type=str, default=None)
    ap.add_argument("--run_intent", type=str, default=None, help="Protocol tag for cross-metric comparability (e.g. mapanything-fairlock-scale2gt-v1)")
    ap.add_argument("--curation_tier", type=str, default=None)
    ap.add_argument("--result", type=str, default=None, help="Optional human-readable result label for the Project field")
    ap.add_argument("--repo_name", type=str, default=None, help="Optional repo short name for the Project field")
    ap.add_argument("--branch", type=str, default=None, help="Optional branch name for the Project field")

    ap.add_argument("--dry_run", action="store_true", help="Only print plan + write manifest; do not write GitHub")
    ap.add_argument("--manifest_out", type=Path, default=None, help="Default: <summary_dir>/benchmark_manifest.json")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    summary_path = args.summary_json.expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary_json not found: {summary_path}")

    doc = _read_json(summary_path)
    manifest = _build_manifest(args, summary_path, doc)
    manifest_out = args.manifest_out.expanduser().resolve() if args.manifest_out else (summary_path.parent / "benchmark_manifest.json")
    _write_manifest(manifest_out, manifest)

    _info("manifest preview:")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))

    if args.dry_run:
        _info("dry-run enabled: skip GitHub Project V2 writes.")
        return

    _sync_to_project(args, manifest)


if __name__ == "__main__":
    main()
