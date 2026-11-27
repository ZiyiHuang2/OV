"""Matched evidence and budget decisions for V4 long-run follow-ups."""

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from isaid_protocol import HELD_CLASSES, file_sha256, read_json
from prepare_isaid import CLASSES


@dataclass(frozen=True)
class Study:
    ids: tuple
    anchor: str
    long_anchors: tuple
    held_jobs: tuple = ()


RESEARCH_UNITS = (
    (Study(("D01",), "B0", ("T01",), ("D01-H1", "H11")),
     Study(("D02",), "C0", ("T00",), ("D02-H1", "H10"))),
    (Study(("E01", "E02"), "B0", ("T01", "T03")),),
    (Study(("F01", "F02"), "B0", ("T01",), ("F01-H1", "F02-H1")),),
    tuple(Study((f"F{i:02d}",), "C2", ("T04",)) for i in range(3, 6)),
    *((Study((f"F{i:02d}",), "C2", ("T04",)),) for i in (6, 7)),
    (Study(("F08", "F09"), "C2", ("T04",)),),
)
REPORT_ERRORS = (OSError, ValueError, KeyError, TypeError, ZeroDivisionError)
TRANSFER_METRICS = frozenset(("external_dev_miou", "held_dev_hiou"))


def unit_ids(unit):
    return tuple(name for study in unit for name in study.ids)


def requested_units(requested_ids):
    requested = None if requested_ids is None else set(requested_ids)
    return [unit for unit in RESEARCH_UNITS if requested is None or set(unit_ids(unit)) <= requested]


def promotion_dependencies(requested_ids=None, *, include_held=True):
    return {job for unit in requested_units(requested_ids) for study in unit
            for job in study.ids + study.long_anchors + (study.held_jobs if include_held else ())}


def report_path(root, job, variant, mode="dev_epoch_015"):
    return root / job / "main" / variant / mode / "report.json"


def supported_iou(report):
    classes, matrix = report["test_classes"], report["confusion_matrix"]
    if (not classes or len(set(classes)) != len(classes) or len(matrix) != len(classes)
            or any(len(row) != len(classes) for row in matrix)
            or any(type(count) is not int or count < 0 for row in matrix for count in row)):
        raise ValueError("Confusion matrix does not match its vocabulary")
    support = tuple(sum(row) for row in matrix)
    scores = {name: 100 * matrix[i][i] / (support[i] + sum(row[i] for row in matrix) - matrix[i][i])
              for i, name in enumerate(classes) if support[i]}
    if not scores:
        raise ValueError("No ground-truth-supported evaluation classes")
    return support, scores


def contrast(metric, ids, scores, baseline=None, **details):
    first, second = ids if len(ids) == 2 else (baseline, ids[0])
    result = {"metric": metric, "state": "matched", "contrast": [first, second],
              "signed_delta_pp": scores[second] - scores[first], "scores": scores,
              **details}
    if baseline in scores:
        result["baseline_deltas_pp"] = {name: scores[name] - scores[baseline] for name in ids}
    return result


def class_deltas(terms, class_scores):
    """Contrast each class with nonzero ground-truth support in every cell."""
    common = set.intersection(*(set(class_scores[name]) for name in terms))
    return {category: sum(weight * class_scores[name][category] for name, weight in terms.items())
            for category in sorted(common)}


def pair_class_deltas(item, class_scores):
    first, second = item["contrast"]
    return {**item, "class_deltas_pp": class_deltas({first: -1, second: 1}, class_scores)}


def evidence_scope(item):
    if item["metric"] == "external_dev_miou":
        return "external:" + item["sample_ids_sha256"]
    if item["metric"] == "held_dev_hiou":
        return "held:" + item["split"]
    return "source"


def qualify_evidence(evidence, minimum_delta):
    material = [item for item in evidence if item["state"] == "matched"
                and abs(item["signed_delta_pp"]) >= minimum_delta]
    qualifying = []
    for item in material:
        sign = 1 if item["signed_delta_pp"] > 0 else -1
        classes = [name for name, delta in item.get("class_deltas_pp", {}).items()
                   if delta * sign >= minimum_delta]
        endpoints = [other for other in material if other["contrast"] == item["contrast"]]
        scopes = {evidence_scope(other) for other in endpoints}
        if item["metric"] == "source_class_miou":
            item["support"] = {"kind": "registered_class_group", "classes": item["classes"]}
        elif len(scopes) >= 2:
            tradeoff = any(other["signed_delta_pp"] * sign < 0 for other in endpoints)
            item["support"] = {"kind": "endpoint_tradeoff" if tradeoff else "multiple_endpoints",
                               "scopes": sorted(scopes)}
        elif len(classes) >= 2:
            item["support"] = {"kind": "directional_class_support", "classes": classes}
        else:
            continue
        qualifying.append(item)
    return material, qualifying


def class_evidence(ids, anchor, reports, groups):
    if not groups:
        return
    source_scores = {key: supported_iou(report)[1] for key, report in reports.items()}
    for group, classes in groups.items():
        if not classes or len(set(classes)) != len(classes) or not set(classes) <= set(CLASSES):
            raise ValueError(f"Invalid preregistered source class group: {group}")
        if not all(set(classes) <= scores.keys() for scores in source_scores.values()):
            yield {"metric": "source_class_miou", "group": group, "classes": classes,
                   "state": "missing_class_support"}
            continue
        scores = {key: sum(row[name] for name in classes) / len(classes) for key, row in source_scores.items()}
        yield contrast("source_class_miou", ids, scores, anchor, group=group, classes=classes)


def checkpoint_digest(source, path):
    # Some historical source reports predate digest recording. Use their final
    # artifact to bind any compatible external evaluation to an actual file.
    return source.get("checkpoint_sha256") or file_sha256(path.parents[1] / "train" / "final.pt")


def training_metadata(report, path):
    training = report.get("checkpoint_training")
    if training is None:
        training = read_json(path.parents[1] / "train" / "report.json")
    return training


def standard_inference(report, source, size):
    return ((report["split"], report["batch_size"], report["dtype"], report["tf32"], report["limit"],
             report["force_keep_all"], report["prompt"], report.get("inference_sizes", [size]),
             report.get("inference_rotations", [0]), report.get("inference_prompts", [source["prompt"]]),
             report.get("native_clip_weight", 0))
            == ("val", 1, "float32", False, None, False, source["prompt"], [size], [0], [source["prompt"]], 0))


def external_evidence(manifest_path, root, reference_root, ids, anchor, sources, paths,
                      historical_evidence_root=None):
    base = {"metric": "external_dev_miou", "manifest": str(manifest_path)}
    if not manifest_path.is_file():
        return {**base, "state": "missing_manifest"}
    try:
        manifest = read_json(manifest_path)
        digest = file_sha256(manifest_path)
        classes = read_json(manifest_path.parent / manifest["test_classes"])
        samples = [row["sample_id"] for row in manifest["val"]]
        sample_hash = sha256("".join(name + "\n" for name in samples).encode()).hexdigest()
        if not samples or len(set(samples)) != len(samples):
            raise ValueError("External development sample IDs must be unique")
        names = ids if len(ids) == 2 else (anchor, *ids)
        scores, class_scores, expected_identity = {}, {}, None
        for name in names:
            source = sources[name]
            parent = reference_root.parent if name == anchor else root / name
            suffix = Path(digest[:16]) / source["config"]["variant"] / "eval" / "report.json"
            path = parent / "external" / suffix
            if name == anchor:
                history = historical_evidence_root or root / "history" / "external"
                candidate = history / suffix
                if candidate.is_file():
                    path = candidate
            if not path.is_file():
                return {**base, "state": "missing_reports", "missing": str(path)}
            data = read_json(path)
            training = training_metadata(source, paths[name])
            if (not standard_inference(data, source, source["size"])
                    or data["manifest"] != digest or data["test_classes"] != classes
                    or data["ignore_id"] != manifest["ignore_label"]
                    or data["sample_ids_sha256"] != sample_hash or data["images"] != len(samples)
                    or data["eval_samples"] is not None or data["checkpoint_kind"] != "evaluation_only"
                    or data["checkpoint_epoch"] != source["checkpoint_epoch"]
                    or data["checkpoint_steps"] != source["checkpoint_steps"]
                    or data["checkpoint_sha256"] != checkpoint_digest(source, paths[name])
                    or data["checkpoint_train_manifest"] != source["checkpoint_train_manifest"]
                    or data["train_classes"] != source["train_classes"]
                    or data["config"] != source["config"] or data["weights"] != source["weights"]
                    or any(data["checkpoint_training"].get(key) != training.get(key) for key in (
                        "manifest", "data_sha256", "config", "weights", "seed", "size", "batch_size", "lr",
                        "weight_decay", "training_policy", "alignment_weight", "limit",
                        "fusion_protocol", "region_to_text_loss_fraction", "patch_to_text_loss_fraction"))):
                raise ValueError(f"External evaluation does not match the source-training checkpoint and protocol: {path}")
            support, per_class = supported_iou(data)
            identity = data.get("data_sha256"), support
            if expected_identity is not None and identity != expected_identity:
                raise ValueError("External evaluations differ in recorded data identity or ground-truth class support")
            expected_identity = identity
            scores[name] = sum(per_class.values()) / len(per_class)
            class_scores[name] = per_class
    except REPORT_ERRORS as exc:
        return {**base, "state": "invalid_report", "reason": str(exc)}
    item = contrast(base["metric"], ids, scores, anchor, manifest=base["manifest"],
                    manifest_sha256=digest, sample_ids_sha256=sample_hash)
    return pair_class_deltas(item, class_scores)


def held_evidence(root, study, sources, source_paths):
    base = {"metric": "held_dev_hiou"}
    if not study.held_jobs:
        return {**base, "state": "not_planned"}
    names = study.ids if len(study.ids) == 2 else (*study.ids, study.anchor)
    values, class_scores = {}, {}
    for name, job in zip(names, study.held_jobs, strict=True):
        source = sources[name]
        split = "H1"
        path = report_path(root, job, source["config"]["variant"])
        if not path.is_file():
            return {**base, "split": split, "state": "missing_reports", "missing": str(path)}
        try:
            data = read_json(path)
            training = data["checkpoint_training"]
            full_training = training_metadata(source, source_paths[name])
            held_manifest = root / job / "held_data" / "manifest.json"
            digest = file_sha256(held_manifest)
            manifest = read_json(held_manifest)
            seen = [category for category in CLASSES if category not in HELD_CLASSES[split]]
            if (not standard_inference(data, source, source["size"])
                    or data["test_classes"] != CLASSES or data["train_classes"] != seen
                    or data["ignore_id"] != source["ignore_id"]
                    or data["sample_ids_sha256"] != source["sample_ids_sha256"]
                    or (data["images"], data["eval_samples"], data["eval_sample_seed"]) != (512, 512, 42)
                    or data["checkpoint_kind"] != "evaluation_only"
                    or data["checkpoint_epoch"] != source["checkpoint_epoch"]
                    or data["checkpoint_steps"] != source["checkpoint_steps"]
                    or data["checkpoint_sha256"] != file_sha256(path.parents[1] / "train" / "final.pt")
                    or (data["manifest"], data["checkpoint_train_manifest"], training["manifest"]) != (digest,) * 3
                    or manifest["source_manifest_sha256"] != source["manifest"]
                    or not data["data_sha256"]
                    or (data["data_sha256"] != source.get("data_sha256") and source.get("data_sha256") is not None)
                    or data["data_sha256"] != manifest["source_data_sha256"]
                    or data["data_sha256"] != training["data_sha256"]
                    or data["config"] != source["config"] or data["weights"] != source["weights"]
                    or any(training.get(key) != full_training.get(key) for key in (
                        "config", "weights", "seed", "size", "batch_size", "lr", "weight_decay", "training_policy",
                        "alignment_weight", "limit", "fusion_protocol", "region_to_text_loss_fraction",
                        "patch_to_text_loss_fraction"))):
                raise ValueError(f"Held evaluation does not match {split} and the registered short-run protocol: {path}")
            support, per_class = supported_iou(data)
            seen_scores = [per_class[category] for category in seen if category in per_class]
            if (support != supported_iou(source)[0] or not seen_scores
                    or not set(HELD_CLASSES[split]) <= per_class.keys()):
                raise ValueError(f"Held development subset differs or lacks class support: {path}")
            a = sum(seen_scores) / len(seen_scores)
            b = sum(per_class[category] for category in HELD_CLASSES[split]) / len(HELD_CLASSES[split])
            values[name] = {"seen_miou": a, "held_miou": b, "hiou": 2 * a * b / (a + b) if a + b else 0.}
            class_scores[name] = per_class
        except REPORT_ERRORS as exc:
            return {**base, "split": split, "state": "invalid_report", "reason": str(exc)}
    item = contrast(base["metric"], study.ids, {name: row["hiou"] for name, row in values.items()},
                    study.anchor, split="H1", breakdown=values)
    return pair_class_deltas(item, class_scores)


def factorial_evidence(path, run, dev_result, anchor_key, anchor_subset, audited_data, scores, class_scores):
    metric = {"metric": "source_dev_factorial_interaction"}
    if not path.is_file():
        return {**metric, "state": "missing_reports", "missing": str(path)}
    try:
        key, subset, content, c1_score = dev_result(path, run)
        if (key != anchor_key or subset != anchor_subset
                or content not in (None, audited_data)):
            raise ValueError("Short-run C1 does not match the factorial cells")
        c1_classes = supported_iou(read_json(path))[1]
    except REPORT_ERRORS as exc:
        return {**metric, "state": "invalid_report", "reason": str(exc)}
    scores = {**scores, "C1": c1_score}
    return {**metric, "state": "matched",
            "signed_delta_pp": scores["C1"] - scores["E01"] - scores["E02"] + scores["B0"],
            "scores": scores, "contrast": "C1 - E01 - E02 + B0",
            "class_deltas_pp": class_deltas({"C1": 1, "E01": -1, "E02": -1, "B0": 1},
                                            {**class_scores, "C1": c1_classes}),
            "multilayer_effects_pp": {"decoder_off": scores["E01"] - scores["B0"],
                                      "decoder_on": scores["C1"] - scores["E02"]},
            "decoder_effects_pp": {"multilayer_off": scores["E02"] - scores["B0"],
                                   "multilayer_on": scores["C1"] - scores["E01"]}}


def study_decision(study, root, reference_root, minimum_delta, fixed, dev_result, audit,
                   external_manifests, class_groups, historical_evidence_root):
    ids, anchor = study.ids, study.anchor
    decision = {"ids": ids, "anchor": anchor, "long_anchors": study.long_anchors,
                "threshold_pp": minimum_delta}
    paths = {name: report_path(root, name, fixed[name].variant) for name in ids}
    paths[anchor] = reference_root / anchor / "dev_epoch_015" / "report.json"
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return {**decision, "state": "missing_reports", "missing": missing}, None
    try:
        anchor_run = replace(fixed[ids[0]], id=anchor, variant=anchor, control="")
        anchor_key, anchor_subset, anchor_data, anchor_score = dev_result(paths[anchor], anchor_run)
        results = {name: dev_result(paths[name], fixed[name]) for name in ids}
        sources = {name: read_json(path) for name, path in paths.items()}
        source_classes = {name: supported_iou(report)[1] for name, report in sources.items()}
    except REPORT_ERRORS as exc:
        return {**decision, "state": "invalid_report", "reason": str(exc)}, None
    audited_data = results[ids[0]][2]
    if (not audited_data or (anchor_data is not None and anchor_data != audited_data)
            or (audit is not None and (anchor_key["manifest"], audited_data)
                != (audit["source_manifest_sha256"], audit["data_sha256"]))
            or any((key, subset, content) != (anchor_key, anchor_subset, audited_data)
                   for key, subset, content, _ in results.values())):
        return {**decision, "state": "protocol_mismatch"}, None

    scores = {anchor: anchor_score, **{name: result[3] for name, result in results.items()}}
    source = pair_class_deltas(contrast("source_dev_miou", ids, scores, anchor), source_classes)
    evidence = [source, *class_evidence(ids, anchor, sources, class_groups),
                held_evidence(root, study, sources, paths)]
    if ids == ("F08", "F09"):
        reverse = contrast("source_dev_reverse_alignment", ("F08",), scores, anchor)
        evidence.append(pair_class_deltas(reverse, source_classes))
    if ids == ("E01", "E02"):
        c1 = replace(fixed["T03"], id="C1", epochs=15)
        c1_path = reference_root / "C1" / "dev_epoch_015" / "report.json"
        evidence.append(factorial_evidence(c1_path, c1, dev_result, anchor_key, anchor_subset,
                                           audited_data, scores, source_classes))
    if external_manifests:
        evidence.extend(external_evidence(Path(path), root, reference_root, ids, anchor, sources,
                                          paths, historical_evidence_root)
                        for path in external_manifests)
    else:
        evidence.append({"metric": "external_dev_miou", "state": "not_configured"})
    material, qualifying = qualify_evidence(evidence, minimum_delta)
    transfer = [item for item in qualifying if item["metric"] in TRANSFER_METRICS]
    strongest = max(transfer or qualifying or material or [source],
                    key=lambda item: abs(item["signed_delta_pp"]))
    decision.update(delta_pp=source["signed_delta_pp"], priority_pp=abs(strongest["signed_delta_pp"]),
                    trigger=strongest["metric"] if qualifying else None, transfer_priority=bool(transfer),
                    evidence=evidence)

    long_paths = {name: report_path(root, name, fixed[name].variant, "eval") for name in study.long_anchors}
    missing = [str(path) for path in long_paths.values() if not path.is_file()]
    if missing:
        return {**decision, "state": "missing_long_anchor", "missing": missing}, None
    try:
        anchors = [dev_result(path, fixed[name], development=False) for name, path in long_paths.items()]
        if not all(key == anchor_key and content == audited_data for key, _, content, _ in anchors):
            raise ValueError("Long-run controls differ from the matched short-run protocol")
    except REPORT_ERRORS as exc:
        return {**decision, "state": "invalid_long_anchor", "reason": str(exc)}, None

    # Two cells may use different encoder sets by design. Require the shared CLIP
    # weights, source sample support, and all other protocol settings to match.
    common_key = {**anchor_key, "weights": {"clip": anchor_key["weights"]["clip"]}}
    state = "eligible" if qualifying else "uncorroborated" if material else "below_threshold"
    return {**decision, "state": state}, (common_key, anchor_subset, audited_data)


def promotion_candidates(root, reference_root, minimum_delta, fixed, dev_result, requested_ids=None, audit=None,
                         *, external_manifests=(), class_groups=None, historical_evidence_root=None):
    fixed = {run.id: run for run in fixed}
    eligible, decisions = [], []
    for unit in requested_units(requested_ids):
        resolved = [study_decision(study, root, reference_root, minimum_delta, fixed, dev_result, audit,
                                   external_manifests, class_groups, historical_evidence_root) for study in unit]
        if len(unit) == 1:
            decision, _ = resolved[0]
            if decision["state"] == "eligible":
                eligible.append((decision["priority_pp"], decision["ids"], decision))
            else:
                decisions.append(decision)
            continue

        ids = unit_ids(unit)
        cells = [decision for decision, _ in resolved]
        summary = {"ids": ids, "cells": cells, "threshold_pp": minimum_delta}
        identities = [identity for _, identity in resolved]
        if any(identity is None for identity in identities):
            decisions.append({**summary, "state": "incomplete_study"})
            continue
        if any(identity != identities[0] for identity in identities):
            decisions.append({**summary, "state": "protocol_mismatch"})
            continue
        candidates = [cell for cell in cells if cell["state"] == "eligible"]
        if not candidates:
            state = "uncorroborated" if any(cell["state"] == "uncorroborated" for cell in cells) else "below_threshold"
            decisions.append({**summary, "state": state})
            continue
        winner = sorted(candidates, key=lambda cell: (-cell["transfer_priority"], -cell["priority_pp"],
                                                        cell["ids"]))[0]
        # The visual encoder study needs both arms. The fusion-choice study has
        # three screening cells but allocates only one long run.
        promoted = ids if ids == ("D01", "D02") else winner["ids"]
        summary.update(promoted_ids=promoted, trigger=winner["trigger"], trigger_cell=winner["ids"],
                       priority_pp=winner["priority_pp"], transfer_priority=winner["transfer_priority"])
        eligible.append((summary["priority_pp"], promoted, summary))
    return eligible, decisions


def select_promotions(ranked, decisions, fixed, limit):
    fixed = {run.id: run for run in fixed}
    decisions = list(decisions)
    selected = []
    for _, ids, decision in sorted(ranked, key=lambda result: (-result[2]["transfer_priority"], -result[0])):
        if len(selected) + len(ids) > limit:
            decisions.append({**decision, "state": "budget"})
            continue
        selected.extend(replace(fixed[name], id=f"{name}-e40", epochs=40) for name in ids)
        decisions.append({**decision, "state": "selected"})
    return selected, decisions
