"""Explicit task definitions, label mappings, and attrition tracking.

Prevents label leakage and arbitrary label assumptions. Ensures missing
or ambiguous (e.g. symptomatic) records are handled explicitly and tracked.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class WorkingSet:
    """Encapsulates a clean, task-specific working set with attrition audit."""
    task_name: str
    records: List[Dict[str, Any]]
    num_samples: int
    class_counts: Dict[int, int]
    class_proportions: Dict[int, float]
    attrition_log: Dict[str, int]

    def to_csv(self, output_path: str | Path) -> Path:
        """Write working set records to a CSV file."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self.records:
            out.write_text("", encoding="utf-8")
            return out

        fieldnames = list(self.records[0].keys())
        with open(out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        return out

    def get_uuids(self) -> List[str]:
        return [r["uuid"] for r in self.records]

    def get_labels(self) -> List[int]:
        return [r["label"] for r in self.records]

    def summary(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "num_samples": self.num_samples,
            "class_counts": self.class_counts,
            "class_proportions": self.class_proportions,
            "attrition_log": self.attrition_log,
        }


def build_working_set(
    rows: List[Dict[str, Any]],
    task: str = "covid_binary",
    cough_threshold: float = 0.8,
    require_audio: bool = True,
) -> WorkingSet:
    """Filter raw metadata into an explicitly labeled working set.

    Parameters
    ----------
    rows : list[dict]
        Parsed metadata rows from loader.load_raw_metadata.
    task : {"covid_binary", "expert_covid", "wet_dry"}, default="covid_binary"
        Target classification task.
    cough_threshold : float, default=0.8
        Threshold on cough_detected score (EPFL guideline).
    require_audio : bool, default=True
        Whether to require existing audio file on disk.

    Returns
    -------
    WorkingSet
        Filtered dataset with 'label' integer column and detailed attrition metrics.
    """
    total_input = len(rows)
    attrition = {
        "total_input_records": total_input,
        "dropped_missing_audio": 0,
        "dropped_low_cough_score": 0,
        "dropped_missing_label": 0,
        "dropped_symptomatic_ambiguous": 0,
        "final_working_set": 0,
    }

    filtered_rows: List[Dict[str, Any]] = []

    for r in rows:
        # Step 1: Check audio availability if required
        if require_audio and not r.get("has_audio", False):
            attrition["dropped_missing_audio"] += 1
            continue

        # Step 2: Check cough detection score threshold
        cd = r.get("cough_detected")
        if cd is None or cd == "" or str(cd).lower() == "nan":
            attrition["dropped_low_cough_score"] += 1
            continue
        try:
            score = float(cd)
            if score < cough_threshold:
                attrition["dropped_low_cough_score"] += 1
                continue
        except (ValueError, TypeError):
            attrition["dropped_low_cough_score"] += 1
            continue

        # Step 3: Task-specific label extraction and validation
        row_copy = dict(r)

        if task == "covid_binary":
            status = row_copy.get("status")
            if not status or str(status).strip() == "" or str(status).strip().lower() == "nan":
                attrition["dropped_missing_label"] += 1
                continue
            status_clean = str(status).strip()
            if status_clean == "symptomatic":
                attrition["dropped_symptomatic_ambiguous"] += 1
                continue
            if status_clean == "healthy":
                row_copy["label"] = 0
            elif status_clean == "COVID-19":
                row_copy["label"] = 1
            else:
                attrition["dropped_missing_label"] += 1
                continue

        elif task == "expert_covid":
            diag = row_copy.get("diagnosis_1")
            if not diag or str(diag).strip() == "" or str(diag).strip().lower() == "nan":
                attrition["dropped_missing_label"] += 1
                continue
            diag_clean = str(diag).strip()
            if diag_clean == "healthy_cough":
                row_copy["label"] = 0
            elif diag_clean == "COVID-19":
                row_copy["label"] = 1
            else:
                attrition["dropped_missing_label"] += 1
                continue

        elif task == "wet_dry":
            ctype = row_copy.get("cough_type_1")
            if not ctype or str(ctype).strip() == "" or str(ctype).strip().lower() == "nan":
                attrition["dropped_missing_label"] += 1
                continue
            ctype_clean = str(ctype).strip()
            if ctype_clean == "dry":
                row_copy["label"] = 0
            elif ctype_clean == "wet":
                row_copy["label"] = 1
            else:
                attrition["dropped_missing_label"] += 1
                continue

        else:
            raise ValueError(
                f"Unknown classification task '{task}'. Supported tasks: "
                f"'covid_binary', 'expert_covid', 'wet_dry'."
            )

        filtered_rows.append(row_copy)

    final_count = len(filtered_rows)
    attrition["final_working_set"] = final_count

    # Calculate class counts and proportions
    class_counts: Dict[int, int] = {}
    for r in filtered_rows:
        lbl = int(r["label"])
        class_counts[lbl] = class_counts.get(lbl, 0) + 1

    class_props = {
        lbl: (count / final_count) if final_count > 0 else 0.0
        for lbl, count in class_counts.items()
    }

    return WorkingSet(
        task_name=task,
        records=filtered_rows,
        num_samples=final_count,
        class_counts=class_counts,
        class_proportions=class_props,
        attrition_log=attrition,
    )
