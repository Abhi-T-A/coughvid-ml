"""Dataset ingestion, metadata schema validation, and leak-free splitting module."""

from coughvid.data.loader import (
    MetadataAudit,
    MetadataRecord,
    audit_metadata,
    load_raw_metadata,
    resolve_audio_path_for_uuid,
)
from coughvid.data.splitters import (
    SplitAuditReport,
    audit_leakage_and_splits,
    create_cross_validation_folds,
    create_holdout_split,
    save_splits_manifests,
)
from coughvid.data.tasks import (
    WorkingSet,
    build_working_set,
)

__all__ = [
    "MetadataRecord",
    "MetadataAudit",
    "load_raw_metadata",
    "audit_metadata",
    "resolve_audio_path_for_uuid",
    "WorkingSet",
    "build_working_set",
    "create_holdout_split",
    "create_cross_validation_folds",
    "audit_leakage_and_splits",
    "save_splits_manifests",
    "SplitAuditReport",
]
