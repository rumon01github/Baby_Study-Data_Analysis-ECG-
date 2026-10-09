"""Load the explicitly synchronized, latest annotation snapshot without offsets."""
import csv
import math
from pathlib import Path


def latest_annotations(root, trial):
    participant, trial_id = trial.split()
    path = Path(root) / 'annotations_latest' / f'{participant}_{trial_id}_annotations.csv'
    if not path.exists():
        return None
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'participant', 'trial', 'label', 'video_start_s', 'video_end_s'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing annotation columns in {path.name}')
        rows = list(reader)
    for row in rows:
        if (row['participant'], row['trial']) != (participant, trial_id):
            raise ValueError(f'Participant/trial mismatch in {path.name}')
        start, end = float(row['video_start_s']), float(row['video_end_s'])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f'Invalid interval in {path.name}')
    return rows


def apply_latest(root,data):
    if data.get('annotation_override'):return data
    latest=latest_annotations(root,data['trial'])
    return dict(data,annotations=latest,annotations_synced=True) if latest is not None else data
