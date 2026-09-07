CREATE TABLE IF NOT EXISTS events_false_positive_archive (
    event_id TEXT PRIMARY KEY,
    event_type TEXT,
    source TEXT,
    created_at TEXT,
    received_at TEXT,
    blockage_status TEXT,
    coverage_percent REAL,
    occlusion_pct REAL,
    roi_profile TEXT,
    lat REAL,
    lon REAL,
    has_image BOOLEAN,
    image_path TEXT,
    raw_payload TEXT,
    dismissed_at TEXT,
    dismiss_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_archive_dismissed_at ON events_false_positive_archive(dismissed_at DESC);
