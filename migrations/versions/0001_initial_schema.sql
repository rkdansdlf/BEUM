CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT,
    source TEXT,
    created_at TEXT,
    received_at TEXT,
    blockage_status TEXT,
    coverage_percent REAL,
    roi_profile TEXT,
    lat REAL,
    lon REAL,
    has_image BOOLEAN,
    image_path TEXT,
    raw_payload TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_received_at ON events(received_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_blockage_status ON events(blockage_status);
