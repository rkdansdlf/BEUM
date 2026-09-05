CREATE TABLE IF NOT EXISTS vehicle_telemetry_states (
    vehicle_code TEXT PRIMARY KEY,
    last_sequence INTEGER NOT NULL,
    last_timestamp REAL NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drain_id INTEGER NOT NULL,
    vehicle_code TEXT NOT NULL,
    status TEXT NOT NULL,
    occlusion_pct REAL,
    reason_code TEXT,
    confidence REAL,
    lat REAL NOT NULL,
    lng REAL NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS drains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lat REAL NOT NULL,
    lng REAL NOT NULL,
    description TEXT
);
