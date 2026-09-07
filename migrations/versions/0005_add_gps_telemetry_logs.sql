CREATE TABLE IF NOT EXISTS telemetry_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_code TEXT NOT NULL,
    client_ip TEXT,
    lat REAL NOT NULL,
    lng REAL NOT NULL,
    speed_mps REAL,
    sequence INTEGER,
    device_timestamp REAL,
    received_at TEXT NOT NULL,
    nearest_drain_id INTEGER,
    distance_m REAL
);

CREATE INDEX IF NOT EXISTS idx_telemetry_logs_vehicle ON telemetry_logs(vehicle_code, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_logs_received_at ON telemetry_logs(received_at DESC);

ALTER TABLE vehicle_telemetry_states ADD COLUMN lat REAL;
ALTER TABLE vehicle_telemetry_states ADD COLUMN lng REAL;
ALTER TABLE vehicle_telemetry_states ADD COLUMN speed_mps REAL;
ALTER TABLE vehicle_telemetry_states ADD COLUMN client_ip TEXT;
