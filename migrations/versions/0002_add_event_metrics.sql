-- SQLite has no ADD COLUMN IF NOT EXISTS. The migration runner checks the
-- existing table definition before executing each ALTER statement.
ALTER TABLE events ADD COLUMN coverage_percent REAL;
ALTER TABLE events ADD COLUMN occlusion_pct REAL;

-- Preserve historical event semantics for rows written before occlusion_pct
-- existed. New writes also populate this value explicitly.
UPDATE events
SET occlusion_pct = coverage_percent
WHERE occlusion_pct IS NULL AND coverage_percent IS NOT NULL;
