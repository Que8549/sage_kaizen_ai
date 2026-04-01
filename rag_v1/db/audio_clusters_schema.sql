-- rag_v1/db/audio_clusters_schema.sql
--
-- KMeans cluster assignments for audio_embeddings.
-- Enables "songs that sound similar to each other" and artist/era grouping.
--
-- Run once:
--   psql -U sage -d sage_kaizen -f rag_v1/db/audio_clusters_schema.sql
--
-- Populated by: python -m rag_v1.media.audio_cluster (or via media_ingest --cluster)
-- Re-run at any time to rebuild clusters (DELETE + re-insert is idempotent).

CREATE TABLE IF NOT EXISTS audio_clusters (
    media_id      uuid        PRIMARY KEY REFERENCES media_files (media_id) ON DELETE CASCADE,
    cluster_id    int         NOT NULL,
    cluster_label text,                      -- e.g. "Cluster 12" or auto-generated description
    clustered_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS audio_clusters_cluster_id
    ON audio_clusters (cluster_id);

-- View: cluster members enriched with file metadata
CREATE OR REPLACE VIEW audio_cluster_view AS
SELECT
    ac.cluster_id,
    ac.cluster_label,
    mf.media_id,
    mf.file_path,
    mf.metadata->>'title'      AS title,
    mf.metadata->>'artist'     AS artist,
    (mf.metadata->>'bpm')::float AS bpm,
    mf.metadata->>'key'        AS key,
    (mf.metadata->>'has_vocals')::bool  AS has_vocals,
    (mf.metadata->>'is_explicit')::bool AS is_explicit,
    mf.ingested_at
FROM audio_clusters ac
JOIN media_files mf ON mf.media_id = ac.media_id
ORDER BY ac.cluster_id, mf.file_path;

GRANT ALL ON TABLE audio_clusters TO sage;
