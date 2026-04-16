-- Table: public.image_embeddings

-- DROP TABLE IF EXISTS public.image_embeddings;

CREATE TABLE IF NOT EXISTS public.image_embeddings
(
    embed_id bigint NOT NULL DEFAULT nextval('image_embeddings_embed_id_seq'::regclass),
    media_id uuid NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT image_embeddings_pkey PRIMARY KEY (embed_id),
    CONSTRAINT image_embeddings_media_id_fkey FOREIGN KEY (media_id)
        REFERENCES public.media_files (media_id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE CASCADE
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.image_embeddings
    OWNER to postgres;

GRANT ALL ON TABLE public.image_embeddings TO postgres;

GRANT ALL ON TABLE public.image_embeddings TO sage;
-- Index: image_embed_hnsw

-- DROP INDEX IF EXISTS public.image_embed_hnsw;

CREATE INDEX IF NOT EXISTS image_embed_hnsw
    ON public.image_embeddings USING hnsw
    (embedding vector_cosine_ops)
    TABLESPACE pg_default;
-- Index: image_embed_media_id

-- DROP INDEX IF EXISTS public.image_embed_media_id;

CREATE INDEX IF NOT EXISTS image_embed_media_id
    ON public.image_embeddings USING btree
    (media_id ASC NULLS LAST)
    WITH (fillfactor=100, deduplicate_items=True)
    TABLESPACE pg_default;