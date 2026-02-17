-- If the table is in public schema:
GRANT USAGE ON SCHEMA public TO sage;

GRANT SELECT, INSERT, UPDATE, DELETE
ON TABLE public.rag_chunks
TO sage;

-- If you want sequences too (needed for bigserial id):
GRANT USAGE, SELECT
ON ALL SEQUENCES IN SCHEMA public
TO sage;
