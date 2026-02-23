-- CREATE SCHEMA IF NOT EXISTS feedback;

-- One row per rated assistant response.
-- id = the UUID assigned to the message in st.session_state (stable across the session).
CREATE TABLE IF NOT EXISTS public.ratings (
  id               uuid        PRIMARY KEY,
  ts_utc           timestamptz NOT NULL DEFAULT now(),
  brain            text        NOT NULL CHECK (brain IN ('FAST', 'ARCHITECT')),
  model_id         text,
  endpoint         text,
  route_score      double precision,
  route_reasons    jsonb,
  templates        jsonb,
  prompt_messages  jsonb       NOT NULL,   -- exact chat messages payload sent to model
  assistant_text   text        NOT NULL,   -- the model's final response text
  thumb            smallint    NOT NULL CHECK (thumb IN (-1, 1)),
  notes            text
);

CREATE INDEX IF NOT EXISTS fb_ratings_ts_idx    ON public.ratings(ts_utc);
CREATE INDEX IF NOT EXISTS fb_ratings_brain_idx ON public.ratings(brain);
CREATE INDEX IF NOT EXISTS fb_ratings_thumb_idx ON public.ratings(thumb);
