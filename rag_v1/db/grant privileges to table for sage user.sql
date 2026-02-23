GRANT SELECT, INSERT ON public.ratings TO sage;
-- Also allow sage to create objects it may need later:
GRANT USAGE ON SCHEMA public TO sage;