from rag_v1.runtime.rag_pipeline import RagPipeline

class RagInjector:

    def __init__(self, cfg):
        self.pipeline = RagPipeline(cfg)

    def maybe_inject(self, messages, user_text, brain, enabled=True):

        if not enabled:
            return messages

        top_k = 4 if brain == "FAST" else 10
        ctx = self.pipeline.build_context(user_text, top_k).strip()

        if not ctx:
            return messages

        rag_block = f"Retrieved context (RAG):\n{ctx}"

        out = list(messages)

        if out and out[0].get("role") == "system":
            out[0]["content"] += "\n\n" + rag_block
        else:
            out.insert(0, {"role": "system", "content": rag_block})

        return out
