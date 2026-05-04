from __future__ import annotations


from input_guard import sanitize_chunk
from rag_v1.config.rag_settings import RetrievedChunk
from rag_v1.runtime.rag_pipeline import RagPipeline


def _prepend_context(content: str | list, prefix: str) -> str | list:
    """Prepend a text prefix to an OpenAI message content value.

    For plain-string content the prefix is concatenated directly.
    For multimodal content (a list of content parts) the prefix is inserted
    as a leading text part so base64 audio/image data is never
    string-formatted into the message (which would tokenise raw bytes as
    millions of tokens).
    """
    if isinstance(content, list):
        return [{"type": "text", "text": prefix}] + list(content)
    return prefix + content


class RagInjector:

    def __init__(self, cfg):
        self.pipeline = RagPipeline(cfg)

    def maybe_inject(
        self,
        messages: list[dict],
        user_text: str,
        brain: str,
        enabled: bool = True,
        top_k: int | None = None,
    ) -> tuple[list[dict], list[RetrievedChunk]]:
        """Inject retrieved context into the last user message and return sources.

        RAG context belongs in the user turn, not the system message.  The system
        message is reserved for persona and instructions; injecting data there
        confuses the model about what is an instruction vs. ephemeral context.

        The augmented user message looks like:

            <context>
            [source_id#chunk3 | score=0.912]
            ...retrieved text...

            ---

            [source_id#chunk7 | score=0.843]
            ...retrieved text...
            </context>

            {original user question}

        Returns:
            (messages, kept_chunks) — messages is a new list (originals untouched);
            kept_chunks is empty when RAG is disabled or no chunks pass the filter.
        """
        if not enabled:
            return list(messages), []

        k = top_k if top_k is not None else (4 if brain == "FAST" else 10)
        ctx, chunks = self.pipeline.build_context(user_text, k)
        ctx = sanitize_chunk(ctx.strip(), max_chars=None)

        if not ctx:
            return list(messages), []

        out = list(messages)

        # Find the last user-role message and prepend the context block to it.
        for i in reversed(range(len(out))):
            if out[i].get("role") == "user":
                prefix = f"<context>\n{ctx}\n</context>\n\n"
                out[i] = {**out[i], "content": _prepend_context(out[i]["content"], prefix)}
                break

        return out, chunks
