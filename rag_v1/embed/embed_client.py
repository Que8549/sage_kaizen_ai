import httpx

class EmbedClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=60.0)

    def embed(self, texts):
        payload = {
            "model": self.model,
            "input": texts
        }

        r = self.client.post(f"{self.base_url}/embeddings", json=payload)
        r.raise_for_status()

        data = r.json()
        out = [None] * len(texts)

        for item in data["data"]:
            out[item["index"]] = item["embedding"]

        return out
