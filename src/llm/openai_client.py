from __future__ import annotations

from typing import List, Optional

from openai import OpenAI

from src.config.settings import Settings
import math


class OpenAIClient:
    def __init__(self, cfg: Optional[Settings] = None) -> None:
        self.cfg = cfg or Settings.load()
        self.client = OpenAI(api_key=self.cfg.openai_api_key)

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Returns embeddings aligned with input texts.
        Batches requests to avoid token/request limits and logs progress.
        """
    

        all_embeddings: List[List[float]] = []
        total = len(texts)
        total_batches = math.ceil(total / batch_size) if total else 0

        for idx, start in enumerate(range(0, total, batch_size), start=1):
            batch = texts[start : start + batch_size]
            print(f"[embeddings] batch {idx}/{total_batches} | items={len(batch)}", flush=True)

            resp = self.client.embeddings.create(
                model=self.cfg.embed_model,
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in resp.data])

        return all_embeddings


    def chat(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""
