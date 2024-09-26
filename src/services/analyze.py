from fastapi import Depends
from pipeline.customIn_ference_pipeline import CustomInferencePipeline

from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import List, Dict, Any


class AnalyzeService:
    def __init__(
        self,
        pipeline: CustomInferencePipeline = Depends(CustomInferencePipeline),
    ) -> None:
        self.pipeline = pipeline

    async def analyze_review(self, review):
        result = self.pipeline(review.text)
        return result

    async def analyze_batch(self, batch) -> list:
        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                executor, self.batch_process, self.pipeline, batch.reviews
            )
        return results


    @staticmethod
    def batch_process(
        pipeline: CustomInferencePipeline, texts: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of texts using the custom inference pipeline.

        Args:
            pipeline (CustomInferencePipeline): The initialized pipeline.
            texts (List[str]): A list of texts to process.

        Returns:
            List[Dict[str, Any]]: A list of processed results.
        """
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(pipeline, texts))
        return results
