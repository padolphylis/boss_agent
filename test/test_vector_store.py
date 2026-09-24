"""Qdrant 职位索引回归：使用本地临时库和伪造 Embedding，不访问外部服务。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient

from vector_store import QdrantJobStore, VectorStoreUnavailable


class QdrantJobStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.client = QdrantClient(path=self.directory.name)
        self.store = QdrantJobStore(client=self.client)
        self.config_patch = patch(
            "vector_store.cfg",
            side_effect=lambda key, default="": {
                "qdrant_enabled": "true",
                "qdrant_url": "",
            }.get(key, default),
        )
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(self.store.close)

    @staticmethod
    def _embedding(texts, **_kwargs):
        return [[1.0, 0.0] if "后端" in text else [0.0, 1.0] for text in texts]

    def test_incremental_upsert_search_and_persistence(self):
        cards = [
            {
                "encryptJobId": "backend",
                "jobName": "后端工程师",
                "postDescription": "负责服务开发",
                "cityName": "杭州",
                "salaryDesc": "20-30K",
            },
            {
                "encryptJobId": "designer",
                "jobName": "产品设计师",
                "postDescription": "负责界面设计",
                "cityName": "上海",
                "salaryDesc": "15-25K",
            },
        ]
        with patch("vector_store._model_name", return_value="test-model"), patch(
            "vector_store._collection_name", return_value="test-jobs"
        ), patch("analysis_work_content.embed_texts", side_effect=self._embedding) as embed:
            self.assertEqual(self.store.upsert_cards(cards), 2)
            self.assertEqual(self.store.upsert_cards(cards), 0)
            self.assertEqual(embed.call_count, 1)

            with patch(
                "analysis_work_content._query_vector", return_value=[1.0, 0.0]
            ):
                results = self.store.search(cards, "后端工程师", threshold=0)
            self.assertEqual([index for index, _ in results], [0, 1])
            self.assertGreater(results[0][1], results[1][1])

            # 搜索结果只能指向调用方传入的本批职位。
            with patch(
                "analysis_work_content._query_vector", return_value=[0.0, 1.0]
            ):
                subset = self.store.search(cards[:1], "界面设计", threshold=0)
            self.assertEqual([index for index, _ in subset], [0])

        self.store.close()
        self.client = QdrantClient(path=self.directory.name)
        self.store = QdrantJobStore(client=self.client)
        with patch("vector_store.cfg", side_effect=lambda key, default="": {
            "qdrant_enabled": "true",
            "qdrant_url": "",
        }.get(key, default)), patch(
            "vector_store._model_name", return_value="test-model"
        ), patch("vector_store._collection_name", return_value="test-jobs"), patch(
            "analysis_work_content._query_vector", return_value=[1.0, 0.0]
        ):
            persisted = self.store.search(cards, "后端工程师", threshold=0)
        self.assertEqual([index for index, _ in persisted], [0, 1])

    def test_content_change_reindexes_job(self):
        card = {
            "encryptJobId": "backend",
            "jobName": "后端工程师",
            "postDescription": "负责服务开发",
        }
        changed = dict(card, postDescription="负责分布式服务开发")
        with patch("vector_store._model_name", return_value="test-model"), patch(
            "vector_store._collection_name", return_value="test-jobs"
        ), patch("analysis_work_content.embed_texts", side_effect=self._embedding) as embed:
            self.assertEqual(self.store.upsert_cards([card]), 1)
            self.assertEqual(self.store.upsert_cards([changed]), 1)
            self.assertEqual(embed.call_count, 2)


class MatchFallbackTests(unittest.TestCase):
    def test_qdrant_failure_uses_existing_cosine_matcher(self):
        from matcher import match_card_batch

        card = {
            "encryptJobId": "backend",
            "jobName": "后端工程师",
            "postDescription": "<p>负责服务开发</p>",
        }
        from vector_store import vector_store

        vector_store.enabled_override = True
        self.addCleanup(lambda: setattr(vector_store, "enabled_override", None))
        with patch(
            "vector_store.vector_store.match_cards",
            side_effect=VectorStoreUnavailable("Qdrant down"),
        ), patch("matcher.match_jobs", return_value=[(0, 0.91)]) as matcher:
            results = match_card_batch([card], "后端工程师", [])

        matcher.assert_called_once_with(["负责服务开发"], "后端工程师", threshold=0.3)
        self.assertEqual(results, [{"job_card": card, "score": 0.91}])


if __name__ == "__main__":
    unittest.main()
