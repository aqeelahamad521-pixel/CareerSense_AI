"""
Regression test suite for AI Career Guide Chatbot integration.
Validates:
1. Module imports and dependency status inspection
2. Candidate path resolution
3. Dataset schema, cleanliness (no duplicate records), and record counts
4. Semantic context retrieval logic and out-of-scope blocking
5. Safe response generation pipeline structure and fallbacks
"""
import unittest
import sys
from pathlib import Path
import json
from unittest.mock import MagicMock, patch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from ai_engine.chatbot_integration import (
    get_dependencies_status,
    resolve_candidate_path,
    DATASET_CANDIDATES,
    retrieve_context,
    generate_chatbot_response,
)


class TestChatbotIntegration(unittest.TestCase):
    def test_dependency_status_structure(self):
        status = get_dependencies_status()
        self.assertIsInstance(status, dict)
        self.assertIn("torch_available", status)
        self.assertIn("rag_ready", status)
        self.assertIn("model_ready", status)
        self.assertIn("dataset_path", status)
        self.assertIn("cache_path", status)

    def test_dataset_resolution_and_cleanliness(self):
        dataset_path = resolve_candidate_path(DATASET_CANDIDATES)
        self.assertIsNotNone(dataset_path, "Dataset file should be resolved")
        self.assertTrue(dataset_path.exists())

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 4960, "Dataset should have 4960 records after duplicate cleanup")

        # Verify no exact duplicate records
        seen = set()
        duplicates = 0
        for item in data:
            key = json.dumps(item, sort_keys=True)
            if key in seen:
                duplicates += 1
            seen.add(key)
        self.assertEqual(duplicates, 0, "Dataset must not contain any exact duplicate records")

    def test_out_of_scope_query_handling(self):
        # Queries clearly outside Sri Lankan IT scope (< 0.20 similarity)
        out_of_scope_query = "recipe for baking sourdough bread at home"
        res = generate_chatbot_response(out_of_scope_query)
        self.assertIsInstance(res, dict)
        self.assertEqual(res.get("source"), "out_of_scope")
        self.assertIn("I do not have information regarding that", res.get("response", ""))
        self.assertLess(res.get("similarity", 1.0), 0.20)

    def test_in_scope_retrieval(self):
        query = "What is the salary of a Software Engineering Intern in Sri Lanka?"
        context, score = retrieve_context(query)
        self.assertIsInstance(context, str)
        self.assertGreater(score, 0.20, "In-scope query should score above threshold")
        self.assertTrue(len(context) > 0, "Retrieved context should not be empty")

    @patch("ai_engine.chatbot_integration.load_local_model")
    def test_generate_chatbot_response_rag_fallback(self, mock_load_model):
        # Test fallback when local model weights are not loaded in memory
        mock_load_model.return_value = (None, None)
        query = "What skills are needed for a software engineer in Colombo?"
        res = generate_chatbot_response(query)
        self.assertIsInstance(res, dict)
        self.assertEqual(res["source"], "rag_direct")
        self.assertIn("Sri Lankan IT career knowledge base", res["response"])
        self.assertGreaterEqual(res["similarity"], 0.20)

    @patch("ai_engine.chatbot_integration.load_local_model")
    def test_generate_chatbot_response_qwen_lora_mocked(self, mock_load_model):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value.to.return_value = {"input_ids": [1, 2, 3]}
        mock_tokenizer.decode.return_value = "Assistant: Typical intern salary is 40k LKR."
        mock_tokenizer.eos_token_id = 0

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = [[1, 2, 3]]

        mock_load_model.return_value = (mock_tokenizer, mock_model)

        query = "What is the salary for software engineering interns in Sri Lanka?"
        res = generate_chatbot_response(query)
        self.assertEqual(res["source"], "qwen_lora")
        self.assertIn("Typical intern salary is 40k LKR.", res["response"])


if __name__ == "__main__":
    unittest.main()
