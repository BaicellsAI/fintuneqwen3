import unittest
import os
import json
from finetune_sft import load_data, SimpleDataset, tokenizer

class TestFinetuneSFT(unittest.TestCase):
    def setUp(self):
        self.test_data_path = "test_data.json"
        test_data = [
            {
                "instruction": "问：什么是中医？",
                "input": "请简要说明。",
                "output": "中医是中国传统医学体系。"
            }
        ]
        with open(self.test_data_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False)

    def tearDown(self):
        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)

    def test_load_data(self):
        dataset = load_data(self.test_data_path, tokenizer)
        self.assertEqual(len(dataset), 1)
        self.assertIn("input_ids", dataset[0])
        self.assertIn("labels", dataset[0])

    def test_simple_dataset(self):
        dataset = load_data(self.test_data_path, tokenizer)
        ds = SimpleDataset(dataset)
        self.assertEqual(len(ds), 1)
        item = ds[0]
        self.assertIn("input_ids", item)
        self.assertIn("labels", item)

if __name__ == "__main__":
    unittest.main()
if __name__ == "__main__":
    unittest.main()
