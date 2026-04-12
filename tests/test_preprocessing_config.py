from __future__ import annotations

import unittest

from pipeline.train.preprocessing import iter_preprocessing_tasks


class PreprocessingConfigTest(unittest.TestCase):
    def test_iter_preprocessing_tasks_uses_custom_param_grid(self) -> None:
        tasks = list(
            iter_preprocessing_tasks(
                selected_ids=["denoise"],
                include_combinations=False,
                param_grids={
                    "denoise": {
                        "kernel_size": [(3, 3)],
                        "sigma": [1],
                    }
                },
            )
        )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].preproc_id, "denoise")
        self.assertEqual(tasks[0].params["kernel_size"], (3, 3))
        self.assertEqual(tasks[0].params["sigma"], 1)


if __name__ == "__main__":
    unittest.main()
