"""The committed lambda-study CSVs must still select the lambdas used in main.py."""

import unittest

import main
from experiments import plot_lambda_sensitivity as study


class LambdaSelectionTests(unittest.TestCase):
    def test_selection_matches_main_protocol(self):
        for dataset in ("orl", "yaleb"):
            with self.subTest(dataset=dataset):
                summary = study.summarize(study.RESULTS_DIR / f"{dataset}_raw.csv")
                selection = study.select_lambda(summary)
                self.assertEqual(selection["chosen_lambda"],
                                 main.DATASET_CONFIGS[dataset]["rnmf"]["lam"])
                self.assertFalse(selection["best_on_grid_edge"])


if __name__ == "__main__":
    unittest.main()
