import unittest

from diffsolver.data.promptdataset import PromptImageDataset


class PromptImageDatasetMock(PromptImageDataset):
    def prepare_metadata(self):
        # Override prepare_metadata to bypass it for testing purposes
        pass


META_PATH = "./datasets/gcc3m/Validation_GCC-1.1.0-Validation.tsv"
DECOMETA_PATH = "./configs/concept_small.yaml"


class TestPromptImageDataset(unittest.TestCase):
    def test_load_and_merge_metadata(self):
        # Instantiate and test
        size = 10
        dataset = PromptImageDatasetMock(
            metadata=META_PATH,
            deconceptmeta=DECOMETA_PATH,
            pipe=None,
            num_inference_steps=10,
            save_dir="/tmp",
            seed=42,
            device="cpu",
            size=size,
            concept=None,
            neutral_concept=None,
            only_deconcept_latent=False,
        )

        dataset._load_and_merge_metadata()

        # Validate the merged DataFrame
        self.assertIsNotNone(dataset.df)
        self.assertEqual(len(dataset.df), size * 2)

        target_list = [1 for _ in range(size * 2)]
        target_list[1::2] = [0 for _ in range(size)]
        self.assertEqual(dataset.df["value"].tolist(), target_list)


if __name__ == "__main__":
    unittest.main()
