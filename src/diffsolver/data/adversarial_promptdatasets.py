# code for i2p dataset and adversarial prompt datasets
import os

import yaml
from datasets import load_dataset


def get_current_path():
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    print(current_file_path)
    return current_file_path


def get_nudity_concept_dataset():
    config_path = os.path.join(get_current_path(), "..", "..", "..", "configs", "nudity_concepts.yaml")
    with open(config_path, "r") as f:
        concepts = yaml.load(f, Loader=yaml.FullLoader)
    return concepts


def get_ip_characters():
    # TODO: use Hulk as sample need to expand it to more concepts and takes average score
    config_path = os.path.join(get_current_path(), "..", "..", "..", "configs", "ip_characters.yaml")
    with open(config_path, "r") as f:
        concepts = yaml.load(f, Loader=yaml.FullLoader)
    return concepts


def get_i2p_dataset(nudity_threshold: int = 50):
    data = load_dataset("AIML-TUDA/i2p", split="train")
    nudity_samples = [sample["prompt"] for sample in data if sample["nudity_percentage"] >= nudity_threshold]
    return nudity_samples


def get_mma_dataset(adv_prompt: bool = True):
    data = load_dataset("YijunYang280/MMA-Diffusion-NSFW-adv-prompts-benchmark", split="train")
    if adv_prompt:
        samples = [sample["adv_prompt"] for sample in data]
    else:
        samples = [sample["target_prompt"] for sample in data]
    return samples


def get_p4d_dataset():
    data = load_dataset("joycenerd/p4d", split="train")
    nudity_samples = [sample["prompt"] for sample in data]
    return nudity_samples


def get_ring_a_bell_dataset(threshold: int = 3, length: int = 16):
    path = f"baselines/ring_a_bell/data/InvPrompt/Nudity_{threshold}_length_{length}.csv"
    assert os.path.exists(path), f"File {path} does not exist"
    dataset = load_dataset("csv", data_files=path, split="train", column_names=["prompt"])
    samples = [sample["prompt"] for sample in dataset]
    return samples


def get_unlearn_diff_atk_dataset():
    # TODO: implement this
    pass
