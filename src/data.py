"""
Dataset loading and prompt construction for text classification.

Supports AGNews and other classification datasets used in the paper.
"""

from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass


# Dataset configurations from Table 1 in the paper
DATASET_CONFIGS = {
    "agnews": {
        "hf_name": "ag_news",
        "label_names": ["World", "Sports", "Business", "Technology"],
        "label_column": "label",
        "text_column": "text",
        "prompt_template": (
            "Classify the news articles into the categories of "
            "World, Sports, Business, and Technology.\n"
        ),
        "example_template": "Article: {text} Answer: {label}\n",
        "query_template": "Article: {text} Answer:",
    },
    "sst2": {
        "hf_name": "glue",
        "hf_config": "sst2",
        "label_names": ["Negative", "Positive"],
        "label_column": "label",
        "text_column": "sentence",
        "prompt_template": "",
        "example_template": "Review: {text} Sentiment: {label}\n",
        "query_template": "Review: {text} Sentiment:",
    },
    "trec": {
        "hf_name": "trec",
        "label_names": ["Description", "Entity", "Abbreviation", "Person", "Number", "Location"],
        "label_column": "coarse_label",
        "text_column": "text",
        "prompt_template": (
            "Classify the questions based on whether their answer type is a "
            "Number, Location, Person, Description, Entity, or Abbreviation.\n"
        ),
        "example_template": "Question: {text} Answer Type: {label}\n",
        "query_template": "Question: {text} Answer Type:",
    },
    "dbpedia": {
        "hf_name": "dbpedia_14",
        "label_names": [
            "Company", "School", "Artist", "Athlete", "Politician",
            "Transportation", "Building", "Nature", "Village", "Animal",
            "Plant", "Album", "Film", "Book"
        ],
        "label_column": "label",
        "text_column": "content",
        "prompt_template": (
            "Classify the documents based on whether they are about a "
            "Company, School, Artist, Athlete, Politician, Transportation, "
            "Building, Nature, Village, Animal, Plant, Album, Film, or Book.\n"
        ),
        "example_template": "Article: {text} Answer: {label}\n",
        "query_template": "Article: {text} Answer:",
    },
}


@dataclass
class ClassificationDataset:
    """Container for classification dataset with prompts."""

    name: str
    label_names: List[str]
    train_texts: List[str]
    train_labels: List[int]
    test_texts: List[str]
    test_labels: List[int]
    prompt_template: str
    example_template: str
    query_template: str

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    def get_few_shot_preface(
        self,
        n_shots: int,
        shot_indices: Optional[List[int]] = None,
        seed: int = 42,
    ) -> str:
        """
        Construct the preface with n few-shot examples.

        Args:
            n_shots: Number of examples to include (0 for zero-shot)
            shot_indices: Specific indices to use (if None, sample randomly)
            seed: Random seed for sampling

        Returns:
            Preface string with instructions and examples
        """
        preface = self.prompt_template

        if n_shots > 0:
            if shot_indices is None:
                rng = np.random.RandomState(seed)
                shot_indices = rng.choice(len(self.train_texts), n_shots, replace=False)

            for idx in shot_indices:
                text = self.train_texts[idx]
                label = self.label_names[self.train_labels[idx]]
                preface += self.example_template.format(text=text, label=label)

        return preface

    def build_prompt(self, preface: str, query_text: str) -> str:
        """Build full prompt from preface and query text."""
        return preface + self.query_template.format(text=query_text)

    def build_prompts_for_split(
        self,
        preface: str,
        split: str = "test",
    ) -> List[str]:
        """Build prompts for all samples in a split."""
        texts = self.test_texts if split == "test" else self.train_texts
        return [self.build_prompt(preface, text) for text in texts]


def load_agnews(
    n_train: int = 600,
    n_test: int = 1000,
    seed: int = 42,
) -> ClassificationDataset:
    """
    Load AGNews dataset with subsampling as in the paper.

    Args:
        n_train: Number of training samples for calibration
        n_test: Number of test samples for evaluation
        seed: Random seed for subsampling

    Returns:
        ClassificationDataset object
    """
    config = DATASET_CONFIGS["agnews"]

    # Load from Hugging Face
    dataset = load_dataset(config["hf_name"])

    rng = np.random.RandomState(seed)

    # Subsample train set
    train_data = dataset["train"]
    train_indices = rng.choice(len(train_data), n_train, replace=False)
    train_texts = [train_data[int(i)][config["text_column"]] for i in train_indices]
    train_labels = [train_data[int(i)][config["label_column"]] for i in train_indices]

    # Subsample test set
    test_data = dataset["test"]
    test_indices = rng.choice(len(test_data), n_test, replace=False)
    test_texts = [test_data[int(i)][config["text_column"]] for i in test_indices]
    test_labels = [test_data[int(i)][config["label_column"]] for i in test_indices]

    return ClassificationDataset(
        name="agnews",
        label_names=config["label_names"],
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        prompt_template=config["prompt_template"],
        example_template=config["example_template"],
        query_template=config["query_template"],
    )


def load_dataset_by_name(
    name: str,
    n_train: int = 600,
    n_test: int = 1000,
    seed: int = 42,
) -> ClassificationDataset:
    """
    Load a dataset by name.

    Args:
        name: Dataset name ('agnews', 'sst2', 'trec', 'dbpedia')
        n_train: Number of training samples
        n_test: Number of test samples
        seed: Random seed

    Returns:
        ClassificationDataset object
    """
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[name]

    # Load from Hugging Face
    if "hf_config" in config:
        dataset = load_dataset(config["hf_name"], config["hf_config"])
    else:
        dataset = load_dataset(config["hf_name"])

    rng = np.random.RandomState(seed)

    # Get train/test splits (handle different split names)
    if "train" in dataset:
        train_data = dataset["train"]
    else:
        raise ValueError(f"No train split found in {name}")

    if "test" in dataset:
        test_data = dataset["test"]
    elif "validation" in dataset:
        test_data = dataset["validation"]
    else:
        raise ValueError(f"No test/validation split found in {name}")

    # Subsample
    n_train = min(n_train, len(train_data))
    n_test = min(n_test, len(test_data))

    train_indices = rng.choice(len(train_data), n_train, replace=False)
    test_indices = rng.choice(len(test_data), n_test, replace=False)

    train_texts = [train_data[int(i)][config["text_column"]] for i in train_indices]
    train_labels = [train_data[int(i)][config["label_column"]] for i in train_indices]
    test_texts = [test_data[int(i)][config["text_column"]] for i in test_indices]
    test_labels = [test_data[int(i)][config["label_column"]] for i in test_indices]

    return ClassificationDataset(
        name=name,
        label_names=config["label_names"],
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        prompt_template=config["prompt_template"],
        example_template=config["example_template"],
        query_template=config["query_template"],
    )
