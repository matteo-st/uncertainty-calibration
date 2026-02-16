"""
LLM loading and inference for text classification.

This module provides the interface to load language models and extract
class probabilities for classification tasks.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm


class LLMClassifier:
    """
    Wrapper for using a causal LM as a text classifier.

    Given a prompt ending with a query, computes P(label_word | prompt)
    for each possible label word and normalizes to get class posteriors.
    """

    def __init__(
        self,
        model_name: str = "gpt2-xl",
        device: Optional[str] = None,
        max_memory: Optional[Dict] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize the classifier.

        Args:
            model_name: Hugging Face model identifier (default: gpt2-xl)
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_memory: Memory allocation for model parallelism
            checkpoint_path: Path to finetuned checkpoint (optional)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path

        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # GPT-2 doesn't have a pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with optional memory management
        load_kwargs = {"torch_dtype": torch.float16}
        if max_memory:
            load_kwargs["max_memory"] = max_memory
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = self.device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )

        # Load checkpoint if provided
        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print("Checkpoint loaded successfully")

        self.model.eval()
        print(f"Model loaded on {self.device}")

    def get_label_token_ids(self, labels: List[str]) -> List[List[int]]:
        """
        Get token IDs for each label word.

        Args:
            labels: List of label words (e.g., ["World", "Sports", "Business", "Technology"])

        Returns:
            List of token ID lists for each label
        """
        label_token_ids = []
        for label in labels:
            # Add space prefix as GPT-2 tokenizes words with leading space
            tokens = self.tokenizer.encode(" " + label, add_special_tokens=False)
            label_token_ids.append(tokens)
        return label_token_ids

    @torch.no_grad()
    def get_next_token_probs(self, prompt: str) -> torch.Tensor:
        """
        Get probability distribution over next token given prompt.

        Args:
            prompt: Input text prompt

        Returns:
            Tensor of shape (vocab_size,) with probabilities
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        # Get logits for the last token position
        last_logits = outputs.logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)

        return probs

    @torch.no_grad()
    def get_label_probabilities(
        self,
        prompt: str,
        labels: List[str],
        label_token_ids: Optional[List[List[int]]] = None,
    ) -> np.ndarray:
        """
        Compute P(y|prompt) for each class label.

        For multi-token labels, computes:
        P(w_k|prompt) = prod_{m=0}^{M-1} P(w_k^{m+1} | prompt, w_k^{1:m})

        Then normalizes: P(y=k|prompt) = P(w_k|prompt) / sum_k' P(w_k'|prompt)

        Args:
            prompt: Input prompt (preface + query)
            labels: List of label words
            label_token_ids: Pre-computed token IDs (optional, for efficiency)

        Returns:
            Array of shape (num_labels,) with normalized probabilities
        """
        if label_token_ids is None:
            label_token_ids = self.get_label_token_ids(labels)

        unnorm_probs = []

        for tokens in label_token_ids:
            # Start with the base prompt
            current_prompt = prompt
            prob = 1.0

            for i, token_id in enumerate(tokens):
                # Get next token probabilities
                inputs = self.tokenizer(current_prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                last_logits = outputs.logits[0, -1, :]
                probs = F.softmax(last_logits, dim=-1)

                # Accumulate probability for this token
                prob *= probs[token_id].item()

                # Extend prompt for next token (if multi-token label)
                if i < len(tokens) - 1:
                    current_prompt += self.tokenizer.decode([token_id])

            unnorm_probs.append(prob)

        # Normalize to get posterior distribution (Eq. 2 in paper)
        unnorm_probs = np.array(unnorm_probs)
        posteriors = unnorm_probs / unnorm_probs.sum()

        return posteriors

    @torch.no_grad()
    def get_batch_label_probabilities(
        self,
        prompts: List[str],
        labels: List[str],
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Compute label probabilities for multiple prompts.

        Args:
            prompts: List of prompts
            labels: List of label words
            batch_size: Batch size for inference
            show_progress: Whether to show progress bar

        Returns:
            Array of shape (num_prompts, num_labels) with probabilities
        """
        label_token_ids = self.get_label_token_ids(labels)
        all_probs = []

        iterator = tqdm(prompts, desc="Computing probabilities") if show_progress else prompts

        for prompt in iterator:
            probs = self.get_label_probabilities(prompt, labels, label_token_ids)
            all_probs.append(probs)

        return np.array(all_probs)
