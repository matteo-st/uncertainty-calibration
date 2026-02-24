"""
Encoder-based transformer classifiers for sequence classification.

Wraps HuggingFace encoder models (ELECTRA, BERT, etc.) for:
1. Fine-tuning on classification tasks via HuggingFace Trainer
2. Extracting penultimate-layer hidden representations (for Mahalanobis distance)
3. Getting class probabilities (for softmax-based uncertainty scores)

Unlike models.py (decoder-based, prompt engineering with GPT-2), this module
uses standard classification heads fine-tuned end-to-end.

Reference: Vazhentsev et al. (ACL 2022)
"Uncertainty Estimation of Transformer Predictions for Misclassification Detection"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from typing import Optional, Dict, Tuple
from pathlib import Path
import logging
import copy

logger = logging.getLogger(__name__)


def apply_spectral_norm_to_classifier(model) -> None:
    """
    Apply spectral normalization to the dense layer of the classification head.

    For ELECTRA, the classification head is:
        [CLS] -> dropout -> dense(768->768) -> GELU -> out_proj(768->num_classes)

    Spectral normalization is applied to the dense layer's weight matrix,
    enforcing a bi-Lipschitz constraint that makes representations more
    distance-preserving (beneficial for Mahalanobis distance).

    Following the paper's approach:
        At each training step: nu = ||W||_2, W_tilde = W / nu

    Args:
        model: HuggingFace model with a classifier head
    """
    # Handle ELECTRA's classifier structure
    if hasattr(model, "classifier"):
        classifier = model.classifier
        # ELECTRA classifier head has: dense -> out_proj
        if hasattr(classifier, "dense"):
            logger.info("Applying spectral normalization to classifier.dense")
            classifier.dense = nn.utils.spectral_norm(classifier.dense)
        else:
            logger.warning(
                "Could not find 'dense' layer in classifier head. "
                "Spectral normalization not applied."
            )
    else:
        logger.warning(
            "Model does not have a 'classifier' attribute. "
            "Spectral normalization not applied."
        )


def compute_classification_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute metrics for HuggingFace Trainer evaluation callback.

    Args:
        eval_pred: EvalPrediction with predictions and label_ids

    Returns:
        Dict with accuracy and f1 metrics
    """
    from sklearn.metrics import accuracy_score, f1_score

    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }


class EncoderClassifier:
    """
    Wrapper for encoder-based transformer classifiers (ELECTRA, BERT, etc.).

    Unlike LLMClassifier (decoder-based, prompt engineering),
    this uses a standard classification head fine-tuned end-to-end.

    Supports:
    - Fine-tuning via HuggingFace Trainer
    - Prediction (class probabilities)
    - Feature extraction from penultimate layer (for Mahalanobis distance)
    - Optional spectral normalization on classification head
    """

    def __init__(
        self,
        model_name: str = "google/electra-base-discriminator",
        num_labels: int = 2,
        use_spectral_norm: bool = False,
        device: Optional[str] = None,
    ):
        """
        Initialize the encoder classifier.

        Args:
            model_name: HuggingFace model identifier
            num_labels: Number of classification labels
            use_spectral_norm: If True, apply spectral normalization to
                             the classifier head's dense layer (for MD SN)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_spectral_norm = use_spectral_norm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

        if use_spectral_norm:
            apply_spectral_norm_to_classifier(self.model)
            logger.info("Spectral normalization applied to classifier head")

        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")

    def finetune(
        self,
        train_dataset,
        val_dataset=None,
        output_dir: str = "checkpoints/encoder",
        learning_rate: float = 5e-5,
        per_device_train_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        num_train_epochs: int = 12,
        weight_decay: float = 0.1,
        warmup_ratio: float = 0.1,
        seed: int = 42,
        logging_steps: int = 50,
        save_strategy: str = "epoch",
        eval_strategy: str = "epoch",
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "f1",
        **extra_args,
    ) -> None:
        """
        Fine-tune the model using HuggingFace Trainer.

        Args:
            train_dataset: Training dataset (HuggingFace Dataset, tokenized)
            val_dataset: Validation dataset (optional, for evaluation during training)
            output_dir: Directory for checkpoints
            learning_rate: Learning rate for AdamW optimizer
            per_device_train_batch_size: Training batch size per device
            per_device_eval_batch_size: Evaluation batch size per device
            num_train_epochs: Number of training epochs
            weight_decay: Weight decay for AdamW
            warmup_ratio: Fraction of total steps for linear warmup
            seed: Random seed
            logging_steps: Log every N steps
            save_strategy: When to save checkpoints ('epoch', 'steps', 'no')
            eval_strategy: When to evaluate ('epoch', 'steps', 'no')
            load_best_model_at_end: Load best checkpoint at end of training
            metric_for_best_model: Metric for best model selection
            **extra_args: Additional TrainingArguments
        """
        logger.info("=" * 60)
        logger.info("Fine-tuning encoder model")
        logger.info("=" * 60)
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Batch size: {per_device_train_batch_size}")
        logger.info(f"  Epochs: {num_train_epochs}")
        logger.info(f"  Weight decay: {weight_decay}")
        logger.info(f"  Warmup ratio: {warmup_ratio}")
        logger.info(f"  Spectral norm: {self.use_spectral_norm}")

        # If no val_dataset, disable evaluation during training
        if val_dataset is None:
            eval_strategy = "no"
            load_best_model_at_end = False

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            seed=seed,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            eval_strategy=eval_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model if val_dataset else None,
            save_total_limit=2,
            report_to="none",  # Disable wandb/tensorboard
            **extra_args,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_classification_metrics,
        )

        logger.info("Starting training...")
        train_result = trainer.train()

        logger.info(f"Training complete.")
        logger.info(f"  Train loss: {train_result.training_loss:.4f}")
        logger.info(f"  Train runtime: {train_result.metrics['train_runtime']:.1f}s")

        # Evaluate on validation set if provided
        if val_dataset is not None:
            eval_results = trainer.evaluate()
            logger.info(f"  Val accuracy: {eval_results.get('eval_accuracy', 'N/A')}")
            logger.info(f"  Val F1: {eval_results.get('eval_f1', 'N/A')}")

    @torch.no_grad()
    def predict(self, dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on a dataset.

        Args:
            dataset: HuggingFace Dataset (tokenized)

        Returns:
            Tuple of:
                probs: (N, num_labels) softmax probabilities
                predictions: (N,) predicted class indices
                labels: (N,) true labels
        """
        self.model.eval()

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir="/tmp/predict",
                per_device_eval_batch_size=64,
                report_to="none",
            ),
        )

        output = trainer.predict(dataset)
        logits = output.predictions  # (N, num_labels)
        labels = output.label_ids    # (N,)

        probs = self._softmax(logits)
        predictions = np.argmax(probs, axis=1)

        accuracy = (predictions == labels).mean()
        logger.info(f"Prediction accuracy: {accuracy:.4f} ({(predictions == labels).sum()}/{len(labels)})")

        return probs, predictions, labels

    @torch.no_grad()
    def extract_features(
        self,
        dataset,
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract penultimate layer representations for Mahalanobis distance.

        For ELECTRA, the classification head is:
            [CLS] -> dropout -> dense(768->768) -> GELU -> out_proj(768->num_classes)

        We extract the output after dense -> GELU, before out_proj.
        This follows the paper's ElectraClassificationHeadIdentityPooler pattern.

        Implementation: we register a forward hook on the activation function
        (GELU) after the dense layer to capture its output, rather than
        modifying the model architecture. This is non-invasive and works
        regardless of spectral normalization.

        Args:
            dataset: HuggingFace Dataset (tokenized)
            batch_size: Batch size for feature extraction

        Returns:
            Tuple of:
                features: (N, hidden_dim) feature matrix
                labels: (N,) true labels
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # Storage for hook output
        features_list = []

        # Identify the activation function after the dense layer in the classifier
        # For ELECTRA: model.classifier.dense -> model.classifier.activation (GELU)
        # The hook captures the output of the activation function
        hook_layer = self._get_feature_hook_layer()

        def hook_fn(module, input, output):
            # output shape: (batch_size, hidden_dim)
            features_list.append(output.detach().cpu().numpy())

        handle = hook_layer.register_forward_hook(hook_fn)

        # Run inference in batches using DataLoader
        from torch.utils.data import DataLoader

        # Temporarily remove format to use default collation if needed
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_labels = []
        for batch in dataloader:
            # Move inputs to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}

            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            # Forward pass (hook captures features automatically)
            self.model(**kwargs)

            # Collect labels
            if "label" in batch:
                all_labels.append(batch["label"].numpy())
            elif "labels" in batch:
                all_labels.append(batch["labels"].numpy())

        # Remove hook
        handle.remove()

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])

        logger.info(f"Extracted features: shape={features.shape}, "
                    f"labels={labels.shape}")

        return features, labels

    def _get_feature_hook_layer(self) -> nn.Module:
        """
        Get the layer to hook for feature extraction.

        For ELECTRA, this is the activation function (GELU) after the
        dense layer in the classifier head. The output of this layer
        is the penultimate representation before the final out_proj layer.

        Returns:
            nn.Module to attach the forward hook to
        """
        classifier = self.model.classifier

        # ELECTRA classifier structure:
        #   ElectraClassificationHead:
        #     (dense): Linear(768, 768)
        #     (activation_fn): GELUActivation() [or similar]
        #     (dropout): Dropout
        #     (out_proj): Linear(768, num_labels)
        #
        # We want the output after activation_fn (after dense -> GELU).
        # In HuggingFace transformers, the ELECTRA classifier head uses
        # get_activation("gelu") which may be stored under different names
        # depending on the transformers version.

        # Try common attribute names for the activation function
        for attr_name in ["activation_fn", "act_fn", "activation"]:
            if hasattr(classifier, attr_name):
                layer = getattr(classifier, attr_name)
                logger.info(f"Feature extraction hook on: classifier.{attr_name} "
                            f"({type(layer).__name__})")
                return layer

        # Fallback: hook on the dense layer itself
        # The output of dense + GELU is what we want, but if we can only hook
        # on dense, we get pre-GELU features. This is acceptable as a fallback
        # since GELU is monotonic and approximately preserves distances.
        if hasattr(classifier, "dense"):
            logger.warning(
                "Could not find activation function in classifier head. "
                "Hooking on classifier.dense instead (pre-activation features)."
            )
            return classifier.dense

        raise AttributeError(
            f"Cannot identify feature extraction layer in classifier head. "
            f"Classifier attributes: {[a for a in dir(classifier) if not a.startswith('_')]}"
        )

    def save(self, path: str) -> None:
        """
        Save the fine-tuned model and tokenizer.

        If spectral normalization was applied, it is temporarily removed
        before saving so that weights are stored as regular parameters
        (compatible with from_pretrained loading).

        Args:
            path: Directory to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Remove spectral norm before saving to get regular weight parameters.
        # remove_spectral_norm folds the normalized weight back into .weight,
        # so the saved model has the correct inference weights.
        sn_removed = False
        if self.use_spectral_norm and hasattr(self.model, "classifier"):
            dense = getattr(self.model.classifier, "dense", None)
            if dense is not None and hasattr(dense, "weight_orig"):
                nn.utils.remove_spectral_norm(dense)
                sn_removed = True
                logger.info("Temporarily removed spectral norm for saving")

        self.model.save_pretrained(path)

        # Re-apply spectral norm after saving (for continued use)
        if sn_removed:
            apply_spectral_norm_to_classifier(self.model)
            logger.info("Re-applied spectral norm after saving")

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        model_name: str = "google/electra-base-discriminator",
        num_labels: int = 2,
        use_spectral_norm: bool = False,
        device: Optional[str] = None,
    ) -> "EncoderClassifier":
        """
        Load a fine-tuned model from checkpoint.

        Args:
            path: Directory containing saved model
            model_name: Original model name (for config reference)
            num_labels: Number of labels
            use_spectral_norm: Whether spectral norm was used during training
            device: Device to load onto

        Returns:
            EncoderClassifier instance with loaded weights
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading fine-tuned model from {path}")
        instance = cls.__new__(cls)
        instance.model_name = model_name
        instance.num_labels = num_labels
        instance.use_spectral_norm = use_spectral_norm
        instance.device = device

        instance.model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=num_labels,
        )

        if use_spectral_norm:
            apply_spectral_norm_to_classifier(instance.model)

        instance.model.to(device)
        logger.info(f"Model loaded on {device}")

        return instance

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities from logits (numpy)."""
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)
