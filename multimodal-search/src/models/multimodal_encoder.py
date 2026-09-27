"""
Multimodal Encoder — loads a pretrained CLIP model and provides a unified
interface for encoding text and images into a shared embedding space.

The encoder is designed to be:
- Loaded once and reused across all queries
- Device-aware (auto-selects CUDA/CPU)
- Batch-capable for efficient offline embedding generation
- Modular — the model can be swapped by changing MODEL_NAME in config

Supported models:
- openai/clip-vit-base-patch32  (default, 512-dim embeddings)
- openai/clip-vit-large-patch14 (768-dim, higher quality)
- Any HuggingFace CLIP-compatible model
"""

import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

# Add project root so we can import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_NAME, DEVICE, BATCH_SIZE


class MultimodalEncoder:
    """
    Encodes text and images into normalized embeddings using a pretrained
    CLIP model.

    Usage:
        encoder = MultimodalEncoder()
        text_embs = encoder.encode_text(["a dog in a park"])
        img_embs  = encoder.encode_images([PIL.Image, ...])
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = DEVICE,
        batch_size: int = BATCH_SIZE,
    ):
        """
        Initialize the encoder by loading the pretrained model.

        Args:
            model_name: HuggingFace model identifier (e.g. "openai/clip-vit-base-patch32").
            device: "cuda" or "cpu". Defaults to auto-detected value from config.
            batch_size: Number of items to process per forward pass.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        print(f"Loading model: {model_name}")
        print(f"Device: {device}")

        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.projection_dim
        print(f"Embedding dimension: {self.embedding_dim}")
        print("Model loaded successfully.\n")

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of text strings into normalized embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), L2-normalized.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        all_embeddings = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]

            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,  # CLIP max token length
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            # Get text embeddings
            text_features = self.model.get_text_features(**inputs)

            # L2 normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(text_features.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    @torch.no_grad()
    def encode_images(self, images: List[Union[Image.Image, str, Path]]) -> np.ndarray:
        """
        Encode a list of images into normalized embeddings.

        Args:
            images: List of PIL Images, or file paths (str/Path) to images.

        Returns:
            np.ndarray of shape (len(images), embedding_dim), L2-normalized.
        """
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        all_embeddings = []

        for start in range(0, len(images), self.batch_size):
            batch_items = images[start : start + self.batch_size]

            # Load images if paths are provided
            batch_images = []
            for item in batch_items:
                if isinstance(item, (str, Path)):
                    try:
                        img = Image.open(item).convert("RGB")
                        batch_images.append(img)
                    except Exception as e:
                        print(f"Warning: Could not load image {item}: {e}")
                        # Use a small blank image as fallback
                        batch_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                elif isinstance(item, Image.Image):
                    batch_images.append(item.convert("RGB"))
                else:
                    raise TypeError(f"Unsupported image type: {type(item)}")

            inputs = self.processor(
                images=batch_images,
                return_tensors="pt",
                padding=True,
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            # Get image embeddings
            image_features = self.model.get_image_features(**inputs)

            # L2 normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(image_features.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    @torch.no_grad()
    def encode_image_single(self, image: Union[Image.Image, str, Path]) -> np.ndarray:
        """
        Encode a single image. Convenience wrapper around encode_images.

        Returns:
            1-D np.ndarray of shape (embedding_dim,), L2-normalized.
        """
        return self.encode_images([image])[0]

    @torch.no_grad()
    def encode_text_single(self, text: str) -> np.ndarray:
        """
        Encode a single text string. Convenience wrapper around encode_text.

        Returns:
            1-D np.ndarray of shape (embedding_dim,), L2-normalized.
        """
        return self.encode_text([text])[0]

    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self.embedding_dim

    def get_device(self) -> str:
        """Return the device the model is running on."""
        return self.device


# ---------------------------------------------------------------------------
# Quick self-test — run this file directly to verify the encoder works
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("MultimodalEncoder — Self-Test")
    print("=" * 60)

    encoder = MultimodalEncoder()

    # Test text encoding
    texts = ["a dog playing in a park", "a red car on a highway"]
    text_embs = encoder.encode_text(texts)
    print(f"Text embeddings shape: {text_embs.shape}")
    print(f"Text embedding norms:  {np.linalg.norm(text_embs, axis=1)}")

    # Test image encoding (create a dummy image)
    dummy_img = Image.new("RGB", (224, 224), (128, 128, 128))
    img_embs = encoder.encode_images([dummy_img])
    print(f"Image embeddings shape: {img_embs.shape}")
    print(f"Image embedding norms:  {np.linalg.norm(img_embs, axis=1)}")

    # Test cosine similarity (dot product of normalized vectors)
    similarity = np.dot(text_embs[0], img_embs[0])
    print(f"Cosine similarity (text[0] vs dummy image): {similarity:.4f}")

    # Verify dimensions match
    assert text_embs.shape[1] == img_embs.shape[1], "Embedding dimensions must match!"
    assert text_embs.shape[1] == encoder.get_embedding_dim()

    # Verify normalization
    norms = np.linalg.norm(text_embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings must be L2-normalized!"

    print("\n✓ All self-tests passed!")
