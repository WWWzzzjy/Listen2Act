from __future__ import annotations

import torch
from torch import nn

from src.models.florence2_vla import _load_model_with_safetensors_fallback
from src.models.florence2_vla import _encode_florence2_multimodal_features


class FakeAutoModel:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        cls.calls.append({"model_name": model_name, **kwargs})
        if len(cls.calls) == 1:
            raise AttributeError("'NoneType' object has no attribute 'get'")
        return "loaded"


def test_load_model_falls_back_when_safetensors_metadata_missing() -> None:
    FakeAutoModel.calls = []

    model = _load_model_with_safetensors_fallback(
        FakeAutoModel,
        model_name="microsoft/Florence-2-base",
        revision="abc123",
        trust_remote_code=True,
    )

    assert model == "loaded"
    assert len(FakeAutoModel.calls) == 2
    assert "use_safetensors" not in FakeAutoModel.calls[0]
    assert FakeAutoModel.calls[1]["use_safetensors"] is False


class FakeEncoder(nn.Module):
    def forward(self, inputs_embeds, attention_mask=None, output_hidden_states=True, return_dict=True):
        return {"last_hidden_state": inputs_embeds + 1.0}


class FakeFlorenceBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(10, 4)
        self.encoder = FakeEncoder()
        self.decoder_called = False

    def get_input_embeddings(self):
        return self.embedding

    def _encode_image(self, pixel_values):
        return torch.ones(pixel_values.shape[0], 2, 4, device=pixel_values.device)

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds):
        image_attention = torch.ones(image_features.shape[:2], device=image_features.device)
        text_attention = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device)
        return torch.cat([image_features, inputs_embeds], dim=1), torch.cat([image_attention, text_attention], dim=1)

    def get_encoder(self):
        return self.encoder

    def forward(self, *args, **kwargs):
        self.decoder_called = True
        raise AssertionError("Full Florence decoder path should not be called")


def test_encode_florence_features_uses_encoder_only() -> None:
    backbone = FakeFlorenceBackbone()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "pixel_values": torch.zeros(1, 3, 8, 8),
    }

    hidden, attention_mask = _encode_florence2_multimodal_features(backbone, inputs)

    assert hidden.shape == (1, 5, 4)
    assert attention_mask.shape == (1, 5)
    assert backbone.decoder_called is False
