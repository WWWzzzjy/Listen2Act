from __future__ import annotations

from src.models.florence2_vla import _load_model_with_safetensors_fallback


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

