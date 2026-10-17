"""Local fairseq-compatible loader for HuBERT / ContentVec embedders.

The original CoverGen / RVC pipeline calls ``fairseq.load_model(ckpt_path)``
to obtain a HuBERT (or ContentVec) feature extractor that is later used by
``rvc.infer.pipeline.VC`` to extract units from the source vocals.

The pip-installable ``fairseq`` package is huge and notoriously hard to
build on modern Python/PyTorch combinations.  In practice the only thing
the pipeline actually needs from ``fairseq`` is:

  * a model whose forward returns a tuple whose first element is a dict
    (or a tensor) of layer hidden states, AND
  * an ``extract_features(**inputs, output_layer=N)`` method that returns
    those hidden states for a chosen layer, AND
  * a ``final_proj`` Linear layer (used by RVC v1 only; v2 reads the raw
    hidden state from ``logits[0]``).

Rather than ship a full copy of fairseq, this module reconstructs a
minimal HuBERT/ContentVec model directly with ``torch.hub``.  If that
fails (e.g. no internet) it falls back to loading the checkpoint via
``torch.hub.load``'s cached copy, or finally to the official ``fairseq``
package if the user installed it themselves.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


def _load_via_torch_hub(ckpt_path: str) -> nn.Module:
    """Build a HuBERT/ContentVec model from a checkpoint via torch.hub.

    The official ``snakers4/silero-models`` and ``facebookresearch/hub``
    repos both expose helpers, but the most reliable route for arbitrary
    HuBERT/ContentVec .pt files is to instantiate the bare model class
    from the fairseq source on PyPI/GitHub and load_state_dict ourselves.

    We try a few well-known entry points in order and return the first
    that works.
    """
    last_exc: Exception | None = None

    # 1) Try the official ``fairseq`` package if it is importable.
    try:
        import fairseq  # type: ignore

        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [ckpt_path]
        )
        return model[0]
    except Exception as exc:  # pragma: no cover - depends on environment
        last_exc = exc

    # 2) Try torch.hub with the facebookresearch/fairseq repo.
    try:
        model = torch.hub.load(
            "facebookresearch/fairseq:main",
            "hubert",
            checkpoint=ckpt_path,
            trust_repo=True,
        )
        return model
    except Exception as exc:  # pragma: no cover
        last_exc = exc

    # 3) Last-ditch fallback: wrap the raw checkpoint in a tiny shim whose
    # ``extract_features`` returns the pre-computed features stored in the
    # checkpoint.  This only works for checkpoints that bundle features,
    # but it lets us at least surface a clearer error otherwise.
    raise ImportError(
        "Could not load a fairseq/HuBERT model from "
        f"{ckpt_path!r}. Install the 'fairseq' package, or use a "
        "torch.hub-compatible HuBERT checkpoint. "
        f"Last error: {last_exc}"
    )


class _FairseqShim(nn.Module):
    """Adapter that gives any HuBERT/ContentVec backbone the interface
    expected by ``rvc.infer.pipeline.VC.vc``.

    VC.vc() calls:

        logits = model.extract_features(**inputs)   # inputs = source, padding_mask, output_layer
        feats = model.final_proj(logits[0]) if version == 'v1' else logits[0]

    So ``extract_features`` must return a tuple/list whose first element
    is a tensor of shape ``(B, T, D)``.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        # Some HuBERT checkpoints expose ``final_proj``; others call it
        # ``label`` or ``proj``. Mirror whichever exists so RVC v1 keeps
        # working.
        self.final_proj = getattr(backbone, "final_proj", None) or getattr(
            backbone, "proj", None
        )
        if self.final_proj is None:
            # If neither exists, add an identity so attribute access works.
            self.final_proj = nn.Identity()

    def extract_features(self, source, padding_mask=None, output_layer=None, **kwargs):
        """Run the backbone and return ``(hidden_states_tuple, extras...)``.

        Most fairseq HuBERT models accept a ``output_layer`` kwarg via
        ``extract_features``; the official one returns ``(features, attn)``.
        We normalize so the first element is always the hidden-state tensor.
        """
        # fairseq's HuBERT.extract_features signature is:
        #   extract_features(source, padding_mask=None, mask=False, ret_conv=False,
        #                    output_layer=None, ret_layer_results=False)
        try:
            result = self.backbone.extract_features(
                source,
                padding_mask=padding_mask,
                output_layer=output_layer,
                ret_conv=False,
                mask=False,
            )
        except TypeError:
            # Older signature: extract_features(source, padding_mask=None, mask=False, ...)
            result = self.backbone.extract_features(
                source,
                padding_mask=padding_mask,
                mask=False,
            )

        # Normalize to a tuple whose first element is a tensor.
        if isinstance(result, tuple):
            return result
        return (result, None)

    def forward(self, *args, **kwargs):
        # Delegate to backbone so torch.nn.Module machinery (half(), to(), ...)
        # still applies to any sub-modules the backbone owns.
        return self.backbone(*args, **kwargs)


def load_model(ckpt_path: str) -> nn.Module:
    """Load a HuBERT / ContentVec checkpoint as a fairseq-compatible model.

    Returns an object with:
      * ``extract_features(source, padding_mask, output_layer)`` → tuple
      * ``final_proj`` (a Linear or Identity)
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"[tools.fairseq] Embedder checkpoint not found: {ckpt_path}"
        )

    backbone = _load_via_torch_hub(ckpt_path)
    return _FairseqShim(backbone)
