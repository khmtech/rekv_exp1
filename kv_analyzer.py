"""
kv_analyzer.py - Core KV Similarity Analysis for Speculative Decoding
=====================================================================

Draft/Target 紐⑤뜽??KV cache瑜?position-level濡?異붿텧?섍퀬 鍮꾧탳?섎뒗 ?듭떖 紐⑤뱢.

HuggingFace??`past_key_values` 諛섑솚媛믪쓣 吏곸젒 ?ъ슜?섏뿬 KV瑜?異붿텧?쒕떎.
Hook???꾨땶 model output?먯꽌 吏곸젒 媛?몄삤誘濡??꾪궎?띿쿂 ?명솚?깆씠 ?믩떎.

Key design decisions:
1. Draft? Target??layer ?섍? ?ㅻ? ??鍮꾨? 留ㅽ븨(proportional mapping)
2. GQA濡??명빐 KV head ?섍? ?ㅻ? ??min heads濡?鍮꾧탳
3. Head dimension???ㅻ? ??truncation
4. DynamicCache / tuple ?뺥깭 紐⑤몢 吏??"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any


# ============================================================
# Data Structures
# ============================================================

@dataclass
class KVSimilarityResult:
    """?⑥씪 position??draft KV vs target KV 鍮꾧탳 寃곌낵."""
    # Per-layer metrics: list of float (layer 李⑥썝)
    key_cosine_per_layer: List[float] = field(default_factory=list)
    value_cosine_per_layer: List[float] = field(default_factory=list)
    key_l2_per_layer: List[float] = field(default_factory=list)
    value_l2_per_layer: List[float] = field(default_factory=list)
    # Per-layer, per-head metrics: list of list of float
    key_cosine_per_layer_head: List[List[float]] = field(default_factory=list)
    value_cosine_per_layer_head: List[List[float]] = field(default_factory=list)

    @property
    def mean_key_cosine(self) -> float:
        return float(np.mean(self.key_cosine_per_layer)) if self.key_cosine_per_layer else 0.0

    @property
    def mean_value_cosine(self) -> float:
        return float(np.mean(self.value_cosine_per_layer)) if self.value_cosine_per_layer else 0.0

    @property
    def mean_key_l2(self) -> float:
        return float(np.mean(self.key_l2_per_layer)) if self.key_l2_per_layer else 0.0

    @property
    def mean_value_l2(self) -> float:
        return float(np.mean(self.value_l2_per_layer)) if self.value_l2_per_layer else 0.0


@dataclass
class RejectionEvent:
    """Speculative decoding?먯꽌 ?⑥씪 rejection ?대깽?몄쓽 ?꾩껜 ?뺣낫."""
    sample_id: int
    round_id: int
    spec_length: int               # K: speculation window size
    accept_length: int             # rejection ?꾧퉴吏 accept??token ??    rejection_pos_in_window: int   # speculation window ??rejection ?꾩튂 (0-indexed)
    rejection_pos_in_seq: int      # ?꾩껜 sequence?먯꽌??position
    draft_token_id: int
    target_token_id: int
    draft_confidence: float        # P_draft(draft_token)
    target_confidence: float       # P_target(target_token)
    draft_token_rank_in_target: int  # target 遺꾪룷?먯꽌 draft token??rank
    kv_sim: Optional[KVSimilarityResult] = None

    def to_dict(self) -> dict:
        d = {
            "sample_id": self.sample_id,
            "round_id": self.round_id,
            "spec_length": self.spec_length,
            "accept_length": self.accept_length,
            "rejection_pos_in_window": self.rejection_pos_in_window,
            "rejection_pos_in_seq": self.rejection_pos_in_seq,
            "draft_token_id": self.draft_token_id,
            "target_token_id": self.target_token_id,
            "draft_confidence": self.draft_confidence,
            "target_confidence": self.target_confidence,
            "draft_token_rank_in_target": self.draft_token_rank_in_target,
        }
        if self.kv_sim:
            d["mean_key_cosine"] = self.kv_sim.mean_key_cosine
            d["mean_value_cosine"] = self.kv_sim.mean_value_cosine
            d["mean_key_l2"] = self.kv_sim.mean_key_l2
            d["mean_value_l2"] = self.kv_sim.mean_value_l2
            d["key_cosine_per_layer"] = self.kv_sim.key_cosine_per_layer
            d["value_cosine_per_layer"] = self.kv_sim.value_cosine_per_layer
            d["key_l2_per_layer"] = self.kv_sim.key_l2_per_layer
            d["value_l2_per_layer"] = self.kv_sim.value_l2_per_layer
            d["key_cosine_per_layer_head"] = self.kv_sim.key_cosine_per_layer_head
            d["value_cosine_per_layer_head"] = self.kv_sim.value_cosine_per_layer_head
        return d


@dataclass
class RoundStats:
    """Speculation round ?듦퀎."""
    sample_id: int
    round_id: int
    spec_length: int
    accept_length: int
    all_accepted: bool
    draft_time_ms: float
    verify_time_ms: float


# ============================================================
# KV Cache Utility Functions
# ============================================================

def _normalize_past_kv(past_key_values: Any):
    """
    Normalize HF cache types to a legacy tuple/list structure when needed.
    Supports transformers.cache_utils.DynamicCache via to_legacy_cache().
    """
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    # Some versions store legacy cache on .cache
    if hasattr(past_key_values, "cache"):
        return past_key_values.cache
    return past_key_values

def extract_kv_at_position(
    past_key_values,
    layer_idx: int,
    position: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    past_key_values?먯꽌 ?뱀젙 layer, position??KV 踰≫꽣瑜?異붿텧.

    past_key_values ?뺥깭:
    - Tuple of (key, value): key/value shape = [batch, num_kv_heads, seq_len, head_dim]
    - DynamicCache: .key_cache[layer], .value_cache[layer]

    Returns:
        key: [num_kv_heads, head_dim]
        value: [num_kv_heads, head_dim]
    """
    past_key_values = _normalize_past_kv(past_key_values)

    # DynamicCache (transformers >= 4.36)
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        k = layer.keys
        v = layer.values
    elif hasattr(past_key_values, 'key_cache'):
        k = past_key_values.key_cache[layer_idx]   # [batch, heads, seq, dim]
        v = past_key_values.value_cache[layer_idx]
    # Some variants expose cache as list/tuple directly
    elif hasattr(past_key_values, "cache") and isinstance(past_key_values.cache, (tuple, list)):
        k, v = past_key_values.cache[layer_idx]
    # Tuple of tuples: ((k, v), (k, v), ...)
    elif isinstance(past_key_values, (tuple, list)):
        k, v = past_key_values[layer_idx]
    else:
        raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")

    # batch dim ?쒓굅, position ?좏깮
    return k[0, :, position, :].detach(), v[0, :, position, :].detach()


def get_num_layers(past_key_values) -> int:
    """KV cache??layer ??諛섑솚."""
    past_key_values = _normalize_past_kv(past_key_values)
    if hasattr(past_key_values, "layers"):
        return len(past_key_values.layers)
    elif hasattr(past_key_values, "key_cache"):
        return len(past_key_values.key_cache)
    elif hasattr(past_key_values, "cache") and isinstance(past_key_values.cache, (tuple, list)):
        return len(past_key_values.cache)
    elif isinstance(past_key_values, (tuple, list)):
        return len(past_key_values)
    else:
        raise TypeError(f"Unsupported type: {type(past_key_values)}")


def compute_layer_mapping(
    num_draft_layers: int,
    num_target_layers: int,
) -> List[Tuple[int, int]]:
    """
    Draft layer? Target layer 媛?鍮꾨? 留ㅽ븨 ?앹꽦.

    ?덉떆:
    - Draft 16 layers, Target 32 layers ??[(0,0), (1,2), (2,4), ..., (15,30)]
    - Draft 16 layers, Target 48 layers ??[(0,0), (1,3), (2,6), ..., (15,45)]

    Returns:
        List of (draft_layer_idx, target_layer_idx) pairs
    """
    mapping = []
    for d in range(num_draft_layers):
        # 鍮꾨? 留ㅽ븨: draft???곷????꾩튂瑜?target?쇰줈 蹂??        t = round(d * (num_target_layers - 1) / (num_draft_layers - 1)) if num_draft_layers > 1 else 0
        mapping.append((d, t))
    return mapping


def compare_kv_at_position(
    draft_past_kv,
    target_past_kv,
    position: int,
    layer_mapping: Optional[List[Tuple[int, int]]] = None,
) -> KVSimilarityResult:
    """
    ?뱀젙 position?먯꽌 draft? target??KV cache瑜?layer蹂? head蹂꾨줈 鍮꾧탳.

    Draft? Target??李⑥씠瑜?泥섎━?섎뒗 諛⑸쾿:
    - Layer ??李⑥씠: layer_mapping???곕씪 鍮꾨? 留ㅽ븨
    - KV head ??李⑥씠 (GQA): min(draft_heads, target_heads)源뚯? 鍮꾧탳
    - Head dim 李⑥씠: min(draft_dim, target_dim)源뚯? truncation

    Args:
        draft_past_kv: draft model??past_key_values
        target_past_kv: target model??past_key_values
        position: 鍮꾧탳??sequence position
        layer_mapping: (draft_layer, target_layer) ??由ъ뒪??    """
    draft_past_kv = _normalize_past_kv(draft_past_kv)
    target_past_kv = _normalize_past_kv(target_past_kv)

    num_draft_layers = get_num_layers(draft_past_kv)
    num_target_layers = get_num_layers(target_past_kv)

    if layer_mapping is None:
        layer_mapping = compute_layer_mapping(num_draft_layers, num_target_layers)

    result = KVSimilarityResult()

    for draft_l, target_l in layer_mapping:
        # KV 異붿텧: [num_kv_heads, head_dim]
        dk, dv = extract_kv_at_position(draft_past_kv, draft_l, position)
        tk, tv = extract_kv_at_position(target_past_kv, target_l, position)

        # Head ??留욎텛湲?        min_heads = min(dk.shape[0], tk.shape[0])
        # Head dim 留욎텛湲?        min_dim = min(dk.shape[1], tk.shape[1])

        dk = dk[:min_heads, :min_dim].float()
        tk = tk[:min_heads, :min_dim].float()
        dv = dv[:min_heads, :min_dim].float()
        tv = tv[:min_heads, :min_dim].float()

        # Per-head cosine similarity: [min_heads]
        k_cos_heads = F.cosine_similarity(dk, tk, dim=-1)  # [min_heads]
        v_cos_heads = F.cosine_similarity(dv, tv, dim=-1)

        # Per-head L2 distance (normalized by sqrt(dim) for comparability)
        k_l2_heads = torch.norm(dk - tk, p=2, dim=-1) / (min_dim ** 0.5)
        v_l2_heads = torch.norm(dv - tv, p=2, dim=-1) / (min_dim ** 0.5)

        # Layer-level: mean over heads
        result.key_cosine_per_layer.append(float(k_cos_heads.mean()))
        result.value_cosine_per_layer.append(float(v_cos_heads.mean()))
        result.key_l2_per_layer.append(float(k_l2_heads.mean()))
        result.value_l2_per_layer.append(float(v_l2_heads.mean()))

        # Per-head detail
        result.key_cosine_per_layer_head.append(k_cos_heads.cpu().tolist())
        result.value_cosine_per_layer_head.append(v_cos_heads.cpu().tolist())

    return result


# ============================================================
# Random Baseline for Reference
# ============================================================

def compute_random_baseline(
    target_past_kv,
    position: int,
    num_samples: int = 100,
) -> Dict[str, float]:
    """
    Random vector? target KV 媛?cosine similarity瑜?痢≪젙.
    "Rejected KV媛 random蹂대떎 ?쇰쭏??醫뗭?媛"??baseline.

    Returns:
        {"key_cosine_mean": float, "value_cosine_mean": float}
    """
    num_layers = get_num_layers(target_past_kv)

    k_sims = []
    v_sims = []

    for layer_idx in range(num_layers):
        tk, tv = extract_kv_at_position(target_past_kv, layer_idx, position)
        # tk: [num_kv_heads, head_dim]
        num_heads, head_dim = tk.shape

        for _ in range(num_samples):
            rand_k = torch.randn_like(tk)
            rand_v = torch.randn_like(tv)

            k_cos = F.cosine_similarity(rand_k, tk.float(), dim=-1).mean()
            v_cos = F.cosine_similarity(rand_v, tv.float(), dim=-1).mean()
            k_sims.append(k_cos.item())
            v_sims.append(v_cos.item())

    return {
        "key_cosine_mean": float(np.mean(k_sims)),
        "value_cosine_mean": float(np.mean(v_sims)),
        "key_cosine_std": float(np.std(k_sims)),
        "value_cosine_std": float(np.std(v_sims)),
    }

