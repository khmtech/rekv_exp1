"""
kv_analyzer.py - Core KV Similarity Analysis for Speculative Decoding
=====================================================================

Draft/Target 모델의 KV cache를 position-level로 추출하고 비교하는 핵심 모듈.

HuggingFace의 `past_key_values` 반환값을 직접 사용하여 KV를 추출한다.
Hook이 아닌 model output에서 직접 가져오므로 아키텍처 호환성이 높다.

Key design decisions:
1. Draft와 Target의 layer 수가 다를 때 비례 매핑(proportional mapping)
2. GQA로 인해 KV head 수가 다를 때 min heads로 비교
3. Head dimension이 다를 때 truncation
4. DynamicCache / tuple 형태 모두 지원
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


# ============================================================
# Data Structures
# ============================================================

@dataclass
class KVSimilarityResult:
    """단일 position의 draft KV vs target KV 비교 결과."""
    # Per-layer metrics: list of float (layer 차원)
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
    """Speculative decoding에서 단일 rejection 이벤트의 전체 정보."""
    sample_id: int
    round_id: int
    spec_length: int               # K: speculation window size
    accept_length: int             # rejection 전까지 accept된 token 수
    rejection_pos_in_window: int   # speculation window 내 rejection 위치 (0-indexed)
    rejection_pos_in_seq: int      # 전체 sequence에서의 position
    draft_token_id: int
    target_token_id: int
    draft_confidence: float        # P_draft(draft_token)
    target_confidence: float       # P_target(target_token)
    draft_token_rank_in_target: int  # target 분포에서 draft token의 rank
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
    """Speculation round 통계."""
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

def extract_kv_at_position(
    past_key_values,
    layer_idx: int,
    position: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    past_key_values에서 특정 layer, position의 KV 벡터를 추출.

    past_key_values 형태:
    - Tuple of (key, value): key/value shape = [batch, num_kv_heads, seq_len, head_dim]
    - DynamicCache: .key_cache[layer], .value_cache[layer]

    Returns:
        key: [num_kv_heads, head_dim]
        value: [num_kv_heads, head_dim]
    """
    # DynamicCache (transformers >= 4.36)
    if hasattr(past_key_values, 'key_cache'):
        k = past_key_values.key_cache[layer_idx]   # [batch, heads, seq, dim]
        v = past_key_values.value_cache[layer_idx]
    # Tuple of tuples: ((k, v), (k, v), ...)
    elif isinstance(past_key_values, (tuple, list)):
        k, v = past_key_values[layer_idx]
    else:
        raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")

    # batch dim 제거, position 선택
    return k[0, :, position, :].detach(), v[0, :, position, :].detach()


def get_num_layers(past_key_values) -> int:
    """KV cache의 layer 수 반환."""
    if hasattr(past_key_values, 'key_cache'):
        return len(past_key_values.key_cache)
    elif isinstance(past_key_values, (tuple, list)):
        return len(past_key_values)
    else:
        raise TypeError(f"Unsupported type: {type(past_key_values)}")


def compute_layer_mapping(
    num_draft_layers: int,
    num_target_layers: int,
) -> List[Tuple[int, int]]:
    """
    Draft layer와 Target layer 간 비례 매핑 생성.

    예시:
    - Draft 16 layers, Target 32 layers → [(0,0), (1,2), (2,4), ..., (15,30)]
    - Draft 16 layers, Target 48 layers → [(0,0), (1,3), (2,6), ..., (15,45)]

    Returns:
        List of (draft_layer_idx, target_layer_idx) pairs
    """
    mapping = []
    for d in range(num_draft_layers):
        # 비례 매핑: draft의 상대적 위치를 target으로 변환
        t = round(d * (num_target_layers - 1) / (num_draft_layers - 1)) if num_draft_layers > 1 else 0
        mapping.append((d, t))
    return mapping


def compare_kv_at_position(
    draft_past_kv,
    target_past_kv,
    position: int,
    layer_mapping: Optional[List[Tuple[int, int]]] = None,
) -> KVSimilarityResult:
    """
    특정 position에서 draft와 target의 KV cache를 layer별, head별로 비교.

    Draft와 Target의 차이를 처리하는 방법:
    - Layer 수 차이: layer_mapping에 따라 비례 매핑
    - KV head 수 차이 (GQA): min(draft_heads, target_heads)까지 비교
    - Head dim 차이: min(draft_dim, target_dim)까지 truncation

    Args:
        draft_past_kv: draft model의 past_key_values
        target_past_kv: target model의 past_key_values
        position: 비교할 sequence position
        layer_mapping: (draft_layer, target_layer) 쌍 리스트
    """
    num_draft_layers = get_num_layers(draft_past_kv)
    num_target_layers = get_num_layers(target_past_kv)

    if layer_mapping is None:
        layer_mapping = compute_layer_mapping(num_draft_layers, num_target_layers)

    result = KVSimilarityResult()

    for draft_l, target_l in layer_mapping:
        # KV 추출: [num_kv_heads, head_dim]
        dk, dv = extract_kv_at_position(draft_past_kv, draft_l, position)
        tk, tv = extract_kv_at_position(target_past_kv, target_l, position)

        # Head 수 맞추기
        min_heads = min(dk.shape[0], tk.shape[0])
        # Head dim 맞추기
        min_dim = min(dk.shape[1], tk.shape[1])

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
    Random vector와 target KV 간 cosine similarity를 측정.
    "Rejected KV가 random보다 얼마나 좋은가"의 baseline.

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
