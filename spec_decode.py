"""
spec_decode.py - Instrumented Speculative Decoding Loop
========================================================

Draft-Target SD를 실행하면서 모든 rejection 지점에서 KV 유사도를 측정.

핵심 구조:
1. Draft model: autoregressive로 K tokens 생성 (past_kv 보관)
2. Target model: prefix + draft tokens를 한번에 forward pass (past_kv 보관)
3. Rejection 발생 시: draft의 past_kv와 target의 past_kv를 동일 position에서 비교
4. 모든 결과를 RejectionEvent / RoundStats로 수집

주의사항:
- Greedy decoding 사용 (argmax). Sampling은 rejection의 원인이 stochasticity인지
  model capability gap인지 구분 불가. Greedy에서의 rejection은 순수하게 모델 차이.
- Target verification 시 전체 sequence를 재계산 (KV cache 없이).
  이유: target의 "ground truth" KV를 모든 position에서 얻기 위해.
  성능이 아닌 분석이 목적이므로 이 overhead는 허용.
"""

import time
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from kv_analyzer import (
    RejectionEvent,
    RoundStats,
    KVSimilarityResult,
    compare_kv_at_position,
    compute_layer_mapping,
    get_num_layers,
    compute_random_baseline,
)


def _normalize_past_kv(past_key_values):
    """
    Normalize HF cache types to legacy tuple/list structure when available.
    """
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    if hasattr(past_key_values, "cache"):
        return past_key_values.cache
    return past_key_values


class InstrumentedSpecDecoder:
    """
    KV 분석이 내장된 Speculative Decoding 실행기.

    사용법:
        decoder = InstrumentedSpecDecoder(draft_model, target_model, tokenizer, ...)
        decoder.run(input_ids, sample_id=0, spec_length=5)
        events = decoder.rejection_events  # List[RejectionEvent]
        stats = decoder.round_stats        # List[RoundStats]
    """

    def __init__(
        self,
        draft_model,
        target_model,
        tokenizer,
        device: str = "cuda",
        collect_per_head: bool = True,
        collect_random_baseline: bool = False,
        random_baseline_samples: int = 50,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.device = device
        self.collect_per_head = collect_per_head
        self.collect_random_baseline = collect_random_baseline
        self.random_baseline_samples = random_baseline_samples

        # Layer mapping 사전 계산 (첫 실행 시 결정)
        self._layer_mapping = None

        # 수집된 결과
        self.rejection_events: List[RejectionEvent] = []
        self.round_stats: List[RoundStats] = []
        self.random_baselines: List[dict] = []

    def _ensure_layer_mapping(self, draft_past_kv, target_past_kv):
        """첫 실행 시 layer mapping을 계산."""
        if self._layer_mapping is None:
            n_draft = get_num_layers(draft_past_kv)
            n_target = get_num_layers(target_past_kv)
            self._layer_mapping = compute_layer_mapping(n_draft, n_target)

    @torch.no_grad()
    def run(
        self,
        input_ids: torch.Tensor,
        sample_id: int,
        spec_length: int = 5,
        max_new_tokens: int = 128,
    ) -> torch.Tensor:
        """
        한 sample에 대해 speculative decoding을 실행하고 KV 분석 데이터를 수집.

        Args:
            input_ids: [1, prompt_len] - 이미 토큰화된 입력
            sample_id: 로깅용 sample index
            spec_length: K - speculation window size
            max_new_tokens: 최대 생성 token 수

        Returns:
            generated_ids: [1, prompt_len + generated_len]
        """
        input_ids = input_ids.to(self.device)
        current_ids = input_ids.clone()
        generated_count = 0
        round_id = 0

        while generated_count < max_new_tokens:
            # ============================================================
            # Phase 1: Draft generation (autoregressive, K tokens)
            # ============================================================
            t0 = time.perf_counter()
            draft_tokens, draft_logits, draft_past_kv = self._draft_generate(
                current_ids, spec_length
            )
            t_draft = (time.perf_counter() - t0) * 1000

            if len(draft_tokens) == 0:
                break

            actual_K = len(draft_tokens)

            # ============================================================
            # Phase 2: Target verification (single forward pass)
            # ============================================================
            # Target에 prefix + draft tokens를 한번에 넣어서 forward
            # use_cache=False가 아닌 use_cache=True로 past_kv를 받음
            verify_ids = torch.cat([
                current_ids,
                torch.tensor([draft_tokens], device=self.device, dtype=torch.long)
            ], dim=1)

            t0 = time.perf_counter()
            target_out = self.target_model(
                input_ids=verify_ids,
                use_cache=True,
                return_dict=True,
            )
            t_verify = (time.perf_counter() - t0) * 1000

            target_logits = target_out.logits           # [1, total_len, vocab]
            target_past_kv = _normalize_past_kv(target_out.past_key_values)

            # Layer mapping 결정
            self._ensure_layer_mapping(draft_past_kv, target_past_kv)

            # ============================================================
            # Phase 3: Token-by-token verification + rejection analysis
            # ============================================================
            prefix_len = current_ids.shape[1]
            accept_length = 0
            rejected = False

            for i in range(actual_K):
                # Target's logit for position (prefix_len + i):
                #   target_logits[0, prefix_len + i - 1, :] predicts token at prefix_len + i
                target_logit_i = target_logits[0, prefix_len + i - 1, :]
                target_probs_i = F.softmax(target_logit_i.float(), dim=-1)
                target_token_i = target_probs_i.argmax().item()

                draft_token_i = draft_tokens[i]
                draft_logit_i = draft_logits[i]
                draft_probs_i = F.softmax(draft_logit_i.float(), dim=-1)
                draft_conf = draft_probs_i[draft_token_i].item()
                target_conf = target_probs_i[target_token_i].item()

                if draft_token_i == target_token_i:
                    # Accepted
                    accept_length += 1
                else:
                    # REJECTED - KV 분석 수행
                    rejected = True

                    # Draft token의 target 분포에서의 rank 계산
                    sorted_indices = target_probs_i.argsort(descending=True)
                    rank = (sorted_indices == draft_token_i).nonzero(as_tuple=True)[0]
                    rank = rank.item() if len(rank) > 0 else -1

                    # KV similarity 비교
                    seq_pos = prefix_len + i  # 전체 sequence에서의 position
                    kv_sim = compare_kv_at_position(
                        draft_past_kv=draft_past_kv,
                        target_past_kv=target_past_kv,
                        position=seq_pos,
                        layer_mapping=self._layer_mapping,
                    )

                    # Random baseline (optional)
                    if self.collect_random_baseline:
                        baseline = compute_random_baseline(
                            target_past_kv, seq_pos,
                            num_samples=self.random_baseline_samples
                        )
                        self.random_baselines.append(baseline)

                    event = RejectionEvent(
                        sample_id=sample_id,
                        round_id=round_id,
                        spec_length=actual_K,
                        accept_length=accept_length,
                        rejection_pos_in_window=i,
                        rejection_pos_in_seq=seq_pos,
                        draft_token_id=draft_token_i,
                        target_token_id=target_token_i,
                        draft_confidence=draft_conf,
                        target_confidence=target_conf,
                        draft_token_rank_in_target=rank,
                        kv_sim=kv_sim,
                    )
                    self.rejection_events.append(event)
                    break

            # ============================================================
            # Phase 4: Sequence update
            # ============================================================
            if accept_length > 0:
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([draft_tokens[:accept_length]], device=self.device)
                ], dim=1)
                generated_count += accept_length

            if rejected:
                # Correction token from target at rejection point
                correction_pos = prefix_len + accept_length - 1
                correction_logit = target_logits[0, correction_pos, :]
                correction_token = correction_logit.argmax().item()
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([[correction_token]], device=self.device)
                ], dim=1)
                generated_count += 1
                if correction_token == self.tokenizer.eos_token_id:
                    break
            else:
                # All accepted → bonus token
                bonus_logit = target_logits[0, -1, :]
                bonus_token = bonus_logit.argmax().item()
                current_ids = torch.cat([
                    current_ids,
                    torch.tensor([[bonus_token]], device=self.device)
                ], dim=1)
                generated_count += 1
                if bonus_token == self.tokenizer.eos_token_id:
                    break

            # Round stats
            self.round_stats.append(RoundStats(
                sample_id=sample_id,
                round_id=round_id,
                spec_length=actual_K,
                accept_length=accept_length,
                all_accepted=(not rejected),
                draft_time_ms=t_draft,
                verify_time_ms=t_verify,
            ))

            round_id += 1

            # GPU memory management: 명시적 KV cache 해제
            del draft_past_kv, target_past_kv, target_out
            if round_id % 5 == 0:
                torch.cuda.empty_cache()

        return current_ids

    def _draft_generate(
        self,
        prefix_ids: torch.Tensor,
        spec_length: int,
    ) -> Tuple[List[int], List[torch.Tensor], object]:
        """
        Draft model로 K tokens을 autoregressive하게 생성.

        Draft의 past_key_values는 매 step 누적되므로, 최종 past_kv에는
        prefix + 모든 draft tokens의 KV가 포함됨.

        Returns:
            tokens: list of int, 생성된 token IDs
            logits: list of Tensor [vocab_size], 각 step의 logits
            past_kv: 최종 past_key_values (prefix + all draft tokens 포함)
        """
        tokens = []
        logits_list = []
        past_kv = None
        current = prefix_ids

        for step in range(spec_length):
            # 첫 step: 전체 prefix 입력
            # 이후 step: 마지막 1 token만 입력 + past_kv
            if past_kv is None:
                out = self.draft_model(
                    input_ids=current,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                out = self.draft_model(
                    input_ids=current[:, -1:],
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )

            logit = out.logits[0, -1, :]  # [vocab]
            past_kv = _normalize_past_kv(out.past_key_values)
            token = logit.argmax().item()

            tokens.append(token)
            logits_list.append(logit.detach())

            # 다음 step을 위해 token 추가
            current = torch.cat([
                current,
                torch.tensor([[token]], device=self.device)
            ], dim=1)

            if token == self.tokenizer.eos_token_id:
                break

        return tokens, logits_list, past_kv

    def clear(self):
        """수집된 결과 초기화."""
        self.rejection_events.clear()
        self.round_stats.clear()
        self.random_baselines.clear()
