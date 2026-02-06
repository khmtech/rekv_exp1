# ReKV Experiment 1: Rejected KV Similarity Analysis

## 개요

Speculative Decoding(SD)에서 rejected token의 KV cache가 target model의 ground-truth KV와 얼마나 유사한지 정량 분석하는 실험입니다. 이 결과가 ReKV 논문의 **핵심 motivation figure**가 됩니다.

## 프로젝트 구조

```
rekv_exp1/
├── run.py              # 메인 entry point (--preset으로 간편 실행)
├── spec_decode.py      # Instrumented SD loop (KV 수집 내장)
├── kv_analyzer.py      # KV 추출, 비교, similarity 계산
├── datasets_loader.py  # MT-Bench, GSM8K, HumanEval 로더
├── stats.py            # 통계 계산, GO/NO-GO 판정
├── plots.py            # 논문용 figure 생성 (6종)
├── scripts.sh          # 실험 실행 스크립트
└── README.md
```

## 모듈 설명

| 모듈 | 역할 |
|------|------|
| `kv_analyzer.py` | Draft/Target KV cache에서 position-level KV 추출, cosine similarity / L2 distance 계산. Layer 수/head 수/head dim 차이 자동 처리. |
| `spec_decode.py` | SD loop 실행. 매 rejection마다 `kv_analyzer`를 호출하여 해당 position의 draft KV vs target KV 비교. Greedy decoding 사용. |
| `stats.py` | 수집된 event들의 통계: position별, layer별, confidence별 breakdown + GO/NO-GO 판정. |
| `plots.py` | 6종의 figure 생성. fig1이 논문 핵심 figure (KV similarity 분포). |

## 환경 설정

```bash
conda create -n rekv python=3.10 -y && conda activate rekv
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate datasets tqdm matplotlib numpy
huggingface-cli login  # LLaMA 접근용
```

## 실행

```bash
# 1. Sanity check (~5분)
python run.py --preset debug

# 2. Preliminary results (~30분)
python run.py --preset quick

# 3. 논문용 full experiment (~4-8시간)
python run.py --preset full

# 4. Speculation length sweep (~6시간)
python run.py --preset sweep

# 5. 데이터셋 비교
chmod +x scripts.sh
./scripts.sh all-datasets

# 6. 다른 모델 쌍
python run.py \
    --draft_model Qwen/Qwen2.5-1.5B \
    --target_model Qwen/Qwen2.5-7B-Instruct \
    --dataset mt_bench --num_samples 50 --spec_length 5 \
    --output_dir results/qwen
```

## 출력

```
results/
├── K5/
│   ├── summary_K5.json      # 집계 통계 + GO/NO-GO
│   ├── events_K5.json       # 모든 rejection event 상세
│   └── plots/
│       ├── fig1_kv_similarity_distribution.pdf  ← 핵심 figure
│       ├── fig2_per_layer_similarity.pdf
│       ├── fig3_confidence_vs_similarity.pdf
│       ├── fig4_position_vs_similarity.pdf
│       ├── fig5_rank_histogram.pdf
│       └── fig6_rejection_position_distribution.pdf
└── combined_results.json    # 여러 K 값 비교
```

## GO/NO-GO 기준

| Mean KV Cosine | 판정 | 조치 |
|----------------|------|------|
| > 0.7 | ✅ GO | ReKV 본격 구현 |
| 0.5 ~ 0.7 | ⚠️ CAUTION | Selective recycling (특정 layer/confidence만) |
| < 0.5 | ❌ NO-GO | 주제 전환 또는 원인 분석 |

## Figure 해석 가이드

- **fig1 (Distribution):** Rejected KV의 cosine similarity가 높게 분포하면 "rejected KV는 유용한 근사"
- **fig2 (Per-layer):** 어느 layer에서 similarity가 높은지 → recycling 대상 layer 선정
- **fig3 (Confidence):** 높은 confidence rejection일수록 similarity 높으면 → confidence 기반 selective recycling 근거
- **fig4 (Position):** Window 앞쪽 rejection vs 뒤쪽 → position에 따른 recycling 전략
- **fig5 (Rank):** Rejected token이 target에서 2등, 3등이면 → KV도 유사할 개연성
- **fig6 (Rejection distribution):** 어느 position에서 rejection이 가장 많은지

## Design Decisions

1. **Greedy decoding:** Sampling 시 rejection이 stochasticity인지 capability gap인지 구분 불가. Greedy에서의 rejection은 순수 모델 차이를 반영.

2. **Target full forward (no KV cache reuse):** Target의 ground-truth KV를 모든 position에서 정확히 얻기 위해 매 round 전체 forward pass. 분석 목적이므로 overhead 허용.

3. **Layer proportional mapping:** Draft 16 layers ↔ Target 32 layers에서 draft[i] ↔ target[round(i * 31/15)] 매핑. 선형 보간이 가장 일반적.

4. **Head 수 차이 처리:** GQA로 인해 draft와 target의 KV head 수가 다를 때 min(heads) 사용.

## Troubleshooting

| 문제 | 해결 |
|------|------|
| CUDA OOM | `--max_new_tokens 64` 또는 `--max_prompt_len 256` |
| 데이터셋 로드 실패 | `--dataset debug` 사용 또는 HuggingFace 로그인 확인 |
| Similarity ≈ 0 | Layer mapping 확인. 모델 아키텍처가 호환되는지 체크 |
| "No rejection events" | 모델 쌍이 너무 유사 (acceptance rate ~100%). Spec length 늘리거나 다른 데이터셋 시도 |
