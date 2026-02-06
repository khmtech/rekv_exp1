"""
datasets_loader.py - Benchmark Dataset Loading
================================================
MT-Bench, GSM8K, HumanEval, 그리고 debug용 프롬프트를 로드.
"""

from typing import List


def load_prompts(
    dataset_name: str,
    num_samples: int,
    tokenizer=None,
) -> List[str]:
    """
    데이터셋에서 프롬프트를 로드.

    Args:
        dataset_name: "debug", "mt_bench", "gsm8k", "humaneval", "alpaca"
        num_samples: 최대 sample 수
        tokenizer: (optional) chat template 적용용

    Returns:
        List of prompt strings
    """
    loader_map = {
        "debug": _load_debug,
        "mt_bench": _load_mt_bench,
        "gsm8k": _load_gsm8k,
        "humaneval": _load_humaneval,
        "alpaca": _load_alpaca,
    }

    if dataset_name not in loader_map:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(loader_map.keys())}"
        )

    prompts = loader_map[dataset_name](num_samples)
    print(f"  Loaded {len(prompts)} prompts from '{dataset_name}'")
    return prompts


def _load_debug(n: int) -> List[str]:
    """디버깅용 간단한 프롬프트."""
    prompts = [
        "Write a Python function to sort a list using merge sort.",
        "Explain the theory of relativity in simple terms.",
        "Solve step by step: If x + 3 = 10, what is x?",
        "What are the main differences between TCP and UDP?",
        "Write a short story about a robot discovering music.",
        "Describe how photosynthesis works.",
        "What is the capital of France and why is it historically significant?",
        "Calculate the derivative of f(x) = 3x^2 + 2x - 5.",
        "Explain recursion with a simple example.",
        "Write a SQL query to find the second highest salary.",
    ]
    return prompts[:n]


def _load_mt_bench(n: int) -> List[str]:
    """MT-Bench 프롬프트 (HuggingFace에서 로드 시도, 실패 시 fallback)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        prompts = []
        for item in ds:
            if "prompt" in item:
                p = item["prompt"]
                prompts.append(p[0] if isinstance(p, list) else p)
            elif "turns" in item:
                prompts.append(item["turns"][0])
        if len(prompts) >= n:
            return prompts[:n]
    except Exception as e:
        print(f"  Warning: MT-Bench load failed ({e}), using fallback")

    return _fallback_diverse(n)


def _load_gsm8k(n: int) -> List[str]:
    """GSM8K 수학 문제."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        prompts = [
            f"Solve this step by step:\n{item['question']}"
            for item in ds
        ]
        return prompts[:n]
    except Exception as e:
        print(f"  Warning: GSM8K load failed ({e}), using math fallback")
        return _fallback_math(n)


def _load_humaneval(n: int) -> List[str]:
    """HumanEval 코드 완성."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test")
        prompts = [item["prompt"] for item in ds]
        return prompts[:n]
    except Exception as e:
        print(f"  Warning: HumanEval load failed ({e}), using code fallback")
        return _fallback_code(n)


def _load_alpaca(n: int) -> List[str]:
    """Alpaca 데이터셋 (다양한 instruction)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        prompts = []
        for item in ds:
            instruction = item["instruction"]
            inp = item.get("input", "")
            if inp:
                prompts.append(f"{instruction}\n\nInput: {inp}")
            else:
                prompts.append(instruction)
        return prompts[:n]
    except Exception as e:
        print(f"  Warning: Alpaca load failed ({e}), using fallback")
        return _fallback_diverse(n)


# ============================================================
# Fallback prompts
# ============================================================

def _fallback_diverse(n: int) -> List[str]:
    """다양한 도메인의 fallback 프롬프트."""
    base = [
        "Write a Python function that implements binary search on a sorted list.",
        "Explain quantum computing to a 10-year-old.",
        "What are the main causes and effects of climate change?",
        "If a train travels 60 mph for 2.5 hours, how far does it go? Show your work.",
        "Write a short story about an AI that learns empathy.",
        "Describe the process of photosynthesis in detail.",
        "Compare TCP and UDP protocols with examples.",
        "Find the derivative of f(x) = 3x^3 - 2x^2 + 5x - 7.",
        "Explain recursion with a practical coding example.",
        "What were the three most important events of World War II?",
        "Write a SQL query to find the top 5 customers by purchase amount.",
        "How does the human immune system fight viral infections?",
        "Compare supervised and unsupervised machine learning with examples.",
        "Write a JavaScript debounce function with explanation.",
        "Explain supply and demand with a real-world example.",
        "What are the benefits and risks of nuclear energy?",
        "Describe the architecture of a modern web application.",
        "How do neural networks learn via backpropagation?",
        "Write a regular expression to validate email addresses and explain it.",
        "Explain the Pythagorean theorem and give 3 real-world applications.",
    ]
    # 반복해서 n개 채우기
    result = []
    while len(result) < n:
        result.extend(base)
    return result[:n]


def _fallback_math(n: int) -> List[str]:
    base = [
        "Solve step by step: A store sells apples for $2 each. If you buy 5 apples and pay with a $20 bill, how much change do you get?",
        "Solve step by step: A rectangle has a perimeter of 40cm. If the length is 12cm, what is the width?",
        "Solve step by step: If 3x + 7 = 22, what is x?",
        "Solve step by step: A car travels 180 miles in 3 hours. What is the average speed?",
        "Solve step by step: A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices are left?",
    ]
    result = []
    while len(result) < n:
        result.extend(base)
    return result[:n]


def _fallback_code(n: int) -> List[str]:
    base = [
        "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
        "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if string is a palindrome.\"\"\"\n",
        "def flatten(lst: list) -> list:\n    \"\"\"Flatten a nested list.\"\"\"\n",
        "def binary_search(arr: list, target: int) -> int:\n    \"\"\"Return index of target in sorted array, or -1.\"\"\"\n",
        "def merge_sort(arr: list) -> list:\n    \"\"\"Sort array using merge sort.\"\"\"\n",
    ]
    result = []
    while len(result) < n:
        result.extend(base)
    return result[:n]
