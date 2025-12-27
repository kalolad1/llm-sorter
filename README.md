# llm-sorter

A Python library that uses LLMs as comparators for semantic sorting. Sort any list by meaning, tone, complexity, urgency, or any criteria expressible in natural language. This library uses
an OpenRouter API key to interface with LLM providers.

## Installation

```bash
pip install llm-sorter
```

Requires Python 3.10+

## Quick Start

```python
from llm_sorter import LLMSorter

sorter = LLMSorter(api_key="your-openrouter-api-key")

# Sort by reading complexity
passages = [
    "The quantum entanglement phenomenon demonstrates non-local correlations...",
    "The cat sat on the mat.",
    "Climate change affects global weather patterns in complex ways."
]
result = sorter.sort(
    items=passages,
    prompt="Sort by reading level, simplest first"
)

# Sort support tickets by urgency
tickets = [
    "Login button color looks off",
    "CRITICAL: Production database is down, all users affected",
    "Would be nice to have dark mode"
]
urgent_first = sorter.sort(
    items=tickets,
    prompt="Sort by urgency, most critical first"
)
```

## How It Works

llm-sorter implements merge sort with LLM-powered comparisons. Instead of numeric comparison, the model decides which of two items should come first based on your prompt. It will use the passed in objects `__str__` dunder method to formulate the prompt.

This enables sorting by criteria that have no numeric representation: persuasiveness, professionalism, emotional intensity, policy compliance, or any semantic property.

It can also enable the sorting of large lists that cannot be sorted within a single
LLM call (with the tradeoff being significant more API calls, and the associated cost
and latency of each call).

## When to Use This

**Good fit:**
- Semantic/subjective ordering (urgency, quality, tone, reading level)
- No training data available — works zero-shot
- Rapid prototyping or internal tools
- Small to medium lists (n < 100-300)
- Approximate "human-like" rankings are acceptable

**Examples:**
- Sort writing samples by reading level
- Prioritize support tickets by urgency
- Rank answers by helpfulness
- Order reviews by sentiment strength

## When NOT to Use This

**Avoid when:**
- You need strict determinism or reproducibility
- A simple numeric key exists (timestamps, severity levels)
- Very large n — O(n log n) API calls get expensive

## Limitations

1. **Non-transitive comparisons**: LLMs can produce A > B, B > C, C > A cycles
2. **Non-deterministic**: Results may vary between runs
3. **Cost at scale**: 100 items ≈ 664 comparisons, 1000 items ≈ 10,000 comparisons
4. **Prompt sensitivity**: Small wording changes can affect results
5. **Potential bias**: Length, style, and positional biases exist

## License

MIT
