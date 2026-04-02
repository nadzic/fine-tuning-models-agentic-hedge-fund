# Fine-Tuned Model Evaluation Rubric

| Eval case | Fact fidelity | Hallucination control | Uncertainty calibration | Investment usefulness | Format consistency | Total /10 | Notes |
|-----------|---------------|-----------------------|-------------------------|-----------------------|--------------------|-----------|-------|
| NVDA      | 1             | 1                     | 2                       | 2                     | 2                  | 8/10      | Strong structure and useful summary, but it adds broader company background and some inferred framing not directly stated in the facts. |
| ADBE      | 1             | 1                     | 2                       | 2                     | 2                  | 8/10      | Solid and fairly disciplined output, but it introduces subscription and cash-conversion framing that is not explicitly supported by the prompt. |
| INTC      | 0             | 0                     | 2                       | 1                     | 2                  | 5/10      | Critical factual error: the business description is inaccurate. This is the clearest production-risk example in the set. |
| TSLA      | 1             | 1                     | 2                       | 2                     | 2                  | 8/10      | Good balanced memo structure, but it still adds strategic interpretation beyond the provided facts. |
| XYZ       | 2             | 2                     | 2                       | 2                     | 2                  | 10/10     | Best example in the set. The model stays appropriately constrained and does not hallucinate when the evidence is thin. |

## Scoring Guide

- `Fact fidelity`
  - `2`: All key claims are directly supported by the provided facts
  - `1`: Minor unsupported extensions, but no major factual mistake
  - `0`: One or more important claims are unsupported or incorrect

- `Hallucination control`
  - `2`: No invented business, financial, or strategic details
  - `1`: One minor unsupported but generic addition
  - `0`: Multiple unsupported additions or a clearly fabricated detail

- `Uncertainty calibration`
  - `2`: Appropriate caution when evidence is incomplete
  - `1`: Slightly overconfident or slightly too vague
  - `0`: Overconfident despite weak or incomplete evidence

- `Investment usefulness`
  - `2`: Clearly helps frame the bullish, bearish, and risk case
  - `1`: Somewhat useful but too generic
  - `0`: Weak decision support value

- `Format consistency`
  - `2`: Fully consistent with the intended memo format
  - `1`: Minor formatting inconsistency
  - `0`: Inconsistent or unclear structure

## Suggested Release Gate

- Average score of at least `8/10`
- No example with `Fact fidelity = 0`
- Low-confidence cases should score `2` on `Uncertainty calibration`