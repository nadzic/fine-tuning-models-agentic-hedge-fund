# Error Log

## INTC
- Type: hallucinated business context
- Severity: high
- Problematic claim: "INTC operates in the semiconductor equipment and foundry services business."
- Why it is wrong: The prompt facts do not support this business description.
- Fix needed: The model should describe only the reported financial and valuation setup, without inventing business context.

## NVDA
- Type: unsupported business expansion
- Severity: medium
- Problematic claim: broader company background beyond the facts
- Why it is an issue: The output adds framing not explicitly provided in the prompt.
- Fix needed: Keep the overview tied to revenue growth, AI demand, margins, valuation, and risk only.

## ADBE
- Type: unsupported strategic framing
- Severity: medium
- Problematic claim: subscription / recurring-revenue style explanation not explicitly stated in the facts
- Why it is an issue: Sounds plausible, but is not grounded in the prompt.
- Fix needed: Stay with growth, margins, free cash flow, valuation, and risks only.

## TSLA
- Type: over-interpretation
- Severity: medium
- Problematic claim: strategic interpretation beyond the facts
- Why it is an issue: The model adds reasoning that may be sensible, but is not directly supported.
- Fix needed: Stay with slower growth, compressed margin, scale/brand, valuation difficulty, and key risks.

## XYZ
- Type: none
- Severity: low
- Strength: good uncertainty calibration
- Why it is good: The model avoids inventing details when evidence is thin.
- Keep: Preserve this behavior in the next run.
