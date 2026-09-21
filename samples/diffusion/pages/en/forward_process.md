---
title: "The Forward Process"
subtitle: "Adding noise, one step at a time."
---

## A Markov chain of noise

Sample prose so the layout can be judged before the real text exists. A model reads a sentence and has to answer a question about it, and ==the one sentence worth highlighting sits here==. The score between a query $q$ and a key $k$ is the dot product $q \cdot k$, and the weights come from a softmax:

$$
\alpha_j = \frac{e^{s_j}}{\sum_i e^{s_i}}, \qquad s_j = \frac{q \cdot k_j}{\sqrt{d_k}}
$$

A second paragraph, because pages are mostly paragraphs. It has a [link](<https://en.wikipedia.org/wiki/Attention_(machine_learning)>), some `inline code`, and a list:

1. **The request.** What the model asks for.
2. **The reply.** What every word hands back.
3. **The score.** How well a reply answers the request.

:::figure{#noising}
![Sample figure standing in for noising.](../../figures/noising.svg)

After enough steps only noise is left.
:::

## The closed form

Sample prose so the layout can be judged before the real text exists. A model reads a sentence and has to answer a question about it, and ==the one sentence worth highlighting sits here==. The score between a query $q$ and a key $k$ is the dot product $q \cdot k$, and the weights come from a softmax:
:::aside[Why the square root]
A collapsible note for something that would break the flow. Variance of a dot product of two $d_k$-dimensional unit-variance vectors is $d_k$, so dividing by $\sqrt{d_k}$ keeps the scores at unit scale.
:::

```python
import torch

scores = Q @ K.T / d_k**0.5
weights = scores.softmax(dim=-1)
out = weights @ V
```

| word  | score | weight |
| ----- | ----: | -----: |
| the   |  0.90 |   0.04 |
| cat   |  2.80 |   0.60 |
| mouse |  0.20 |   0.12 |
