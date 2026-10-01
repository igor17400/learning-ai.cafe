---
title: "A Decision With No Label"
subtitle: "The router picks an expert, and nothing tells it which one was right."
---

The router is just a single matrix. It is physically smaller than a single expert, visually the least interesting object in the layer, and yet it decides everything. It dictates which experts are trained, on what data, and ultimately, what those experts become.

This page follows a single token through that decision process, tracing it forwards and then backwards. For here we shall keep explanations a bit easier to follow so there is only one token, one router, and nothing is allowed to go wrong yet.

## One token through the router

The router is the gating network of [Committee or Competition](../competing_experts/), stripped to its simplest form: a single weight matrix $W_g \in \mathbb{R}^{N \times d}$, one row per expert, with no hidden layers and usually no bias. For a token $\mathbf{x} \in \mathbb{R}^{d}$ it produces one score per expert,

$$
\mathbf{s} = W_g\,\mathbf{x} \;\in\; \mathbb{R}^{N}
$$

and the score for expert $i$ is the inner product of the token with that expert's row. An expert's row is a direction in representation space, and the tokens that expert receives are the ones pointing that way.

The scores then pass through the two steps from [Compute You Can Skip](../conditional_computation/), the mask and the softmax, and what comes out is $k$ non-zero gate values that sum to one.

The figure below traces this exact sequence for a layer with eight experts ($N=8$) where we only evaluate two ($k=2$). The initial multiplication yields eight raw logits. The KeepTopK operation intervenes, identifying the two highest scores (here, 2.7 and 2.1) and pushing the rest to negative infinity. When the softmax is applied, those negative infinities collapse to zero, while the top two scores re-normalize into final gate values of 0.65 and 0.35. The token is physically dispatched only to $E_3$ and $E_6$, and their outputs are ultimately combined using those exact weights.

:::figure{#router_path}
![One token multiplied by the router matrix gives eight logits. The top two are kept and the other six are set to minus infinity, then a softmax over what remains gives gate values of 0.65 and 0.35 with six zeros. Arrows send the token to experts three and six.](../../figures/router_path.svg)

The whole router, at $N = 8$ and $k = 2$. Six of the eight decisions are made by a comparison that never appears in the loss.
:::

Notice the extreme asymmetry in scale. The routing matrix $W_g$ contains exactly $dN$ parameters. At $d = 4096$ and $N = 64$, that amounts to just 262,144 parameters, compared to the 8.59 billion sitting in the experts it chooses between. The router is roughly thirty thousand times smaller than the network it controls.

## Where the gradient goes

Now let's go through the backward pass. The layer's output is $y = \sum_i g_i(\mathbf{x})\, E_i(\mathbf{x})$, so the loss reaches the layer through $y$, and from there it has three routes.

1. **Into the chosen experts**, scaled by their gate values. Expert $E_3$ receives $0.65$ of the correction and $E_6$ receives $0.35$. This is the same responsibility-weighted update as [Committee or Competition](../competing_experts/), with the responsibility replaced by the gate.
2. **Into the gate values**, and through them into $W_g$. The derivative with respect to $g_i$ is $E_i(\mathbf{x})$, so the router is told to raise the weight on this expert if its output was in the direction the loss wanted, lower it otherwise. The router learns which of the experts it tried was better.
3. **Nowhere else.** $\operatorname{KeepTopK}$ is piecewise constant, so its derivative is zero everywhere it is defined. No gradient passes through the act of choosing, and the six experts that were not chosen appear nowhere in the graph.

:::figure{#gradient_reach}
![The forward chain from token to router matrix to logits to the top-k selection to the gate values to the chosen experts to the output. Underneath, dashed backward arcs run right to left along the same chain, reaching the chosen experts, the gate values and the router matrix. Hanging below the selection is a greyed box of the six experts that were not chosen, with the arrow that would connect them crossed out.](../../figures/gradient_reach.svg)

The router is trained only on the experts it already picked. $W_g$ is corrected through the two that ran and through the gate values that weighted them, and through nothing else. An expert that is never picked receives no gradient, and the router is told nothing about it either.
:::

Follow that third point through, because it is the defect that later sections try to solve. An expert that is never chosen is never evaluated, so it is never trained, so it never improves, so it goes on not being chosen. And the router is not told it is missing anything since it receives no signal from an expert it did not try, so nothing in the gradient ever says _expert 5 would have been better here_. ==The router can only learn to prefer an expert it has already tried==, and the set it tries is the set it already prefers.

## Noise, and what it buys

The original sparsely-gated layer breaks that loop with the bluntest available instrument. It adds tunable noise to the scores before the top-$k$:

$$
H(\mathbf{x})_i = (\mathbf{x} \cdot W_g)_i + \operatorname{StandardNormal}() \cdot \operatorname{Softplus}\big((\mathbf{x} \cdot W_{\text{noise}})_i\big)
$$

- $W_{\text{noise}} \in \mathbb{R}^{N \times d}$ is a **second learned matrix**, the same shape as $W_g$, giving a per-token per-expert noise scale.
- $\operatorname{Softplus}$ keeps that scale positive, and being learned means the model can choose to be noisy about some experts and confident about others.

The noise buys two different things, and the second is the one that matters later.

The first is exploration. An expert whose score is close behind the top $k$ will occasionally be sampled into it, get evaluated, get trained, and be judged on its merits. It is a way of trying experts the router would not have tried.

The second is that it makes a hard count differentiable. We are about to want a loss that depends on _how many tokens_ land on each expert, and a count has no useful derivative: nudge $W_g$ a little and the count either does not move at all or jumps by one. With noise in the scores, the probability that a given token lands in the top $k$ of a given expert is a smooth function of $W_g$ and $W_{\text{noise}}$, and a smooth estimate of a count is something we can put in a loss. _Teaching the Router to Share_ does exactly that.

## A decision with no label

Step back from the mechanics and look at what has been built. The router is a classifier over $N$ classes. It takes a vector, produces $N$ scores, and picks the top few, which is what a classifier does.

What it does not have is a training set. Nobody knows which expert should have received this token. There is no label, there is no oracle, and there is not even a well-posed question, because which expert _should_ handle a token depends on what the experts have already learned, which depends on the routing so far. The target moves as the router moves.

So the router is trained on a proxy: whether the experts it happened to pick produced an output the loss liked. That proxy is silent about every expert it did not pick, blind to the choice itself, and happy with any assignment that works, including one that sends every token to the same place.

_Teaching the Router to Share_ adds a second term to the loss whose only job is to object to that last outcome. It is worth remembering, when we get there, that a balance penalty is not a refinement of the routing objective. It is a substitute for one.
