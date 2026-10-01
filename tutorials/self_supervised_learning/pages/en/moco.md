---
title: "MoCo"
subtitle: "Momentum contrast and the dictionary-as-queue mechanism."
---

## Decoupling the dictionary from the batch

> "Contrastive learning is a way of building a discrete dictionary on high-dimensional continuous inputs such as images. The dictionary is dynamic in the sense that the keys are randomly sampled, and that the key encoder evolves during training. Our hypothesis is that good features can be learned by a large dictionary that covers a rich set of negative samples, while the encoder for the dictionary keys is kept as consistent as possible despite its evolution."

[He et al. (2020)](https://arxiv.org/abs/1911.05722) propose to view contrastive learning as **dictionary lookup**. An encoded query $q$ is compared against a set of encoded keys $\{k_0, k_1, \ldots, k_K\}$. One key, $k_+$, is the positive (it matches the query); the remaining $K$ keys are negatives. Training succeeds when the query is most similar to its positive key.

From the [InfoNCE section](../info_nce/), we know that more negatives raise the ceiling on how much mutual information the loss can capture. SimCLR addressed this by using very large batch sizes (up to 8192), but this requires significant GPU memory. MoCo takes a different approach: ==decouple the dictionary size from the batch size== by maintaining a queue of encoded keys from recent mini-batches.

This raises a consistency problem. If the key encoder changes rapidly during training, the oldest keys in the queue were produced by a very different encoder than the newest ones. The dictionary would contain representations that are not comparable to each other. MoCo solves this with a **momentum-updated encoder** that evolves slowly, keeping the dictionary keys consistent even as they span many training steps.

## Contrastive loss as dictionary lookup

Given a query $q$ and a dictionary of $K+1$ keys $\{k_0, k_1, \ldots, k_K\}$, where $k_+$ denotes the single positive key, MoCo uses the InfoNCE loss:

$$
\mathcal{L}_q = -\log \frac{\exp(q \cdot k_+ / \tau)} {\displaystyle\sum_{i=0}^{K} \exp(q \cdot k_i / \tau)}
$$

where $\tau > 0$ is the temperature. The sum in the denominator runs over the positive and all $K$ negatives, so this is a $(K{+}1)$-way softmax classifier that tries to identify $k_+$.

The query and keys are produced by encoder networks. Given an input image $x$, two random augmentations yield $x^q$ and $x^k$. A **query encoder** $f_q$ produces $q = f_q(x^q)$, and a **key encoder** $f_k$ produces $k_+ = f_k(x^k)$. The negative keys come from other images encoded by $f_k$ in recent mini-batches, stored in a queue.

For this dictionary lookup to produce good representations, the authors argue two properties are needed:

- **Large.** A bigger dictionary samples the underlying visual space more densely, providing harder negatives that force the encoder to learn finer distinctions.
- **Consistent.** The keys should be produced by the same or a similar encoder so that the query can meaningfully compare against them.

## The MoCo mechanism

### Dictionary as a queue

The dictionary is maintained as a **first-in, first-out (FIFO) queue** of size $K$. At each training step, the current mini-batch of encoded keys is enqueued and the oldest mini-batch is dequeued.

This decouples the dictionary size from the batch size: **the dictionary can be much larger than a typical mini-batch**, allowing $K = 65{,}536$ keys even with a batch size of just 256.

The oldest keys are removed because they were encoded by the most outdated version of the key encoder and are therefore the least consistent with the current keys.

### Momentum update

Let $\theta_q$ denote the parameters of the query encoder and $\theta_k$ the parameters of the key encoder.

- Only $\theta_q$ is updated by back-propagation.

The key encoder parameters are updated as an exponential moving average of the query encoder:

$$
\theta_k \leftarrow m\,\theta_k + (1 - m)\,\theta_q
$$

where $m \in [0, 1)$ is the momentum coefficient. With $m = 0.999$ (the value used in the paper), only 0.1% of the query encoder's weights are blended into the key encoder at each step. ==This makes $f_k$ evolve very slowly, so keys encoded many steps apart remain comparable.==

The authors find that momentum is critical. Setting $m = 0.9$ (a faster update) causes a significant accuracy drop, because the key encoder changes too rapidly and the queue keys become inconsistent. At the other extreme, $m = 1$ means the key encoder never updates at all, which also fails because $f_k$ drifts away from the improving $f_q$.

:::figure{#moco_pipeline}
![MoCo pipeline: input x is augmented twice into x^q and x^k. The query encoder f_q produces q; the momentum encoder f_k produces k. The key k is enqueued into a FIFO queue of K keys. The contrastive loss compares q against the positive k+ and all keys in the queue. Gradients flow only through f_q; f_k is updated via momentum.](../../figures/moco_pipeline.svg)

The MoCo pipeline. An image $x$ is augmented into a query view $x^q$ and a key view $x^k$. The query encoder $f_q$ (updated by gradient descent) produces $q$; the momentum encoder $f_k$ (updated via $\theta_k \leftarrow m\theta_k + (1{-}m)\theta_q$) produces $k$. The key is enqueued into a FIFO dictionary of $K$ keys. The InfoNCE loss compares $q$ against $k_+$ and all keys in the queue.
:::

### Putting it together

The full training step is compact. For each mini-batch:

1. Augment each image twice to obtain a query view and a key view.
2. Encode the query view with $f_q$ to get $q$; encode the key view with $f_k$ to get $k$ (with no gradient through $f_k$).
3. Compute the InfoNCE loss using $q$, the positive $k_+$, and the $K$ negatives from the queue.
4. Back-propagate through $f_q$ only and update $\theta_q$.
5. Update the key encoder via momentum: $\theta_k \leftarrow m\,\theta_k + (1 - m)\,\theta_q$.
6. Enqueue the current batch of keys; dequeue the oldest batch.

The gradient never flows through the key encoder or the queue. This is what makes the design efficient: **the dictionary can hold tens of thousands of keys without any additional memory cost for storing their computation graphs**.
