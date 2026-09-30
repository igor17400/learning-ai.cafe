---
title: "Committee or Competition"
subtitle: "Two ways to combine several networks, and only one of them specialises."
---

[The previous topic](../dense_layer/) explained one architecture of interest which is a layer that holds a great deal of parameters and uses only a few of them on any given token. That wish is about compute, and it is how mixtures of experts are usually motivated today.

## The interference problem

Nowlan and Hinton open with the difficulty in one sentence: if backpropagation is used to train a single network to perform different subtasks on different occasions, there will generally be strong interference effects which lead to slow learning and poor generalisation. Every weight is pulled by every case. Cases that need different behaviour pull in different directions, and what settles is a compromise that suits neither.

The proposal introduces a **collection of distinct sub-networks, defined as the experts $E$**, alongside a dedicated routing module known as the **gating network $g$**. The sole responsibility of the gate is to analyze the current input and assign it to the appropriate expert.

The term "gating" here can be misleading, as it suggests a simple on/off switch when it is actually nothing of the sort. **The gate is an ordinary neural network**, usually a small multi-layer perceptron that takes the input, passes it through zero or more hidden layers, and ends in an output layer with **exactly one unit per expert**. In the original experiments, the gate sometimes had five or ten hidden units, and sometimes none at all, making it a single linear layer and nothing more.

The values sitting at those $N$ output units, before any normalisation, are the **scores $\mathbf{s} = (s_1, \dots, s_N)$**. There is one score per expert, and $s_i$ is the network's raw, unbounded opinion about how suitable expert $i$ is for this input. Being unbounded, the scores are not yet a routing decision given that they can be negative, they can be large, and they do not sum to anything in particular.

Concretely, in the simplest case where the gate has no hidden layers, producing the scores is a single matrix multiplication:

$$
\mathbf{s} = W_g\,\mathbf{x} + \mathbf{b}
$$

- $\mathbf{x} \in \mathbb{R}^{d}$ is the **input** the gate is shown, of size $d$.
- $W_g \in \mathbb{R}^{N \times d}$ is the **gate's weight matrix**, with one row per expert. Row $i$ is a direction in input space, and $s_i$ is how far the input points along it.
- $\mathbf{b} \in \mathbb{R}^{N}$ is the **bias**, one per expert.

:::aside[A gate with several hidden layers]
Number the layers $\ell = 1, \dots, L$ and let $\mathbf{h}^{(0)} = \mathbf{x}$. Each hidden layer is

$$
\mathbf{h}^{(\ell)} = \phi\!\left(W^{(\ell)}\mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)}\right), \qquad \ell = 1, \dots, L-1
$$

and the output layer, which produces the scores, is

$$
\mathbf{s} = W^{(L)}\mathbf{h}^{(L-1)} + \mathbf{b}^{(L)}
$$

with $W^{(\ell)} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$, where $n_\ell$ is the width of layer $\ell$. Only two of those widths are forced: $n_0 = d$, the size of the input, and **$n_L = N$**, one output unit per expert. Everything in between is a free choice. The single-layer case above is $L = 1$.

The layer index is written in parentheses, $W^{(\ell)}$, because a bare superscript already means the training case, as in $g_i^{c}$. And the hidden nonlinearity is written $\phi$ rather than $\sigma$, because $\sigma$ is the width of the Gaussians further down this page.
:::

With hidden layers this is the last of several such steps, and nothing about the argument changes. What matters is that **$W_g$ and $\mathbf{b}$ are learned**, by the same gradient descent that trains the experts. Nobody writes down a routing rule. The gate discovers one, and later on this page we derive exactly what the gradient arriving at $s_i$ turns out to be.

:::figure{#gating_network}
![The case's input enters a small multi-layer perceptron whose output layer has one unit per expert. The four pre-softmax values 1.2, minus 0.4, 2.7 and 0.1 are the scores. A softmax turns them into gate values of 0.17, 0.03, 0.74 and 0.06, which weight the four experts, and the arrow into the third expert is much thicker than the others.](../../figures/gating_network.svg)

The gate is a network like any other. What makes it a **gate** is the shape of its output layer, one unit per expert.
:::

To turn the scores into a routing decision we need them non-negative and summing to one, so they can be read as probabilities. That is exactly what the **softmax function** does:

$$
g_i = \frac{e^{s_i}}{\sum_j e^{s_j}}
$$

The exponential makes every value positive and the denominator makes them sum to one, so the largest score becomes the largest share without any of them ever reaching zero. From the first paper onwards the $g_i$ are read as **the probability of selecting expert $i$**.

One piece of notation before we go further. A **case** is one training example which contains an input $\mathbf{x}$ together with a target $\mathbf{y}$ we want the network to predict. When we need to tell one case from another we write it as a superscript, so $\mathbf{x}^{c}$ and $\mathbf{y}^{c}$ are the input and the target of case $c$, and $g_i^{c}$ is what the gate said about expert $i$ on that case. A sum written $\sum_c$ runs over the training set and a sum written $\sum_i$ runs over the experts. The figure below illustrates this process.

:::figure{#gate_scores_cases}
![At the top, one input runs through the gate to three scores of plus 0.4, plus 2.1 and minus 0.6, and a softmax turns them into a bar of height one split into 0.15, 0.80 and 0.05. Below, a table with one row per case and one column per expert holds the gate's outputs, with a brace across the first row showing that it sums to one and a brace under the second column showing that it sums to 1.44 down the training set.](../../figures/gate_scores_cases.svg)

The two indices do different work. Across a row the gate splits one case between the experts, so the row sums to one. Down a column it accumulates how much of the training set an expert was handed, and that quantity is the one every load argument later in the series is about.
:::

Here a case is a whole training example, because that is what these networks were built to handle: one input, one target, one decision by the gate. From [Compute You Can Skip](../conditional_computation/) onwards a case is a single **token** inside a sequence, and the gate makes its decision several times per example rather than once.

## Two ways to combine them

Defining a multi-network architecture with a gate does not yet tell us how this mechanism should learn. Let's go through this learning process.

### The committee

The standard ensemble approach is to blend the outputs via the gate and compute the error on the final prediction. Let $\mathbf{o}_i$ represent the individual output vector produced by the $i$-th expert. The overall network prediction $\hat{\mathbf{y}}$ is given as a weighted sum of these expert outputs proportioned by the gate's activation $g_i$. That is,

$$
\hat{\mathbf{y}} = \sum_i g_i\, \mathbf{o}_i, \qquad \text{minimising } \big\|\hat{\mathbf{y}} - \mathbf{y}\big\|^{2}
$$

Look at what it does to an individual expert. The error is computed **after** the blend, so an expert receives gradient signals based on the committee's collective performance rather than their own. The expert is told whether the committee was right, and it is nudged by its share of that. If expert 2 is already good on this case and expert 1 is badly wrong, the loss still pulls expert 1 toward the same target, because the global loss $\|\hat{\mathbf{y}} - \mathbf{y}\|^{2}$ cannot disentangle individual contributions, it merely treats Expert 1's output as a deficit in the aggregate prediction. Thus, every expert is trained to reduce the residual of the others. ==Under a squared error on the blend, the best thing any expert can do is learn the average of everything==, which is the interference we were trying to escape.

### The competition

The alternative is to measure each expert on its own, and blend the measurements rather than the outputs. Treat each expert as predicting the mean of a Gaussian, treat the gate as supplying the **mixing proportions**, and take the negative log probability of the target under that mixture:

$$
E^{c} = -\log \sum_i g_i^{c}\, e^{-\|\mathbf{d}^{c} - \mathbf{o}_i^{c}\|^{2} / 2\sigma^{2}}
$$

- $E^{c}$ is the error on training case $c$.
- $g_i^{c}$ is the **gate's output** for expert $i$ on the case $c$.
- $\mathbf{d}^{c}$ is the **desired output** and $\mathbf{o}_i^{c}$ the **output of expert $i$**.
- $\sigma$ is a constant, the width of each Gaussian.

The outputs are vectors, as they are in the original paper. Since $\sigma$ is a single shared width, each expert defines a spherical Gaussian centred on its own prediction, and $\|\mathbf{d}^{c} - \mathbf{o}_i^{c}\|$ is the distance from the target to that centre. Everything below holds unchanged for a one-dimensional output, where the norm is read as an absolute value.

The two expressions might look similar, but their behavior is fundamentally changed by swapping the order of operations. Instead of blending the predictions, this equation treats each expert's output as the **center of its own Gaussian distribution**. The inner term $e^{-\|\mathbf{d}^{c} - \mathbf{o}_i^{c}\|^{2} / 2\sigma^{2}}$ directly measures how well Expert $i$ alone explains the target. The gate $g_i^{c}$ then scales this by the expert's assigned routing probability.

By summing these independent probabilities before applying the logarithm, the loss evaluates the mixture as a whole without forcing the experts' outputs to average out. If even a single expert perfectly hits the target, the sum inside the logarithm becomes large, the overall loss drops, and the other experts are left alone rather than being dragged in to compensate. The figure below illustrates the process for each combination strategy.

:::figure{#blend_vs_compete}
![Two panels. On the left, three experts feed a weighted sum, the sum is compared with the target, and gradient arrows of equal thickness return to all three experts. On the right, each expert is compared with the target on its own, the three likelihoods are combined inside a logarithm, and the gradient arrow returning to the middle expert is far thicker than the other two.](../../figures/blend_vs_compete.svg)

Blending before the measurement makes the experts cooperate. Measuring before the blend makes them compete, and the arrows coming back are the whole difference.
:::

It's important to highlight the gradient effect of the architectures above. **The committee** takes a weighted sum of outputs and then measures the error. **The competition** measures the error of each expert separately, takes a weighted sum of those probabilities, and then applies the logarithm. _But why does this matter?_ Because the summation happens inside the logarithm which allows the gradients to decouple. If one expert is highly accurate, it dominates the sum, minimizing the loss for that specific case. The remaining experts aren't penalized or forced to adjust their weights to fix a residual gap. They are simply ignored for that case. This is somewhat a "true competition" in the sense that experts are rewarded for individual specialization, not for collective compromise, allowing each one of them to specialize.

## Prior, posterior, and the gradient between them

The competition reading gives us a second quantity that the committee formulation does not have, before the target arrives, the gate's output $g_i$ is a **prior**, meaning it measures how likely expert $i$ is to be the right one, judged from the input alone. Once the target has arrived we can ask, how likely expert $i$ is to have been the one that produced it. That is a **posterior**, if we recall Bayes' theorem:

$$
P(y \mid x) = \frac{P(x \mid y)\,P(y)} {\sum_{y' \in \mathcal{Y}} P(x \mid y')\,P(y')}
$$

We can map this directly to our experts. The "prior" $P(y)$ becomes our gate's prediction $g_i$. The "likelihood" $P(x \mid y)$ becomes the probability that expert $i$ produces the target, $P(\mathbf{y} \mid \mathbf{x}, i)$. The denominator normalises this across _all_ $j$ experts. And this allows us to write the posterior probability $h_i = P(i \mid \mathbf{x}, \mathbf{y})$ for each expert:

$$
h_i = P(i \mid \mathbf{x}, \mathbf{y}) = \frac{g_i\, P(\mathbf{y} \mid \mathbf{x}, i)} {\sum_j g_j\, P(\mathbf{y} \mid \mathbf{x}, j)}
$$

$h_i$ is the _responsibility_ of expert $i$ for this case. It is what the gate should have said, updated by the evidence of the actual target.

### Examples for scalar and vector outputs

**The one-dimensional case.** Consider a simplified one-dimensional scenario where the predictions and the target are just points on a line as the figure below shows. Before the target $y = 7.0$ arrives, the gate expresses a mild prior preference, distributing weights of $0.30, 0.45$, and $0.25$ across three experts. Once the target is revealed, we measure the gap between $y$ and each expert's prediction $\mu_i$. Because Expert 2 ($\mu_2 = 6.8$) lands extremely close to the target, its likelihood increases. The posterior $h_i = 0.98$ incorporates this evidence, sharpening the gate's guess that this specific case should be of responsibility of Expert 2 due to how close to the target this expert prediction was. The crucial takeaway is the difference between these two states,

$$
\text{posterior} - \text{prior} = h_i - g_i
$$

which forms the exact gradient. The gate is heavily rewarded (+0.53) for the expert that was right, and penalized (-0.30, -0.23) for the two that were wrong.

:::figure{#responsibility_split}
![Three expert predictions on a value axis at 2.1, 6.8 and 9.5, with the target at 7.0. Below, two rows of bars: the gate's prior at 0.30, 0.45 and 0.25, and the responsibilities at 0.00, 0.98 and 0.02. The difference between the rows is given as the gradient on the gate.](../../figures/responsibility_split.svg)

One case, drawn in one dimension so the predictions and the target are points on a line. The gate is trained on the gap between the prior and the posterior.
:::

**The vector case.** When $\mathbf{y}$ is a vector, the underlying arithmetic remains completely unchanged; only the geometric interpretation of the likelihood shifts. The simple gap on a number line becomes a spatial distance. To understand how this works, we can break down the variables:

- **$\mathbb{R}^{d_y}$**: The multi-dimensional space where the target and predictions exist, with $d_y$ representing the number of dimensions.
- **$\Sigma = I$**: The covariance matrix ($\Sigma$) is set to the identity matrix ($I$). This is a way of stating that we treat every dimension equally and assume no correlation between them.
- **$P(\mathbf{y} \mid \mathbf{x}, i) = e^{-\|\mathbf{y} - \boldsymbol{\mu}_i\|^2 / 2}$**: The likelihood function. Instead of measuring a simple scalar gap, it evaluates the probability based on spatial distance.
- **$\boldsymbol{\mu}_i$**: The multi-dimensional prediction vector generated by expert $i$.
- **$\|\cdot\|$**: The Euclidean norm, representing the standard straight-line distance between the expert's prediction and the actual target.

Looking at the figure, the dashed lines represent the Euclidean distance between the expert predictions ($\boldsymbol{\mu}_i$) and the target vector ($\mathbf{y}$) in 2D space. Because Expert 2 sits much closer to the target (0.72) than the others, its likelihood spikes. This spatial proximity drives the posterior $h_i$ up to 0.98, overriding the gate's uncertain prior. The resulting gradient, $h_i - g_i$, actively rewards the closest expert (+0.53) and penalizes the rest.

:::figure{#responsibility_vector}
![The same case drawn on a plane rather than a line. Three expert predictions sit at the coordinates 2.1 and 2.5, 6.6 and 4.6, and 9.5 and 3.0, with the target marked at 7.0 and 4.0 and dashed segments giving the distances 5.12, 0.72 and 2.69. Below, the same two rows of bars: the prior at 0.30, 0.45 and 0.25, and the responsibilities at 0.00, 0.98 and 0.02.](../../figures/responsibility_vector.svg)

Widening the target changes what counts as close, and nothing else. The prior, the posterior and the gap between them are the same quantities, arrived at by measuring the distance in every component at once.
:::

### Log likelihood gradient

Before differentiating, we must define the likelihood of the entire training set under our mixture model, denoted as $L$. Because the training cases are independent, the total likelihood is simply the product of the marginal probabilities across all cases and experts. Taking the log of this product retrieves a sum over cases $c$ and experts $i$:

$$
\log L = \sum_c \log \sum_i g_i^{c}\, P\!\left(\mathbf{y}^{c} \mid \mathbf{x}^{c}, i\right)
$$

This is exactly the objective function we defined earlier, just with the sign flipped and summed over the dataset. Maximizing $\log L$ is identical to minimizing $\sum_c E^c$. When we differentiate this objective with respect to the gate's pre-softmax activations (let's call them $s_i$), the resulting gradient arriving at the $i$-th output unit is:

$$
\frac{\partial \log L}{\partial s_i} = \sum_c \left( h_i^{c} - g_i^{c} \right)
$$

:::aside[Where $h_i - g_i$ comes from, step by step]
Let's work with a single case and drop the superscript $c$, since the sum over cases just carries through at the end. This means, just to be clear, that we drop $\sum_c$ and look at a single term of it. We also rename the summation index from $i$ to $j$, so that $i$ is free to mean the **one expert we are differentiating with respect to**:

$$
\log \sum_j g_j\, P(\mathbf{y} \mid \mathbf{x}, j)
$$

Write $P_j$ for the likelihood of the target under expert $j$, and $M$ for the marginal that sits inside the logarithm:

$$
P_j = P(\mathbf{y} \mid \mathbf{x}, j), \qquad M = \sum_j g_j P_j, \qquad \log L = \log M
$$

**Step 1. Differentiate the logarithm.** To find the gradient with respect to $s_i$ (the raw pre-activation score computed by the gate for expert $i$) we apply the chain rule. Because the expert likelihoods $P_j$ are completely independent of the gating network, they act as constants here. The raw score $s_i$ reaches the marginal $M$ only through the gates, and it reaches _every_ one of them, because the softmax denominator is shared. That is why the sum below runs over all $j$ and not just over $j = i$:

$$
\frac{\partial \log M}{\partial s_i} = \frac{1}{M} \frac{\partial M}{\partial s_i} = \frac{1}{M} \sum_j P_j \frac{\partial g_j}{\partial s_i}
$$

**Step 2. Differentiate the softmax.** Let the denominator be $Z = \sum_m e^{s_m}$, meaning the gate's output is $g_j = e^{s_j} / Z$. To find the derivative, we apply the standard quotient rule:

$$
\frac{\partial}{\partial s_i}\!\left(\frac{u}{v}\right) = \frac{u'\,v - u\,v'}{v^{2}}, \qquad u = e^{s_j}, \quad v = Z
$$

Take the denominator first, because it behaves the same way whatever $j$ is. Differentiating the sum with respect to $s_i$, every term where $m \neq i$ is a constant and drops to zero:

$$
\frac{\partial Z}{\partial s_i} = \frac{\partial}{\partial s_i} \left( e^{s_1} + \dots + e^{s_{i-1}} + e^{s_i} + e^{s_{i+1}} + \dots \right)
$$

$$
\frac{\partial Z}{\partial s_i} = 0 + \dots + 0 + e^{s_i} + 0 + \dots = e^{s_i}
$$

so $v' = e^{s_i}$, whatever $j$ happens to be.

The numerator is where we have the two cases. $u = e^{s_j}$ depends on $s_i$ only when $j$ happens to be $i$, so

$$
u' = \frac{\partial e^{s_j}}{\partial s_i} = \begin{cases} e^{s_i} & \text{if } j = i \\ 0 & \text{otherwise} \end{cases}
$$

Feeding those two into the quotient rule gives the two cases:

**First**: for the target unit ($j = i$), we have $u' = e^{s_i}$, meaning both the top and bottom of the fraction depend on $s_i$:

$$
\frac{\partial g_i}{\partial s_i} = \frac{u'\,v - u\,v'}{v^{2}} = \frac{e^{s_i}Z - e^{s_i}e^{s_i}} {Z^2} = \frac{e^{s_i}}{Z} - \left( \frac{e^{s_i}}{Z} \right)^2
$$

$$
\frac{\partial g_i}{\partial s_i} = g_i - g_i^{2} = g_i(1 - g_i)
$$

**Second**: for any other unit ($j \neq i$) we have $u' = e^{s_j}$ is independent of $s_i$. Its derivative is $0$, meaning the first term of the quotient rule vanishes:

$$
\frac{\partial g_j}{\partial s_i} = \frac{u'\,v - u\,v'}{v^{2}} = \frac{0 \cdot Z - e^{s_j}e^{s_i}} {Z^2}
$$

$$
\frac{\partial g_j}{\partial s_i} = -\,\frac{e^{s_j}}{Z} \cdot \frac{e^{s_i}}{Z} = -\,g_j g_i
$$

Both are the same expression, written with the Kronecker delta $\delta_{ij}$, which is $1$ when $i = j$ and $0$ otherwise:

$$
\frac{\partial g_j}{\partial s_i} = g_j\left(\delta_{ij} - g_i\right)
$$

**Step 3. Substitute and split the sum.** Putting that back into step 1, the delta picks out the single term $j = i$ and the $-g_i$ multiplies the whole sum:

$$
\frac{\partial \log M}{\partial s_i} = \frac{1}{M} \sum_j P_j\, g_j \left(\delta_{ij} - g_i\right) = \frac{1}{M}\left( g_i P_i - g_i \sum_j g_j P_j \right)
$$

**Step 4. Recognise the marginal.** The sum that is left is $M$ itself $\left( M = \sum_j g_j P_j \right)$, so the second term collapses:

$$
\frac{\partial \log M}{\partial s_i} = \frac{1}{M}\left( g_i P_i - g_i M \right) = \frac{g_i P_i}{M} - \frac{g_i M}{M}
$$

and in the second term the $M$ on top and the $M$ underneath are the same number, so they cancel and leave the gate on its own:

$$
\frac{\partial \log M}{\partial s_i} = \frac{g_i P_i}{M} - g_i
$$

**Step 5. Recognise Bayes.** The first term is exactly the responsibility defined earlier, a prior times a likelihood over the marginal:

$$
\frac{g_i P_i}{M} = \frac{g_i\, P(\mathbf{y} \mid \mathbf{x}, i)} {\sum_j g_j\, P(\mathbf{y} \mid \mathbf{x}, j)} = h_i
$$

Substituting that back into step 4 closes the chain for a single case:

$$
\frac{\partial \log M}{\partial s_i} = h_i - g_i
$$

Restoring the case superscript and summing over the training set turns $\log M$ back into $\log L$ and gives the result stated above, $\;\partial \log L / \partial s_i = \sum_c \left(h_i^{c} - g_i^{c}\right)$.

It is worth seeing which step did the work. The $1/M$ in step 1 comes from differentiating the logarithm, and it is precisely the normalising constant that turns the raw quantity $g_i P_i$ into a posterior in step 5. Had the logarithm been inside the sum rather than outside it, there would be no shared $M$, no normalisation, and no posterior. The competition and the clean learning rule come from the same placement of the logarithm.
:::

With the derivation complete, we can see that this is a clean learning rule. The gate trains to predict its own posterior, settling only when the prior and the posterior perfectly agree. The gradient arriving at the individual experts reveals the other half of the story. To understand this update, we differentiate the likelihood with respect to the expert's output prediction, denoted as $\boldsymbol{\mu}_i$. Factoring in $\sigma^2$ as the constant variance of our Gaussian distribution, the gradient for the expert is:

$$
\frac{\partial \log L}{\partial \boldsymbol{\mu}_i} = \sum_c \frac{h_i^{c}}{\sigma^{2}}\left(\mathbf{y}^{c} - \boldsymbol{\mu}_i^{c}\right)
$$

The familiar error term $\mathbf{y} - \boldsymbol{\mu}_i$ is there, scaled by $h_i$. An expert is corrected in proportion to how responsible it was. Typically only one expert has a large posterior on any case, so in practice ==only one expert learns each training case==, and different experts end up learning different cases. The specialisation is not arranged. It falls out of the derivative.

## Sparsity was never imposed

If we follow the learning dynamics to their fixed point, a clear pattern emerges. An expert that wins a case is trained on it and improves, raising its likelihood for that region of the input space. This increased likelihood raises its responsibility, which correspondingly raises the gate's prior. Consequently, after training, the gating network nearly always assigns a mixing proportion of one to a single expert per case.

Note that there's no enforcing of a sparse gate. There is no top-$k$ operation or penalty for spreading the probability mass. The gate therefore naturally sharpens because doing so maximizes the likelihood.

While this allows a partitioned model to significantly outperform a single dense network on heterogeneous tasks, it **completely fails to reduce computational cost**. To compute the objective $E^c$, the forward pass requires the prediction $\mathbf{o}^c_i$ for every expert $i$, as every expert is evaluated inside the sum. The posterior $h_i$ requires them all over again.

The gate ends up nearly one-hot, but we still evaluate every expert anyway. Ultimately, this makes the mixture more expensive than the standard dense layer we started with. The specialization comes for free, but the computational savings do not. Translating this specialization into actual efficiency is the challenge of the next section.
