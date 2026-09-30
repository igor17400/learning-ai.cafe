---
title: "TPR and FPR"
subtitle: "Why evaluating ratios strictly within class boundaries defines the entire evaluation mechanism."
---

The previous section ended on a complaint. Accuracy moves when the crowd moves, and once it has moved there is no way to recover how much of the number belonged to the model.

That complaint has a possible solution to it: split the cases by their true label first, then score each group on its own terms.

Ask how often the positive cases are caught. Ask, separately, how often the negative cases are wrongly flagged. Neither question mentions the other class, so neither answer can be shifted by recruiting more of the opposite class.

## Conditioning on the truth

The previous section built a prediction out of a score and a threshold, where

$$
\widehat{Y} = \mathbf{1}[s(X) \geq \tau]
$$

And accuracy asked how often $\widehat{Y}$ and $Y$ agree across everybody. The two questions above ask that same thing twice, each time inside a single class.

$$
\mathrm{TPR}(\tau) \;=\; P\big(s(X) \geq \tau \;\big|\; Y = 1\big)
$$

$$
\mathrm{FPR}(\tau) \;=\; P\big(s(X) \geq \tau \;\big|\; Y = 0\big)
$$

The vertical bar ( $|$ "given" or "conditioned on") is the entire mechanism. It says: throw away every case whose true label is not $1$, and among the ones that remain, count the fraction that clear the threshold. That is the **true positive rate** (TPR), also called sensitivity or **recall**. Do the same on the other side, keeping only the true negatives, and the fraction that wrongly clear the threshold is the **false positive rate** (FPR), which is one minus specificity.

Now look for $\pi$ in either expression. It is not there, and it cannot be, because the prevalence describes how the two classes are mixed and both expressions have already discarded one of them. Doubling the number of negatives changes neither the numerator nor the denominator of $\mathrm{TPR}$, since no negative case was ever counted in it.

### Where the prevalence went

Accuracy did not lose the prevalence. It absorbed it. Split the population by label, and the accuracy of any threshold rule can be written out in full:

$$
\mathrm{Acc}(\tau) \;=\; \underbrace{\pi \cdot \mathrm{TPR}(\tau)}_{\text{positives, caught}} \;+\; \underbrace{(1 - \pi)\big(1 - \mathrm{FPR}(\tau)\big)}_{\text{negatives, left alone}}
$$

Every one of the three settlements is now visible in a single line. $\tau$ appears twice, once in each rate. $\pi$ appears twice, weighting them. And the two weights sum to one, which is the equal-cost assumption written as arithmetic meaning a positive caught and a negative left alone are worth exactly the same.

The identity also explains the grids from the previous section without any new work. A model that answers negative to everything has $\mathrm{TPR} = 0$, because it catches nothing, and $\mathrm{FPR} = 0$, because it flags nothing. Both numbers are the same on both crowds. Only $\pi$ differs, and the identity delivers $0.5 \cdot 0 + 0.5 \cdot 1 = 0.50$ on the balanced grid and $0.01 \cdot 0 + 0.99 \cdot 1 = 0.99$ on the imbalanced one.

==The pair $(\mathrm{TPR}, \mathrm{FPR})$ did not move between those two grids==. Nothing about the model changed, but accuracy moved because $\pi$ moved. The image below shows one example for that:

:::figure{#rates_vs_prevalence}
![Four horizontal bars in two blocks. The upper block, labelled the model's two rates, shows a TPR bar filled to 0.60 and a one-minus-FPR bar filled to 0.95, braced together as unchanged in both cases below. The lower block, labelled the population it meets, shows two mixing bars: one split evenly for a prevalence of 0.50, giving an accuracy of 0.78, and one that is almost entirely negative for a prevalence of 0.01, giving an accuracy of 0.95.](../../figures/rates_vs_prevalence.svg)

**(1)** Assume a model that catches $60\%$ of the positive cases and wrongly flags $5\%$ of the negative ones. **(2)** Now send that same model into two very different populations: one where half the cases are positive, and one where a single case in a hundred is. The two rates in **(1)** do not move between them. The accuracy does, from $0.78$ to $0.95$. Same model both times.
:::

## The denominator decides

None of that was special to these two rates. The invariance came from something structural, and the same structure settles the question for every metric in this tutorial.

A threshold rule sorts every case into one of four boxes. Each case arrives with a true label, $Y$, and leaves with a predicted one, $\widehat{Y}$, and there are only four ways those can combine.

There are two ways to group those four boxes. Group them by **column** and each group is a class: everyone who really was positive, everyone who really was negative. Group them by **row** and each group is a decision: everything the model flagged, everything it let through. A column holds one class. A row holds both.

:::figure{#which_denominator}
![The same two-by-two table of TP, FP, FN and TN drawn twice, with each of the four cells in its own colour: gold for TP, red for FP, green for FN, blue for TN. Every symbol in the formulas beneath is printed in its cell's colour. On the left the columns are outlined, giving TPR as TP over TP plus FN, and FPR as FP over FP plus TN. On the right the rows are outlined instead, giving precision as TP over TP plus FP, and NPV as TN over TN plus FN.](../../figures/which_denominator.svg)

The same four counts, divided two ways. Each cell keeps its colour in the formulas below it, so every denominator can be read straight off the table. On the left the denominators run down the columns, and each one draws from a single class. On the right they run across the rows, and every one of them mixes gold with red, or blue with green.
:::

$\mathrm{TPR}$ divides by a **column** total and so does $\mathrm{FPR}$. That is the whole reason $\pi$ vanished. A column total is a class count, and adding cases to one class leaves the other column exactly as it was.

**Precision** divides by a **row** total, and a row holds cases from both classes. Of everything the model flagged, what share really was positive? Adding negatives to the population adds them to that row, which changes the denominator without the model doing anything differently.

The other row has a name too. Of everything the model let through, the share that really was negative is the **negative predictive value** (NPV), and it is precision's mirror image.

A later section takes precision up properly. For now it is enough to see that both of them sit on the other axis.

## What the pair leaves open

Three things were settled back in the first section: a prevalence, a cost, and a threshold. Conditioning on the truth deals with the prevalence. The other two are exactly where they were.

### Two numbers do not rank models

Consider two models on the same screening problem.

- Model A catches $90\%$ of the positive cases and wrongly flags $30\%$ of the negative ones.
- Model B catches $70\%$ of the positive cases and wrongly flags $5\%$ of the negative ones.

_Which model is better?_

Nothing in the two pairs of numbers answers that. Model A catches more disease and generates a great deal more alarm. On the other hand, model B is quieter and misses more. Neither dominates the other, and no amount of staring and reflecting about $(0.90, 0.30)$ and $(0.70, 0.05)$ will lead us to a winner.

Any rule that does produce a winner should say **how much a missed case costs relative to a false alarm**.

Prevalence was removed from the arithmetic; cost was not, and ideally shouldn't be, because ranking two models is a decision and decisions inevitably need costs.

### The threshold never left

The third settlement is still in place, and the notation has been saying so all along. Both rates were written $\mathrm{TPR}(\tau)$ and $\mathrm{FPR}(\tau)$, with the threshold as an argument, because that is what they are: functions of $\tau$. Change it and both numbers change. Model A and model B might even be the same model, read at two different thresholds.

That is not a flaw to be engineered away. It is the opening for the next idea. If a single threshold gives a single pair, then sweeping the threshold across its whole range gives a curve, and a curve describes the model without committing to any operating point at all.

## Two ways a population can change

A population can change in quite different ways, let me highlight two:

1. **The mix changes.** More negatives arrive, and they look like the negatives already there. Their scores fall in the same places. $\pi$ moves and nothing else does.
2. **The cases themselves change.** More negatives arrive, and they are a different sort of person. Their scores sit somewhere else entirely. $\pi$ moves, and so does what a negative case looks like to the model.

Everything on this page so far has been about the first kind. There, the argument holds exactly: $\mathrm{FPR}$ is a property of the negative score distribution, and pouring in more of the same negatives leaves that distribution where it was.

The second kind breaks it because suppose in the clinical example that the extra healthy patients are recruited from a workplace screening programme rather than from the clinic waiting room. They are younger and have no symptoms, so the model scores them lower than the negatives it was tested on. Fewer of them cross the threshold, and $\mathrm{FPR}$ falls. Nothing about the model changed but the negative class did.

==The invariance is a claim about the mix, not about the population==. It says the rates survive a change in "how many". It does not say they survive a change in "who" ([Richardson et al., 2024](../references/)).
