---
title: "Balanced Accuracy"
subtitle: "Accuracy, evaluated at a prevalence of one half."
---

The previous section left accuracy with a number that reports partly on the model and partly on the crowd, and no way to separate the two. Balanced accuracy is the standard response, and it is a real improvement.

It is worth being precise about what it improves, because it is not what the name suggests. The prevalence commitment is not removed. It is replaced, and a cost commitment arrives in its place.

## One substitution

Balanced accuracy is the mean of the two rates from [section 02](../tpr_fpr/).

$$
\mathrm{BA}(\tau) \;=\; \tfrac{1}{2}\Big(\mathrm{TPR}(\tau) \;+\; \big(1 - \mathrm{FPR}(\tau)\big)\Big)
$$

where $\mathrm{TPR}(\tau)$ is the share of positive cases the model catches at threshold $\tau$, and $1 - \mathrm{FPR}(\tau)$ is the share of negative cases it correctly leaves alone. Sensitivity and specificity, averaged.

Set that beside the identity section 02 established, which splits accuracy into the same two rates.

$$
\mathrm{Acc}(\tau) \;=\; \pi \cdot \mathrm{TPR}(\tau) \;+\; (1 - \pi)\big(1 - \mathrm{FPR}(\tau)\big)
$$

The two expressions have the same shape and differ only in their weights. Accuracy weights each rate by the share of the population its class occupies. Balanced accuracy weights them equally, which is to say it sets $\pi = \tfrac{1}{2}$.

$$
\mathrm{BA}(\tau) \;=\; \mathrm{Acc}(\tau)\big|_{\pi = 1/2}
$$

==Balanced accuracy is not a correction applied to accuracy. It is accuracy, read at a prevalence of one half.== Whatever population the test set was drawn from, the number reported is the accuracy this model would have achieved on a population holding equal numbers of the two classes.

:::figure{#accuracy_vs_balanced}
![A plot of metric value against prevalence for one fixed model. Accuracy is a straight line sloping down from 0.90 at prevalence zero to 0.60 at prevalence one. Balanced accuracy is a horizontal line at 0.75. The two cross exactly at prevalence one half, marked with a dot. At prevalence 0.01 on the far left, two dots on a dashed vertical show the same model scoring 0.897 by accuracy and 0.75 by balanced accuracy.](../../figures/accuracy_vs_balanced.svg)

One model throughout, with $\mathrm{TPR} = 0.60$ and $1 - \mathrm{FPR} = 0.90$. Its two rates never move, so $\mathrm{BA}$ is flat. $\mathrm{Acc}$ slides from $0.90$ to $0.60$ because it reweights those same two rates by whoever is in the room. The two agree at exactly one point, and that point is $\pi = \tfrac{1}{2}$.
:::

## What it genuinely fixes

Section 03 left accuracy with a scale that collapses. At $\pi = 0.01$ a model that answers negative to everything already scores $0.99$, and every model worth discussing is packed into the remaining hundredth.

That failure is gone entirely. Take any model that ignores its input. Answer negative always, and $\mathrm{TPR} = 0$ with $\mathrm{FPR} = 0$, so $\mathrm{BA} = \tfrac{1}{2}(0 + 1) = 0.5$. Answer positive always, and $\mathrm{TPR} = 1$ with $\mathrm{FPR} = 1$, so $\mathrm{BA} = \tfrac{1}{2}(1 + 0) = 0.5$. Both score exactly one half, at every prevalence, with no exceptions. _Chance is pinned to $0.5$ and stays there_, so the entire range of the metric is available to describe the model.

The invariance is real too, for the reason section 02 gave. Balanced accuracy is assembled from $\mathrm{TPR}$ and $\mathrm{FPR}$, and both divide by a column total, so recruiting more negatives changes neither. On the first of the three failures, dependence, balanced accuracy is a strict improvement over accuracy.

The same condition section 02 attached still applies. The claim concerns a change in how many, not a change in who. Negatives drawn from a different population move $\mathrm{FPR}$, and balanced accuracy moves with it.

## The cost that arrives instead

Section 03 made accuracy legible by putting its payoff table on the page. Balanced accuracy has one too.

|                   |           $Y = 1$ |                 $Y = 0$ |
| ----------------- | ----------------: | ----------------------: |
| $\widehat{Y} = 1$ | $\dfrac{1}{2\pi}$ |                     $0$ |
| $\widehat{Y} = 0$ |               $0$ | $\dfrac{1}{2(1 - \pi)}$ |

where $\pi$ is the prevalence of the evaluation set ([Flores et al., 2025](../references/)). The entries are no longer $1$ and $0$. Each correct call is now worth the reciprocal of twice its own class's share, so the rarer a class is, the more a correct call on it pays.

The entries are payoffs in an arbitrary unit, and only their ratio carries any meaning. At $\pi = 0.01$ the table assigns $1 / (2 \times 0.01) = 50$ payoff units to a caught positive and $1 / (2 \times 0.99) \approx 0.505$ to a negative correctly left alone. Dividing one by the other cancels the unit, and in general

$$
\frac{1 / 2\pi}{1 / 2(1 - \pi)} \;=\; \frac{1 - \pi}{\pi}
$$

which is $99$ for $\pi = 0.01$. ==Balanced accuracy declares one caught positive worth ninety-nine negatives correctly left alone==, or equivalently one missed positive worth ninety-nine false alarms. For some problems that is a defensible exchange rate. But it is not neutral, and most importantly it wasn't actually chosen. Meaning _it was fixed by the base rate of whichever evaluation set happened to be used._

The same number can be reached from either direction, and Flores et al. write it both ways. A small example shows why. Take $1000$ cases at $\pi = 0.01$, so $10$ positives and $990$ negatives. At some threshold the model catches $6$ of the positives and correctly leaves $891$ of the negatives alone, which makes ordinary accuracy $(6 + 891) / 1000 = 0.897$.

Both routes reuse the same five counts.

$$
n = 1000, \quad P = \pi n = 10, \quad N = (1 - \pi) n = 990, \quad \mathrm{TP} = 6, \quad \mathrm{TN} = 891
$$

$n$ is the number of cases, $P$ and $N$ the two class counts. So $\mathrm{TPR} = \mathrm{TP} / P = 0.6$ and $1 - \mathrm{FPR} = \mathrm{TN} / N = 0.9$.

**Route one: rebalance the data, keep the table.** Copy each positive $(1 - \pi) / \pi = 99$ times, then count correct answers as usual.

$$
\frac{\underbrace{6}_{\mathrm{TP}} \times \underbrace{99}_{(1 - \pi)/\pi} \;+\; \underbrace{891}_{\mathrm{TN}}}{\underbrace{1980}_{2N}} \;=\; \frac{1485}{1980} \;=\; 0.75
$$

**Route two: keep the data, change the table.** Pay each case the entry the table above gives it.

$$
\frac{\underbrace{\tfrac{1}{2\pi}}_{50} \times \underbrace{6}_{\mathrm{TP}} \;+\; \underbrace{\tfrac{1}{2(1 - \pi)}}_{1/1.98} \times \underbrace{891}_{\mathrm{TN}}}{\underbrace{1000}_{n}} \;=\; \frac{300 + 450}{1000} \;=\; 0.75
$$

The same $0.75$, twice, and neither route is the $0.897$ that accuracy reported. The first describes a population that does not exist. The second describes the implicit cost structure. _They are the same arithmetic._

:::figure{#balanced_two_routes}
![A flow diagram. One box at the top describes the evaluation set: 1000 cases at prevalence 0.01, with 6 of 10 positives caught and 891 of 990 negatives left alone. Two arrows lead down to two cards. The left card rebalances the data by copying each positive 99 times and computes ordinary accuracy. The right card leaves the data alone and applies the weighted payoff table. Both cards feed arrows into a single boxed result reading BA equals 0.75. A footnote records that ordinary accuracy on the untouched set is 0.897.](../../figures/balanced_two_routes.svg)

The two routes drawn side by side. Reweight the data and keep the plain payoff table, or keep the data and reweight the table. The arithmetic meets at $0.75$, and neither branch passes through the $0.897$ that ordinary accuracy reports on the same $1000$ cases.
:::

### Where the threshold lands

Section 03 showed that for a [calibrated score](../accuracy/#the-threshold-the-table-implies), accuracy is maximised by predicting positive when $P(Y = 1 \mid x) > \tfrac{1}{2}$. That one half came out of a symmetric payoff table where both correct answers were worth the same, so the tie sat in the middle. The table above is not symmetric, so the tie sits somewhere else. Take a single case whose posterior is $p$, and price the two things we could do with it. Both prices come straight from the table.

- **Flag it.** With probability $p$ the case really is positive and the table pays $\tfrac{1}{2\pi}$. With probability $1 - p$ it is negative and the table pays $0$.

$$
\text{Expected payoff} = p \cdot \tfrac{1}{2\pi}
$$

- **Leave it.** With probability $1 - p$ the case really is negative and the table pays $\tfrac{1}{2(1 - \pi)}$. With probability $p$ it is positive and the table pays $0$.

$$
\text{Expected payoff} = (1 - p) \cdot \tfrac{1}{2(1 - \pi)}
$$

Flagging is worth it when the first beats the second, which we can write $\frac{p}{2\pi} \;>\; \frac{1 - p}{2(1 - \pi)}$ and thus yields:

$$
\frac{p}{2\pi} \;>\; \frac{1 - p}{2(1 - \pi)} \quad \Longleftrightarrow \quad p\,(1 - \pi) \;>\; \pi\,(1 - p) \quad \Longleftrightarrow \quad p - p\pi \;>\; \pi - p\pi
$$

The $p\pi$ term sits on both sides and cancels, which is why the answer comes out as clean as it does.

$$
p \;>\; \pi
$$

Let's read the statement above step by step. When $p = \pi$ the two options are worth exactly the same and the metric is indifferent. Above it, flagging pays. Below it, leaving pays. **The threshold that maximises balanced accuracy is therefore the prevalence itself**:

$$
\tau^{\star} = \pi
$$

One caveat on that last step. The rule was derived for the posterior $p$, and it becomes a rule about the _score_ only when the score is the posterior, which is to say when the model is calibrated. For an uncalibrated score the optimum still exists, but it has to be found by sweeping rather than read off.

**Balanced accuracy is maximised by thresholding the posterior at the prevalence itself**. At $\pi = 0.01$ that instructs us to act on every case whose probability exceeds one percent. For a cancer screening programme this may well be correct. For a spam filter it would quarantine most of the inbox. Both rules come out of the same metric, and the metric chose between them by reading the base rate off the data rather than by asking anybody what a mistake was worth.

## A different assumption, not the absence of one

What balanced accuracy estimates is perfectly well defined: the accuracy this model would achieve on a population **containing equal numbers of the two classes**. The question is whether that population is one anybody will meet.

Two settings where it is the right quantity:

- comparing one model across several datasets whose class mixes differ for reasons of collection rather than deployment,
- benchmark reporting where no deployment population is being claimed at all.

And one where it is not. Wherever somebody will act on the model at a known base rate, one half is simply the wrong prevalence, and the observed $\pi$ was the better guess of the two. Replacing a measured quantity with a fixed one is only a repair if the fixed one is closer to the truth.

Balanced accuracy stays inside the columns. It is built from $\mathrm{TPR}$ and $\mathrm{FPR}$, and inherits both their invariance and their silence about cost. The next section takes up the first metric that leaves the columns.
