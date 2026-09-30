---
title: "Accuracy"
subtitle: "An exact answer to a question almost nobody meant to ask."
---

Accuracy has already been through this tutorial once. It carried the three settlements in the opening section, and it was exposed there as a cautionary tale. A number that hides a threshold, a cost and a prevalence, and that calls a model excellent for answering _negative_ to everything.

This section gives it a fairer hearing, because that reputation is only half deserved. Accuracy is not broken and it is not biased. It is an exact statement about a specific decision, made under assumptions that are occasionally true.

## Accuracy is a payoff

Accuracy is usually introduced as counting. How many did the model get right, out of how many were shown to it. Counting is what the arithmetic does, but it is not what the number means.

Let's visualize it in another way. Attach a value to each of the four outcomes before any counting starts. A correct call is worth $1$ and a mistake is worth $0$.

|                   | $Y = 1$ | $Y = 0$ |
| ----------------- | ------: | ------: |
| $\widehat{Y} = 1$ |     $1$ |     $0$ |
| $\widehat{Y} = 0$ |     $0$ |     $1$ |

Now run the model over the population, look up each case in the table, and take the average of what comes back.

$$
\mathrm{Acc}(\tau) \;=\; \mathbb{E}\big[\,U(\widehat{Y}, Y)\,\big]
$$

where $U$ is the table above, read as a function: it takes the prediction and the truth, and returns what that pairing is worth. The expectation $\mathbb{E}[\cdot]$ is taken over the population, so it averages $U$ across every case the model will meet.

That average is accuracy. Not a count of correct answers, but ==the expected payoff of running this model on this population==, under one particular schedule of payoffs. The schedule above has a name, it is the **zero-one loss**, so called because every mistake costs exactly one and every success costs nothing.

This is also the shape every metric in the later sections turns out to have. Net benefit is this table with different numbers in it. So is cost-sensitive accuracy, and so is the bounded score at the end of the tutorial. Once a metric is a **payoff**, the interesting question stops being how it is computed and becomes which payoffs it assumes.

## The threshold the table implies

[Section 01](../what_is_a_metric/) left a threshold sitting at $0.5$ with no justification beyond habit. The payoff table supplies one, provided the score is a **calibrated probability**. The aside below explains what we mean by that:

:::note[What calibrated means]

A score is **calibrated** when it means, as a number, what it claims to mean as a probability.

$$
P\big(Y = 1 \;\big|\; s(X) = v\big) \;=\; v \qquad \text{for every } v
$$

Gather every case the model scored $0.30$. If close to thirty percent of them turn out to be positive, the score is calibrated there. If three percent are, it is not, and $0.30$ was a label rather than a probability.

It is a different property from ranking well, and **neither implies the other**. A model that reports the base rate $\pi$ for every case is perfectly calibrated and completely useless. A model that ranks flawlessly but squashes all of its scores into $[0.45, 0.55]$ has an untouched ROC curve and is badly calibrated.

Calibration is what lets a score be compared against a cost. Without it, asking whether $s(x) > \tfrac{1}{2}$ compares a number to nothing in particular. Most raw outputs are not calibrated, a margin, a vote share and an unadjusted network output included, and a later section dives deeper into this subject.
:::

Take a single case with score $p = s(x)$, and work out what each of the two available answers is worth. Predicting positive pays $1$ if the case really is positive, which happens with probability $p$, and $0$ otherwise. Predicting negative pays $1$ with probability $1 - p$.

$$
\mathbb{E}\big[U \mid \widehat{Y} = 1\big] = p, \qquad \mathbb{E}\big[U \mid \widehat{Y} = 0\big] = 1 - p
$$

Take whichever is larger. Predicting positive wins exactly when $p > 1 - p$, which is to say when $p > \tfrac{1}{2}$. So the accuracy of the model is maximised by the rule

$$
\widehat{Y} = 1 \quad \text{exactly when} \quad P(Y = 1 \mid x) > \tfrac{1}{2}
$$

which is the **Bayes rule** for the zero-one loss. The one half was never arbitrary after all. It fell out of the table, and it fell out because the table is symmetric: **the two mistakes cost the same, so the tie sits in the middle**. Put different numbers in the table and the tie moves.

### A still threshold

There is a trap here and I think it's worth spending some time on it.

The rule above is optimal at any prevalence. Nothing in the derivation mentioned $\pi$, so one half is the right cut whether positives are half the population or one case in a thousand. That sounds as though prevalence has stopped mattering.

It has not. _The threshold is stable, but the quantity being thresholded is not._ $P(Y = 1 \mid x)$ is a statement about a population as well as about a case. The same patient, with the same symptoms and the same measurements, carries a different probability of disease in a specialist referral clinic than in a general screening programme, because the two populations differ in how much disease is in them to begin with.

**Move the population and every posterior moves with it**, so cases that used to sit above one half can drop below it **without a single feature changing**. Prevalence did not leave the decision. It moved out of the threshold and into the score. A later section makes that migration exact.

### Without calibration, none of this holds

Every line above _assumed $s(x)$ is a genuine probability_. Most scores are not. A margin, a logit, an ensemble vote share and an uncalibrated network output are all perfectly usable for ranking, and for none of them does $0.5$ carry any meaning at all. It is a point on an arbitrary scale.

For such a score the accuracy-maximising threshold still exists, but it has to be found rather than derived: sweep $\tau$ across its range and read off the peak, exactly the curve from the opening section. The peak is wherever it is, and it moves with the prevalence, because for an uncalibrated score the threshold is doing the job the posterior was supposed to do.

## Biased is the wrong word

Here is the standard objection, stated as generously as it deserves:

> On a dataset where ninety-nine percent of cases are negative, a model that answers negative every time scores $0.99$. Accuracy is therefore biased under class imbalance, and should be replaced by a metric that is robust to it.

The first sentence is arithmetic and it is correct. The advice at the end is broadly good as reporting a bare accuracy figure on a $99/1$ problem really is bad practice. Everything in between is somewhat wrong.

### Nothing here is biased

An estimator is biased when its average value across repeated samples differs from the quantity it is meant to estimate. Accuracy computed on a test set is an average of independent indicators, and its expected value is exactly the population accuracy it names. It is unbiased, at any prevalence, for any model, always.

The $0.99$ in the objection is not a distortion of the truth. It is the truth. On that population, at that threshold, with those payoffs, the model really does earn $0.99$ per case. Nothing has been overstated. The number is correct and the complaint has to be about something else.

### Three complaints

The phrase covers three separate objections, and they have different causes and different cures.

1. **The number describes a population, not a model.** Move to a population with a different mix and the number moves, which is what the opening section showed. That is real, but it is not bias. It is the metric correctly reporting a different quantity, because a different quantity is what was asked for. The cure is to state the prevalence, not to change metric.
2. **The estimate is noisy when one class is scarce.** With forty positives in the test set, any number computed from those forty carries a wide interval. Also real, also not bias, and not about accuracy at all since it would harm recall, precision, or anything else estimated from the same forty cases.
3. **The payoff table is wrong for the problem.** A missed case and a false alarm were entered as costing the same, and depending on the application they are not.

### The third complaint is not about imbalance

**Imbalance without the defect.** Suppose the task is predicting whether a delivery arrives on a Tuesday, positives run at one percent, and the two mistakes genuinely cost the same because nothing much depends on either. Accuracy is fine. The $0.99$ baseline is real and uninteresting, and a model beating it is really doing something. The imbalance is extreme and no defect appears.

**The defect without imbalance.** Now take a perfectly balanced problem: half of the welds inspected have a flaw, half do not. A missed flaw eventually kills somebody. A false alarm costs an hour of re-inspection. Accuracy weights those two identically and is a catastrophic choice of metric, at $\pi = 0.5$, where nobody would think to call it an imbalance problem.

==Imbalance is neither necessary nor sufficient for accuracy to mislead.== What makes it mislead is a payoff table that does not match the application. Imbalance is merely the condition under which a bad table produces an embarrassing number rather than a merely wrong one, which is why the two get confused.

### Why the confusion is expensive

A misdiagnosis leads to the wrong prescription. If the problem is believed to be imbalance, the cure looks like finding a metric that is robust to imbalance, and the search ends at whichever metric moves least when the mix changes. If the problem is understood as a wrong payoff table, the cure is to write down the right one.

## What cannot be repaired

Only one of the three complaints was about accuracy itself, and it was the payoff table. So correct it. Work out what a missed case is worth, put that number in, and carry on.

That works, and it is the end of accuracy. The values $1$, $0$, $0$, $1$ are not a setting accuracy ships with. They are what the word means. Replace them and the arithmetic still runs, but what comes out is a different quantity with a different name. _There is no version of accuracy that carries a cost._

### So why is it always low prevalence

Imbalance is neither necessary nor sufficient, and that stands. Low prevalence is still where accuracy fails nearly every time, for three unrelated reasons.

1. **Rarity and asymmetry travel together.** Things are usually rare on account of being bad: disease, fraud, fracture, default. Low prevalence does not imply a broken table. It predicts one.
2. **The scale collapses.** At $\pi = 0.01$ everything worth arguing about is packed into the last hundredth, so large real differences surface in the third decimal place.
3. **The measurement gets noisy.** The $\pi \cdot \mathrm{TPR}$ half is estimated from however many positives the test set happened to hold. A later section is devoted to this.

The first needs a cost, the second a baseline, the third more positives. _None of them is answered by a metric that moves less when the class mix changes._

If accuracy is reported anyway, print $\max(\pi,\, 1 - \pi)$ beside it, which is what a model earns by ignoring its input and always naming the larger class. The gap is the only part of the number that describes the model. At $\pi = 0.20$, a reported $0.94$ sits against a baseline of $0.80$. At $\pi = 0.01$, a reported $0.99$ sits against a baseline of $0.99$, and the gap is nothing at all.

Balanced accuracy, next, leaves the payoff table exactly where it is and changes the prevalence instead.
