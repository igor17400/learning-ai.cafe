---
title: "ROC and AUROC"
subtitle: "Invariance is not neutrality. It is a prior over deployment prevalence, chosen by the model."
---

Every metric in the last four sections picked a threshold and reported what happened there. Each picked a different one, and none of them said so. The obvious escape is to stop picking.

This is the pivot of the tutorial, and it needs two readings held at once. One says the resulting number is the only honest thing here. The other says it is the most deceptive. Both are correct, and they are answers to different questions.

## The curve

Section 02 ended with the observation that a single threshold gives a single pair of rates, so sweeping the threshold gives a curve. That curve is the **Receiver Operating Characteristic (ROC)**.

$$
\mathrm{ROC} \;=\; \big\{\,\big(\mathrm{FPR}(\tau),\; \mathrm{TPR}(\tau)\big) \;:\; \tau \in \mathbb{R} \,\big\}
$$

Read the definition for what is missing. _Both coordinates are column ratios, so no prevalence appears anywhere on either axis_, and $\tau$ has been swept away rather than chosen. It also puts the last four sections in their place. Accuracy, balanced accuracy and F1 each nominated a threshold, and a threshold is a point on this curve.

:::figure{#roc_operating_points}
![A receiver operating characteristic curve rising steeply from the origin, with the area beneath it shaded and labelled AUROC equals 0.83, and a dashed diagonal chance line. Three points are marked on the curve. Accuracy sits lowest at a threshold of 0.50, catching nine percent of cases. F1 sits in the middle at a threshold of 0.21, catching fifty-four percent. Balanced accuracy sits highest at a threshold of 0.10, catching eighty percent.](../../figures/roc_operating_points.svg)

One model at $\pi = 0.10$, and the three thresholds the previous sections handed us. They are not three results, but three places to stand on one curve.
:::

### Where those three thresholds came from

Each was derived in an earlier section, and the model in the figure is a single fixed one, so the three rules can be applied to it and compared directly.

| Metric            | Rule                   |   Here | Fixed by                                           |
| ----------------- | ---------------------- | -----: | -------------------------------------------------- |
| Accuracy          | $\tau = \tfrac{1}{2}$  | $0.50$ | nothing. It is the same constant on every problem. |
| Balanced accuracy | $\tau = \pi$           | $0.10$ | the evaluation set, through its base rate.         |
| $F_1$             | $\tau = F_1^{\star}/2$ | $0.21$ | the model, through the score it manages to attain. |

Those three numbers fall in the order $0.50$, $0.21$, $0.10$, and that ordering is the reason the dots climb the curve. A lower threshold flags more cases, which raises $\mathrm{TPR}$ and $\mathrm{FPR}$ together and slides the dots to the right. ==The disagreement between the three metrics is not about the model. It is about how far along this one curve to walk==. The curve says something none of those four numbers could. It **describes the model across every operating point at once**, so two models can be compared without first agreeing on where to stand.

## The area, and what it means

To systematically compare models across their operating range, we calculate the area under their ROC curves (AUROC). The interesting part is that this geometric area is exactly equal to a specific ranking probability:

$$
\mathrm{AUROC} \;=\; P\big(s(X^{+}) > s(X^{-})\big)
$$

where $X^{+}$ is a positive case drawn at random and $X^{-}$ a negative one. The area under the ROC curve is the probability that the model scores a random positive above a random negative. _It is a statement about ranking_, and it never mentions a threshold because it considers all of them.

McDermott et al. give a second form that makes the structure explicit ([McDermott et al., 2024](../references/)).

$$
\mathrm{AUROC} \;=\; 1 \;-\; \mathbb{E}_{\tau \sim s \mid Y = 1}\big[\mathrm{FPR}(\tau)\big]
$$

Think of this as a step-by-step procedure. Take a positive case, use its score as a threshold, and record the false positive rate at that exact point. Average that result across all positive cases. Here's one of the most interesting parts to this. _Every false positive is penalized equally, no matter where it sits on the score distribution_, because the expectation is unweighted.

### The invariance

Neither of the forms above contains the prevalence $\pi$ and this is structural since both are built from $\mathrm{TPR}$ and $\mathrm{FPR}$, and section 02 established that those divide by column totals. What happens practically is that we can sample as many extra negatives as we like, drawn from the same population, and the ROC curve does not move.

Richardson et al. demonstrate exactly this by simulation, holding the class-conditional score distributions fixed and varying the ratio across $1{:}1$, $1{:}9$ and $1{:}99$. The ROC-AUC distribution is unchanged ([Richardson et al., 2024](../references/)). And their conclusion on this is that the ROC-AUC of a classifier is not inflated by class imbalance.

## What the invariance costs

Now the other reading. Flores et al. prove that for a **calibrated score**, AUROC is not threshold-free at all. It is an average of threshold decisions ([Flores et al., 2025](../references/)).

$$
\mathrm{AUROC}(s) \;=\; \tfrac{1}{2}\, \mathbb{E}_{t \sim s[\mathcal{D}_{1/2}]}\Big[\mathrm{PAMNB}\big(\mathcal{D}_{1 - t},\, s, \, \tfrac{1}{2},\, \tfrac{1}{2}\big)\Big]
$$

The inner term, the prior-adjusted maximum net benefit (PAMNB), quantifies the model's payoff on a population with prevalence $1-t$ under equal misclassification costs. Consequently, AUROC represents this payoff averaged across a spectrum of prevalences. Contrary to popular belief, **AUROC does not ignore deployment prevalence, it implicitly averages over all possible prevalences**.

But that raises a question: how is that average weighted? The equation's subscript gives it away. The averaging distribution, $s[\mathcal{D}_{1/2}]$, is just the model's own distribution of scores. The metric gives the most weight to the prevalences that sit closest to where your model happens to output most of its scores.

Ok... that's probably not clear. Let's try to make it easier. If we have a screening problem where the true prevalence is somewhere between $0.5\%$ and $2\%$, AUROC will still anchor its evaluation wherever the scores pile up. If they cluster around $0.3$, AUROC is heavily weighting its evaluation on how the model would perform on a dataset that is $30\%$ positive. The metric answers a question about a hypothetical population we might never see, and we cannot tell it otherwise, because the formula has no way to input the actual base rate.

### Holding both readings

The two papers agree completely on the mathematics and disagree entirely on what it is worth. It is worth being precise about why, because it is not a dispute that resolves.

| Question                                                                        | Verdict on the invariance                                                                                     |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Does the number move when I change the class mix on a fixed population?         | A **virtue**. It does not, and it should not, because the model did not change. (Richardson)                  |
| Which decision, at which deployment prevalence, does this number correspond to? | A **defect**. Invariance means the metric refuses to let you state the prevalence you actually face. (Flores) |

Both are right. A metric that does not move when we resample the negatives is **measuring the model**. A metric that cannot be told what prevalence it will deploy at is not answering a **decision question**. ==Invariance and neutrality are not the same thing.== AUROC has an opinion about deployment prevalence. Its opinion is _all of them, weighted by its own score histogram_.

## Two conditions worth naming

**The class-conditional score distributions must be fixed.** Adding negatives identical to the existing distribution preserves the AUROC. However, real-world prevalence shifts often introduce out-of-distribution negatives. As Richardson et al. demonstrate ([Richardson et al., 2024](../references/)), enriching a dataset with novel, low-scoring negatives alters the underlying score distribution and forces the AUROC to shift.

**Scores must be calibrated.** Flores's interpretation of AUROC as an averaged net benefit strictly requires calibrated scores. If scores are uncalibrated, the metric remains purely a measure of ordinal ranking. AUROC only translates decision-theoretic insights when the score values carry reliably probabilistic meaning.
