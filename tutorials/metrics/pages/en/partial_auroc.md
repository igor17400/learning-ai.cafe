---
title: "Partial AUROC"
subtitle: "Caring about the top of the ranking without leaving ROC space."
---

AUROC averages over every threshold, giving each the same weight. That is the right thing to do when every threshold is a live possibility. Very often none of them are.

## Most of the curve is unreachable

Richardson et al. give the motivating case. A drug discovery campaign screens $100{,}000$ candidate antibodies and can afford to synthesise and test $100$ of them ([Richardson et al., 2024](../references/)). The model ranks the candidates, the top hundred get made, and every other point on the ROC curve describes an experiment nobody will run.

The pattern is everywhere once you look. A radiologist can review forty flagged scans a day. A fraud team can call three hundred customers. A hiring pipeline can interview twelve people. _The capacity to act is fixed and small, and it fixes where on the curve you will stand_, long before any model is trained.

Under that constraint, AUROC is answering a question with the wrong scope. Half its integral covers the region past $\mathrm{FPR} = 0.5$, where the model flags the majority of healthy people, and no budget survives contact with that.

## Integrate less of it

The repair is to stop integrating where you cannot go. Pick a ceiling $f$ on the false positive rate and take the area only up to it.

$$
\mathrm{pAUC}(f) \;=\; \int_{0}^{f} \mathrm{TPR}\big(\mathrm{FPR} = u\big)\, \mathrm{d}u
$$

This is not yet comparable to anything. Its range depends on $f$: a model ranking at random earns $f^2/2$, and a perfect one earns $f$, so for $f = 0.1$ every possible value is squeezed between $0.005$ and $0.1$. That is section 03's collapsing scale, in a new costume.

The McClish correction rescales the interval so the familiar anchors come back.

$$
\mathrm{AUROC}_{f} \;=\; \frac{1}{2}\left(1 \;+\; \frac{\mathrm{pAUC}(f) - \tfrac{1}{2}f^{2}}{f - \tfrac{1}{2}f^{2}} \right)
$$

Random scores now give $0.5$ and a perfect ranking gives $1$, exactly as with full AUROC, so the two live on the same scale and a reader knows what $0.7$ means. In scikit-learn this is `roc_auc_score(y, s, max_fpr=0.1)`, which applies the correction ([Richardson et al., 2024](../references/)).

### What survives the restriction

Everything section 07 established. $\mathrm{pAUC}$ is built from $\mathrm{TPR}$ and $\mathrm{FPR}$ and nothing else, both divide by column totals, so ==restricting the range does not reintroduce prevalence==. The metric expresses a preference for the top of the ranking while remaining invariant to the class mix.

## When one number hides two models

The restriction is not a cosmetic adjustment. Two models can have identical AUROC and behave completely differently in the only region you can reach.

:::figure{#partial_auroc_crossing}
![Two ROC curves on the same axes, both with area 0.80, crossing at a false positive rate of about 0.28. A shaded vertical band marks the region below false positive rate 0.1. Inside that band Model A rises far more steeply, reaching a true positive rate of 0.56 while Model B reaches only 0.34. A legend gives both models AUROC 0.80, with corrected partial AUROC of 0.71 for A and 0.57 for B.](../../figures/partial_auroc_crossing.svg)

Two models, both scoring $\mathrm{AUROC} = 0.80$. Inside the affordable band, model A catches $56\%$ of the positives and model B catches $34\%$. Corrected to the same scale that is $0.71$ against $0.57$. A single summary number reported these two as equals, and on the only stretch of curve anybody can use, one of them finds two thirds again as much as the other.
:::

What makes the example possible is that the curves cross, and that deserves a second look, because there is an objection lurking in it. Richardson et al. repeat Fawcett's warning that areas should not be compared when the underlying curves cross, since the summary is then averaging over a reversal ([Richardson et al., 2024](../references/)). We have just compared two areas.

The warning lands on the full-range number, not the restricted one. Across the whole curve, $\mathrm{AUROC} = 0.80$ for both models, and that equality is a truce: a stretch where A is better traded against a stretch where B is. Averaging the two into one figure is exactly the thing Fawcett says not to do, and it is what the reported equality did.

The restricted comparison is safe for a specific reason. The crossing sits at $\mathrm{FPR} \approx 0.28$, which is outside the band. Inside $\mathrm{FPR} \leq 0.1$ the two curves never meet and model A is above model B at every point, _so the partial areas are summarising an agreement rather than papering over a disagreement._

Which gives a rule worth carrying: compare areas only over a range on which one curve dominates the other throughout. Choosing that range by what you can afford to do will often hand you the condition for free, and when it does not, the curves crossing inside your own operating band is itself the finding.

## Where $f$ has to come from

Richardson et al. recommend $f = 0.1$. That value comes from B cell and T cell epitope prediction, where it is conventional, and the paper presents it as general guidance ([Richardson et al., 2024](../references/)). The same paper elsewhere criticises a competing method for fixing a parameter that ought to depend on the data, and the objection applies to their own choice with equal force. ==Inheriting $f = 0.1$ because a paper used it is the same mistake as inheriting $\tau = 0.5$ because the library defaulted to it.==

Fortunately $f$ can be derived rather than borrowed. A ceiling on $\mathrm{FPR}$ is a ceiling on the number of false positives, since $\mathrm{FP} \approx f \cdot N$. If the negative class has $N$ cases and you can act on $k$ of them in total, then

$$
f \;\approx\; \frac{k}{N}
$$

For the antibody campaign, $k = 100$ out of roughly $N = 100{,}000$ negatives, so $f \approx 0.001$, which is a hundred times stricter than the recommended convention. For the radiologist reviewing forty scans a day out of two thousand, $f \approx 0.02$. Neither number is a matter of taste. Both are read off a schedule.

It is worth naming what just happened, because the rest of the tutorial builds on it. _A capacity constraint is a cost statement wearing work clothes._ Saying that you can afford a hundred experiments is saying something about what a wasted experiment costs relative to a missed discovery. Partial AUROC lets that enter the metric, but only through a single blunt instrument, a hard ceiling, with everything below the ceiling still weighted equally.

Section 14 does the same job properly, with a cost ratio instead of a cutoff. Before that, section 09 takes the other route out of the ROC curve, and shows what happens when the preference for the top of the ranking is expressed implicitly instead.
