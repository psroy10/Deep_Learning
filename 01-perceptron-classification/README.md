# Perceptron Classification (One-vs-One)

A single-layer perceptron built entirely from scratch in Python — no `sklearn.Perceptron`,
no autograd, no neural network libraries. Just weights, a dot product, an activation
function, and manually-coded gradient descent. Extended to multi-class problems using the
**One-vs-One (OvO)** strategy: for *k* classes, train `k*(k-1)/2` independent binary
classifiers, then combine their votes.

The results tell a genuinely interesting story: **this simple model is perfect on one
dataset and falls apart on the other — for reasons that make complete sense once you look
at the data.**

## TL;DR results

|                  | Linearly Separable | Nonlinearly Separable |
|------------------|---------------------|------------------------|
| Sigmoid accuracy | **100%**            | 55.6%                  |
| Tanh accuracy    | **100%**            | 60.7%                  |
| Verdict          | Solved perfectly    | Can't be solved by a straight line |
## The two experiments

### `linearly_separable/` — where the perceptron shines
Three classes, each a tight, well-separated 2D cluster. A straight line comfortably fits
between every pair. Both activations converge in **under 10 epochs** and hit **100% test
accuracy** with a perfectly diagonal confusion matrix. This is the perceptron doing exactly
what it's good at.

### `nonlinearly_separable/` — where a straight line just isn't enough
Three classes shaped like **concentric rings** — a small cluster, surrounded by an arc,
surrounded by a much larger ring. No straight line can separate an inner region from
something that fully encircles it, so two of the three pairwise classifiers never converge —
they plateau at a stubbornly high error and stay there for the full 10,000+ epochs.

The consequence is worse than "a bit less accurate": since 2 of the 3 pairwise votes
involve the outer-ring class, that class ends up **winning almost every majority vote**,
even for points that are clearly in the inner cluster. With sigmoid, this collapses
completely — every single test point gets predicted as the outer class (0% recall for the
other two). Tanh (with online/stochastic updates) does somewhat better, spreading real
predictions across all three classes, but still tops out around 60% — a linear model
genuinely cannot solve this problem, no matter how it's trained.

**This isn't a bug — it's the perceptron correctly showing its limits.**

## Code structure
