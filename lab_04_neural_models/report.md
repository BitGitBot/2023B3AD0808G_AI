# Neural models: short report

The [executed notebook](neural_models.ipynb) contains the code, exact implementation prompts, plots, full gradient tensors, and checks in task order. Code and written answers were prepared with Codex and verified by execution.

## Task 1: Problem

Inputs are the four points in `{0,1}²`; binary labels are `[0,1,1,0]`. Opposite-corner classes have intersecting convex hulls, so one linear boundary cannot separate XOR.

## Task 2: Design

Use a 2-2-1 model with tanh hidden units and binary logits, trained with mean BCEWithLogitsLoss. Sigmoid converts logits to probabilities. CPU Adam uses seed 0, learning rate 0.05, and 3,000 full-batch steps. Success requires loss below 0.05 and below its initial value, 4/4 correct labels, and finite nonzero early gradients.

## Task 3: Implementation

The notebook records the implementation and extension prompts before their code. The submitted code needed no corrections before execution. The forward pass, scalar loss, backward pass, and optimizer update are identified there. Runtime: PyTorch 2.14.1+cpu, NumPy 2.3.5.

## Task 4: Results

Binary loss fell from **0.715159** to **0.000083**. Probabilities, in input order `(0,0), (0,1), (1,0), (1,1)`, were `[0.000059, 0.999899, 0.999877, 0.000051]`; predictions were `[0,1,1,0]`.

The first-step hidden-weight gradient was `[[0.000505, 0.000607], [-0.042569, -0.044773]]`. The batch gradient matched the mean of four independently computed example gradients.

Zeroing every weight and bias preserved identical hidden rows at steps 0, 1, 10, 100, 1,000, and 3,000. Probabilities stayed at 0.5; final loss was 0.693147.

| Hidden activation | Final loss | Correct | Early gradient norm |
|---|---:|---:|---:|
| Sigmoid | 0.477423 | 3/4 | 0.000891 |
| Tanh | 0.000083 | 4/4 | 0.061785 |
| ReLU | 0.693149 | 2/4 | 0.001699 |

Only the activation changed. Sigmoid showed saturated outputs; all final ReLU pre-activations were negative. These observations describe one seed and training configuration, not a universal ranking.

## Task 5: Three classes

The output became three logits with a `(3,2)` weight matrix and mean CrossEntropyLoss; the two tanh hidden units and training settings were retained. Final loss was **0.000039**, with 4/4 correct predictions.

| Input | P(class 0) | P(class 1) | P(class 2) | Prediction |
|---|---:|---:|---:|---:|
| (0,0) | 0.999951 | 0.000049 | 0.000000 | 0 |
| (0,1) | 0.000023 | 0.999969 | 0.000007 | 1 |
| (1,0) | 0.000023 | 0.999970 | 0.000007 | 1 |
| (1,1) | 0.000000 | 0.000045 | 0.999955 | 2 |

The first probability vector summed to 0.99999994. Adding 100 to every logit changed probabilities by at most 2.33e-10. The measured logit gradient matched `(p - one_hot(y))/4`; dividing by four follows from mean reduction. Subtracting the maximum logit before exponentiation avoids overflow. Language-model vocabulary classification uses the same output mathematics with a much larger contextual architecture.

## Reflection

1. Affine-only depth cannot express XOR; nonlinear hidden features can.
2. Lower loss and correct labels demonstrate a useful learning signal.
3. Identical units receive identical gradients, preserving their symmetry.
4. Activation derivatives affect gradients; the table reports the measured effect for these settings.
5. Output and loss must match binary versus categorical targets.
6. Codex reduced implementation work; execution was essential for validating claims and exposing failed runs.
7. Keep loss, prediction, shape, normalization, and finite-gradient checks at scale; sample costly finite-difference checks.
