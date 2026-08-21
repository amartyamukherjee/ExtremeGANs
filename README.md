# ExtremeGANs: A Library and Benchmark for GAN-Based Rare-Event Generation

`extreme_gans` provides GAN models for extreme multivariate time-series generation through a Python-first API.

## Install

Once the package is published to PyPI:

```bash
pip install extreme_gans
```

For local development from a clone of this repository:

```bash
pip install -e .
```

## Quick start

```python
import extreme_gans

# 1. Load and prepare the time-series dataset
dataset = extreme_gans.data.load_generic_csv(
    "data/exchange_rate.csv",
    horizon=64,
    drop_cols=("date", "OT"),
    normalize="zscore",
)

# 2. Configure the algorithm
model = extreme_gans.algos.htgan(
    alpha=1.5,
    latent_dim=128,
)

# 3. Train
model.fit(
    dataset,
    epochs=200,
    batch_size=256,
    lr=1e-4,
)

# 4. Evaluate
metrics = model.evaluate(dataset, n_samples=10_000)
print(metrics)

# 5. Generate synthetic time-series windows in the original data scale
samples = model.generate(n_samples=10_000)
print(samples.shape)  # (10000, 64, number_of_features)
```

The dataset object carries the normalized windows, original-scale windows,
feature names, and fitted scaler internally. The model object carries the
trained generator and any algorithm-specific transform needed to map generated
samples back to the original scale.

## Available algorithms

```python
extreme_gans.algos.htgan(...)
extreme_gans.algos.evgan(...)
extreme_gans.algos.tailgan(...)
extreme_gans.algos.paretogan(...)
extreme_gans.algos.cdcgan(...)
extreme_gans.algos.wgan(...)
extreme_gans.algos.vanilla(...)
```

All return the same estimator-style object supporting:

```python
model.fit(dataset, ...)
model.evaluate(dataset, ...)
model.generate(n_samples=...)
```

Users can access the underlying PyTorch generator after fitting via
`model.generator_`, and the recorded training history via `model.history_`.
