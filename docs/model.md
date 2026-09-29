# Model
U-Net with one input channel and one sigmoid-logit output channel. Training loss is BCE-with-logits plus Dice loss. Train/validation/test splits are scene-level where scene IDs can be derived from the pairing table. Metrics are generated only after actual evaluation.
