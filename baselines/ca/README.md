# Use Concept Ablation

## Generate Dataset

example for gun concept

```bash
python generate_dataset.py --concept gun --output-dir data/gun
```

## Train

```bash
python train.py --concept gun --dataset-dir data/gun -e 10 -b 0.1 --output-dir output/ca_epoch_10_beta_0_1 --device1 0 --device2 2
```

## Inference
Load the model as normal

```python
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("output/ca_epoch_10_beta_0_1", torch_dtype=torch.bfloat16)
```
