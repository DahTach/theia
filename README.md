# Theia Autolabeling

## Installation

```bash
pdm install
```

## To fix the plot issue

Replace row 62 of `.venv/lib/python3.12/site-packages/autodistill/utils.py`

```python
    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _, _ in detections
    ]
```

## To run the inference on an image folder

```bash
label -i <path to image folder> optional: --save --batch
```

results are saved in `<image-folder>_results` folder

## Results and Observations

The better performing models seem to be:

- GroundingSam
- GroundingDino

GroundingDino appears to achieve better results in non batched mode.


test
