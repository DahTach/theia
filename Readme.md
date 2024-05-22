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
label -i <path to image folder> optional: -m <model name >
```

