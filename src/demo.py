import gradio as gr
from dino import Dino
from autodistill.detection import CaptionOntology
from dataset import Dataset
import benchmark as bench
import os

dataset = Dataset("/Users/francescotacinelli/Developer/datasets/pallets_sorted/test/")
cap_ont = dataset.captions.captions_ontology

ground_truths = []

try:
    model = Dino(ontology=CaptionOntology(cap_ont))
except Exception as e:
    raise e


def predict(img_path, alias):
    predictions = model.gradio_predict(img_path, alias)
    metrics = get_pr(predictions)
    return (img_path, predictions), metrics


def get_grounds(grounds: str):
    global ground_truths
    with open(grounds, "r") as f:
        gr_lines = f.read().splitlines()
        for line in gr_lines:
            class_id, *bbox = line.split(" ")
            ground_truths.append((int(class_id), [float(coord) for coord in bbox]))

    ground_truths = ground_truths


def get_pr(predictions, class_id=2):
    true_positives, false_positives, false_negatives = bench.get_confMatr(
        predictions, ground_truths, class_id
    )
    labels = {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
    return labels


with gr.Blocks() as demo:
    label = gr.Textbox(label="Label", placeholder="Enter your label here")
    input = gr.Image(label="Image", type="filepath")
    grounds = gr.File(label="Ground Truth", type="filepath")
    pred_btn = gr.Button(value="Predict")
    output = gr.AnnotatedImage(
        label="Output",
    )
    metrics = gr.Label(label="Metrics")
    pred_btn.click(predict, inputs=[input, label], outputs=[output, metrics])
    grounds.upload(get_grounds, inputs=[grounds])

if __name__ == "__main__":
    demo.launch()
