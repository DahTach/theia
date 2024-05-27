import gradio as gr
from caption import captions
from dino import Dino
from autodistill.detection import CaptionOntology
import utils
import cv2 as cv
from tqdm import tqdm
import numpy as np
import torch


class GradioApp:
    def __init__(self) -> None:
        self.max_captions = 8
        self.aliases = {}
        self.update_calls = 0

    def load_model(self):
        try:
            self.model = Dino(
                ontology=CaptionOntology(utils.get_captions(captions)), device=None
            )

            return "Grounding DINO Ready"
        except Exception as e:
            raise e

    def get_device(self):
        if torch.cuda.is_available():
            print("using cuda")
            return "cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("using cpu because mps is not fully supported yet")
            return "cpu"
        else:
            print("using cpu")
            return "cpu"

    def predict_base(self, image, image_path, progress):
        base = self.model.base_predict(image_path, progress=progress)
        base_image = self.model.draw_base(image, base)
        return base_image

    def predict_nms(self, image, progress):
        nms_results = self.model.nms_predict(image, progress=progress)
        nms_image = self.model.draw_nms(image, nms_results)
        return nms_image

    def predict_batch(self, image, progress):
        batch_results = self.model.batch_predict(image, progress=progress)
        batch_image = self.model.draw_batch(image, batch_results)
        return batch_image

    def predict(self, image_path, progress=gr.Progress(track_tqdm=True)):
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        prompts = sum(len(captions[cls]) for cls in captions)
        base_progress = prompts
        nms_progress = prompts
        batch_progress = len(captions.keys())
        total_progress = base_progress + nms_progress + batch_progress
        progress(progress=0, total=total_progress, desc="Importing", unit="documents")
        base_image = self.predict_base(image, image_path, progress=progress)
        nms_image = self.predict_nms(image, progress=progress)
        batch_image = self.predict_batch(image, progress=progress)
        return base_image, nms_image, batch_image

    def interface(self):
        with gr.Blocks() as demo:
            with gr.Row():
                model_status = gr.Textbox(
                    label="Model Status",
                    placeholder="Loading...",
                    value="Loading...",
                    interactive=False,
                )
                gr.Button(interactive=False, value=self.get_device())
            with gr.Row():
                image = gr.Image(label="Target Image", type="filepath", height=500)

            comparison = gr.Gallery(
                label="Generated images",
                show_label=False,
                rows=1,
                columns=3,
                object_fit="contain",
                height=700,
            )

            predict = gr.Button(value="Predict")

            predict.click(
                fn=self.predict,
                inputs=[image],
                outputs=comparison,
            )

            demo.load(self.load_model, outputs=model_status, show_progress=True)

        self.demo = demo


def main():
    app = GradioApp()
    app.interface()
    app.demo.launch()


if __name__ == "__main__":
    main()
