import gradio as gr
import torch
from itertools import zip_longest

# from gradio_counter import Counter as grCounter
from caption import captions
import utils
from dino import Dino
from autodistill.detection import CaptionOntology
import cv2 as cv
from typing import List, Tuple
import itertools

"""
Interface Steps:
- define captions dictionary
- dropdown for model selection
- image input
- button to predict
- annotated image output
- counter output
"""


class GradioApp:
    def __init__(self) -> None:
        self.max_captions = 8
        self.aliases = {}
        self.update_calls = 0
        self.device = self.get_device()

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

    def load_model(self):
        try:
            self.model = Dino(
                ontology=CaptionOntology(utils.get_captions(captions)),
                device=self.device,
            )
        except Exception as e:
            raise e

    def predict(self, image_path, progress=gr.Progress(track_tqdm=True)):
        """
        As output component:
        Expects a a tuple of a base image and list of annotations:
        a tuple[Image, list[Annotation]].
        The Image itself can be str filepath, numpy.ndarray, or PIL.Image.
        Each Annotation is a tuple[Mask, str].
        The Mask can be either:
        - a tuple of 4 int's representing the bounding box coordinates (x1, y1, x2, y2)
        - 0-1 confidence mask in the form of a numpy.ndarray of the same shape as the image,
        while the second element of the Annotation tuple is a str label.

        """
        results = self.model.base_predict(image_path)
        """
        results = sv.Detections(
            xyxy=xyxy,
            mask=mask,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
        )
        """
        boxes = results.xyxy
        ids = results.class_id
        aliases_list = []
        for _, aliases in captions.items():
            for alias in aliases:
                aliases_list.append(alias)

        annotations = []
        for box, id in zip(boxes, ids):
            try:
                name = aliases_list[id]
            except IndexError:
                name = "unknown"
                print(f"Unknown class id: {id}")
            box = (int(b) for b in box)
            annotations.append((box, str(name)))
        return image_path, annotations

    def make_captions(self, k):
        k = int(k)
        return [gr.Textbox(visible=True)] * k + [gr.Textbox(visible=False)] * (
            self.max_captions - k
        )

    def pairwise(self, iterable):
        iterable = list(iterable)
        for i in zip_longest(iterable[0::2], iterable[1::2]):
            yield (i)

    def get_captions(self):
        """Inverts a dictionary with list values.

        Args:
            d: The dictionary to invert.

        Returns:
            The inverted dictionary.
        """

        inverted_dict = {}
        for key, value in self.aliases.items():
            for item in value:
                inverted_dict.setdefault(
                    item, key
                )  # Set default value for repeated values
        return inverted_dict

    def update_aliases(self, alias, cls):
        self.update_calls += 1
        try:
            self.aliases[cls].append(alias)
        except KeyError:
            self.aliases.setdefault(cls, [alias])

    def interface(self):
        with gr.Blocks() as demo:
            with gr.Row() as labels:
                for elem1, elem2 in self.pairwise(captions.keys()):
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                with gr.Column(scale=3):
                                    cls = gr.Dropdown(
                                        label="Class",
                                        choices=list(captions.keys()),
                                        value=elem1,
                                    )

                                    s = gr.Slider(
                                        1,
                                        maximum=self.max_captions,
                                        value=len(captions[elem1]),
                                        step=1,
                                        visible=False,
                                    )

                                with gr.Column(scale=1):
                                    with gr.Row():

                                        def plus(n):
                                            return n + 1

                                        add_alias = gr.Button("Add Alias", size="sm")
                                        add_alias.click(plus, inputs=s, outputs=s)

                                        def minus(n):
                                            return n - 1

                                        del_alias = gr.Button("Remove Alias", size="sm")
                                        del_alias.click(minus, inputs=s, outputs=s)

                            with gr.Row():
                                aliases1 = []

                                for i in range(self.max_captions):
                                    alias = gr.Textbox(
                                        show_label=False,
                                        value=(
                                            captions[elem1][i]
                                            if i < len(captions[elem1])
                                            else "alias"
                                        ),
                                        interactive=True,
                                    )

                                    if alias.value != "alias":
                                        try:
                                            self.aliases[elem1].append(alias.value)
                                        except KeyError:
                                            self.aliases.setdefault(
                                                elem1, [alias.value]
                                            )

                                    alias.blur(
                                        self.update_aliases,
                                        inputs=[alias, cls],
                                    )

                                    aliases1.append(alias)

                                s.change(self.make_captions, s, aliases1)

                            demo.load(
                                self.make_captions,
                                inputs=s,
                                outputs=aliases1,
                            )

                        if elem2:
                            with gr.Column(scale=1):
                                with gr.Row():
                                    with gr.Column(scale=3):
                                        cls = gr.Dropdown(
                                            label="Class",
                                            choices=captions.keys(),
                                            value=elem2,
                                        )

                                        s = gr.Slider(
                                            1,
                                            maximum=self.max_captions,
                                            value=len(captions[elem2]),
                                            step=1,
                                            visible=False,
                                        )

                                    with gr.Column(scale=1):
                                        with gr.Row():

                                            def plus(n):
                                                return n + 1

                                            add_alias = gr.Button(
                                                "Add Alias", size="sm"
                                            )
                                            add_alias.click(plus, inputs=s, outputs=s)

                                            def minus(n):
                                                return n - 1

                                            del_alias = gr.Button(
                                                "Remove Alias", size="sm"
                                            )
                                            del_alias.click(minus, inputs=s, outputs=s)

                                with gr.Row():
                                    aliases2 = []

                                    for i in range(self.max_captions):
                                        alias = gr.Textbox(
                                            show_label=False,
                                            value=(
                                                captions[elem2][i]
                                                if i < len(captions[elem2])
                                                else "alias"
                                            ),
                                            interactive=True,
                                        )

                                        if alias.value != "alias":
                                            try:
                                                self.aliases[elem2].append(alias.value)
                                            except KeyError:
                                                self.aliases.setdefault(
                                                    elem2, [alias.value]
                                                )

                                        alias.blur(
                                            self.update_aliases,
                                            inputs=[alias, cls],
                                        )
                                        aliases2.append(alias)

                                    s.change(self.make_captions, s, aliases2)

                                demo.load(
                                    self.make_captions,
                                    inputs=s,
                                    outputs=aliases2,
                                )

            # TODO: Implement class addition and removal

            with gr.Row():
                image = gr.Image(label="Image", type="filepath")

            with gr.Row():
                results = gr.AnnotatedImage(label="Results")

            with gr.Row():
                predict = gr.Button(value="Predict")

            predict.click(self.predict, inputs=[image], outputs=[results])

            demo.load(self.load_model, show_progress=True)

        self.demo = demo


def main():
    app = GradioApp()
    app.interface()
    app.demo.launch()


if __name__ == "__main__":
    main()
