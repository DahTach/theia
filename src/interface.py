import gradio as gr
import atexit
import torch
from itertools import zip_longest

# from gradio_counter import Counter as grCounter
from caption import Captions
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
        self.captions = Captions()
        self.update_calls = 0
        self.device = self.get_device()

    @property
    def aliases(self):
        return self.captions.dict

    def exit_handler(self):
        self.captions.save()
        # unload model
        self.model = None

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
                ontology=CaptionOntology(utils.get_captions(self.captions.dict)),
                device=self.device,
            )
        except Exception as e:
            raise e

    def predict(self, image_path, progress=gr.Progress(track_tqdm=True)):
        prompts = []
        for cls, aliases in self.aliases.items():
            for alias in aliases:
                prompts.append(alias)

        results = self.model.base_predict(
            image_path, prompts=prompts, progress=progress
        )

        annotations = self.postprocess_results(results)

        return image_path, annotations

    def postprocess_results(self, results):
        boxes = results.xyxy
        ids = results.class_id
        aliases_list = []
        for _, aliases in self.aliases.items():
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
        return annotations

    def make_captions(self, k):
        k = int(k)
        return [gr.Textbox(visible=True)] * k + [gr.Textbox(visible=False)] * (
            self.max_captions - k
        )

    def pairwise(self, iterable):
        iterable = list(iterable)
        for i in zip_longest(iterable[0::2], iterable[1::2]):
            yield (i)

    # def make_aliases(self):
    #     self.aliases = captions
    #     for key, item in captions.items():
    #         self.aliases.setdefault(key, [])
    #         self.aliases[key] = item + ["alias" * (self.max_captions - len(item))]

    def update_captions(self, caption, key):
        class_id, index = key.split(":")
        self.captions.edit(class_id, caption, key)

    def old_update_aliases(self, alias, key):
        class_id, index = key.split(":")
        self.update_calls += 1
        try:
            self.aliases[class_id][index] = alias
        except KeyError:
            self.aliases.setdefault(class_id, [alias])

    def interface(self):
        with gr.Blocks() as demo:
            with gr.Row() as labels:
                for elem1, elem2 in self.pairwise(self.aliases.keys()):
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                with gr.Column(scale=3):
                                    gr.Button(value=elem1, interactive=False)

                                    s = gr.Slider(
                                        1,
                                        maximum=self.max_captions,
                                        value=len(self.aliases[elem1]),
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
                                    key = gr.Textbox(
                                        visible=False, value=f"{elem1}:{i}"
                                    )

                                    alias = gr.Textbox(
                                        show_label=False,
                                        value=(
                                            self.aliases[elem1][i]
                                            if i < len(self.aliases[elem1])
                                            else "alias"
                                        ),
                                        interactive=True,
                                        key=key.value,
                                    )

                                    alias.blur(
                                        self.update_captions,
                                        inputs=[alias, key],
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
                                        gr.Button(value=elem2, interactive=False)

                                        s = gr.Slider(
                                            1,
                                            maximum=self.max_captions,
                                            value=len(self.aliases[elem2]),
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
                                        key = gr.Textbox(
                                            visible=False, value=f"{elem2}:{i}"
                                        )

                                        alias = gr.Textbox(
                                            show_label=False,
                                            value=(
                                                self.aliases[elem2][i]
                                                if i < len(self.aliases[elem2])
                                                else "alias"
                                            ),
                                            interactive=True,
                                            key=key.value,
                                        )

                                        alias.blur(
                                            self.update_captions,
                                            inputs=[alias, key],
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
                # FIX: file explorer not working
                image = gr.FileExplorer(
                    label="Select Image",
                    glob="*/*.jpg;*/*.jpeg;*/*.png",
                    root_dir="/Users/francescotacinelli/Developer/datasets/pallets_sorted/vertical/",
                    file_count="single",
                )

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
    # app.demo.launch(share=True, allowed_paths=["~/dataset/pallets_sorted/"])
    app.demo.launch()
    atexit.register(app.exit_handler)


if __name__ == "__main__":
    main()
