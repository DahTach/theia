import gradio as gr
import torch
from itertools import zip_longest

# from gradio_counter import Counter as grCounter
from model import models, Model
from caption import captions

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

    def load_model(self, model_name):
        self.model = Model(model_name, self.get_captions())

    def predict(self, image):
        results = self.model.predict(image)
        image = self.model.show(image, results)
        return image

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

            # with gr.Row():
            #     n_classes = gr.Slider(
            #         1,
            #         maximum=self.max_captions,
            #         value=len(captions.keys()),
            #         step=1,
            #         visible=False,
            #     )
            #
            #     cls_name = gr.Textbox(
            #         placeholder="New Class", visible=True, interactive=True
            #     )
            #
            #     def add_cl(n, cls):
            #         captions.setdefault(cls, [])
            #         return n + 1
            #
            #     add_cls = gr.Button("Add Class", size="sm")
            #     add_cls.click(add_cl, inputs=[n_classes, cls_name], outputs=n_classes)
            #
            #     def rm_cl(n, cls):
            #         captions.pop(cls)
            #         return n - 1
            #
            #     del_cls = gr.Button("Remove Class", size="sm")
            #     del_cls.click(rm_cl, inputs=[n_classes, cls_name], outputs=n_classes)
            #
            #     def rerender():
            #         labels.render()
            #
            #     n_classes.change(rerender)

            with gr.Row():
                model_name = gr.Dropdown(
                    label="Model", choices=[model for model in models.keys()]
                )

                model_name.change(self.load_model, inputs=model_name)

                predict = gr.Button(value="Predict")

            with gr.Row():
                image = gr.Image(label="Image")

            with gr.Row():
                results = gr.Image()

            predict.click(self.predict, inputs=[image], outputs=[results])

        self.demo = demo


def main():
    app = GradioApp()
    app.interface()
    app.demo.launch()


if __name__ == "__main__":
    main()
