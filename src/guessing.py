import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict
import utils

# Set random seed for reproducibility
torch.random.manual_seed(0)


class Guesser:
    def __init__(self) -> None:
        self.device = utils.get_device()
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct",
            device_map=self.device,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct"
        )
        self.preprompt = "Given this benchmark, find the aliases that can improve the performance of the model."
        self.aliases = []

    def guess(self, alias, aliasAP, prev_alias, prev_AP, description):
        # TODO: force model to answer with a json object
        """
        Guess_for_alias_bitter_pack = [
            {
                "generated_text": ' To improve the Average Precision from 0.144, we need to consider aliases that are distinct enough from "bitter pack" to potentially yield better detection rates. Here are some suggestions for new aliases, considering phonetic similarity, visual similarity, and contextual usage:\n\n1. Bitter Pouch: This alias maintains the core concept of "bitter" and changes the container type to "pouch," which might be less common and thus'
            }
        ]
        Guess_for_alias_bitter_crate = [
            {
                "generated_text": ' Considering the goal is to improve the Average Precision, a more descriptive and unique alias that could potentially be more distinctive and less likely to be confused with other items could be beneficial. Here are some suggestions:\n\n1. "VintageGlassBottle" - This alias emphasizes the vintage aspect and the material of the packaging, which might help in distinguishing it from other items.\n\n2. "ScratchGlass'
            }
        ]
        """
        # TODO: define the json object
        json_object = {}
        messages = [
            {
                "role": "user",
                "content": f"The aim is to detect {description} detecting its altrady trained aliases. Average Precision for the alias {prev_alias} = {prev_AP}. Find a better alias to improve the Average Precision. Aliases already tried: {self.aliases}",
            },
            {"role": "assistant", "content": f"{json_object}"},
            {
                "role": "user",
                "content": f"The aim is to detect {description} detecting its altrady trained aliases. Average Precision for the alias {alias} = {aliasAP}. Find a better alias to improve the Average Precision. Aliases already tried: {self.aliases}",
            },
        ]

        self.aliases.append(alias)

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        generation_args = {
            "max_new_tokens": 100,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        return pipe(messages, **generation_args)

    def combine(self, aliases: List[str], aliasesAPs: List[float]):
        """
        Find the best combination of aliases to maximize the Average Precision.
        """
        # TODO: Implement the function
        pass
