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
        self.aliasesAPs = []

    def guess(self, alias, aliasAP, description):
        json_object = {
            "alias": "pouch",
            "reasoning": "To improve the Average Precision from 0.144, we need to consider aliases that are distinct enough from bitter pack to potentially yield better detection rates. Here are some suggestions for new aliases, considering phonetic similarity, visual similarity, and contextual usage: Bitter Pouch: This alias maintains the core concept of bitter and changes the container type to pouch which might be less common ...",
        }

        prev_alias = self.aliases[-1] if self.aliases else "bitter"
        prev_AP = self.aliasesAPs[-1] if self.aliasesAPs else 0.086

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
        self.aliasesAPs.append(aliasAP)

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        generation_args = {
            "max_new_tokens": 200,
            "return_full_text": False,
            # "temperature": 0.0,
            "do_sample": False,
        }

        return pipe(messages, **generation_args)

    def combine(self, aliases: List[str], aliasesAPs: List[float]):
        """
        Find the best combination of aliases to maximize the Average Precision.
        """
        # TODO: Implement the function
        pass


def main():
    import logging

    logger = logging.getLogger("transformers.modeling_utils")
    logger.setLevel(logging.ERROR)  # Set the logger level to ERROR to suppress warnings
    utils.filter_phi_warnings()

    guesser = Guesser()
    description = "A pack of bitters (small glass bottles with screw caps in a cardboard box or plastic wrap)"

    # make guess until the user is satisfied
    prev_alias = ""
    while True:
        # let the user write the alias and the AP from the cli
        input_alias = str(input("Enter the alias: "))
        input_aliasAP = float(input("Enter the alias AP: "))
        answer = guesser.guess(input_alias, input_aliasAP, description=description)
        print("answer:", answer, "\n")
        satisfied = input("Are you satisfied? (y/n): ")
        if satisfied == "y":
            break


if __name__ == "__main__":
    main()
