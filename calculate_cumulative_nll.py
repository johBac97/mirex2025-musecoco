import sys
import torch
import argparse

from pathlib import Path
from utils_midi.utils_midi import RemiTokenizer
from music_json_convert import json_prompt_to_midi

sys.path.insert(0, str(Path("./transformers/src").resolve()))
from transformers import MuseCocoConfig, MuseCocoTokenizer, MuseCocoLMHeadModel


ATTRIBUTE_PROMPT = [
    "I1s2_0_0",
    "I1s2_1_2",
    "I1s2_2_2",
    "I1s2_3_2",
    "I1s2_4_2",
    "I1s2_5_2",
    "I1s2_6_2",
    "I1s2_7_2",
    "I1s2_8_2",
    "I1s2_9_2",
    "I1s2_10_2",
    "I1s2_11_2",
    "I1s2_12_2",
    "I1s2_13_2",
    "I1s2_14_2",
    "I1s2_15_2",
    "I1s2_16_2",
    "I1s2_17_2",
    "I1s2_18_2",
    "I1s2_19_2",
    "I1s2_20_2",
    "I1s2_21_2",
    "I1s2_22_2",
    "I1s2_23_2",
    "I1s2_24_2",
    "I1s2_25_2",
    "I1s2_26_2",
    "I1s2_27_2",
    "I4_28",
    "C1_4",
    "R1_2",
    "R3_3",
    "S2s1_17",
    "S4_0_2",
    "S4_1_2",
    "S4_2_2",
    "S4_3_2",
    "S4_4_2",
    "S4_5_2",
    "S4_6_2",
    "S4_7_2",
    "S4_8_2",
    "S4_9_2",
    "S4_10_2",
    "S4_11_2",
    "S4_12_2",
    "S4_13_2",
    "S4_14_2",
    "S4_15_2",
    "S4_16_2",
    "S4_17_2",
    "S4_18_2",
    "S4_19_2",
    "S4_20_2",
    "S4_21_2",
    "B1s1_3",
    "TS1s1_0",
    "K1_2",
    "T1s1_1",
    "P4_12",
    "ST1_14",
    "EM1_4",
    "TM1_2",
]


def __parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=Path,
        help="The model root folder which contains both the model and the tokenizer as subfolders.",
    )

    parser.add_argument("generation", help="Generation JSON file.", type=Path)

    parser.add_argument("prompt", help="Prompt JSON file.", type=Path)
    return parser.parse_args()


def main():
    args = __parse_args()

    config_fp = str(args.model / "model" / "config.json")
    config = MuseCocoConfig.from_pretrained(config_fp)
    model = (
        MuseCocoLMHeadModel.from_pretrained(str(args.model / "model"), config=config)
        .cuda()
        .eval()
    )
    tokenizer = MuseCocoTokenizer.from_pretrained(str(args.model / "tokenizer"))

    remi_tokenizer = RemiTokenizer()

    TEMP_MIDI_FOLDER = Path.cwd() / "temp_midi"
    TEMP_MIDI_FOLDER.mkdir(exist_ok=True)

    # Load the generation, transform into note matrix and then serialize to MIDI file.
    generation_midi_path = TEMP_MIDI_FOLDER / "generation.midi"
    json_prompt_to_midi(
        args.generation, str(generation_midi_path), keyword="generation"
    )

    # Tokenizer the generation MIDI file using the REMI tokenizer
    remi_tokens_generation = remi_tokenizer.midi_to_remi(
        str(generation_midi_path), False, include_tempo=True, include_velocity=True
    )

    # Load the prompt, transform into note matrix and then serialize to MIDI file.
    prompt_midi_path = TEMP_MIDI_FOLDER / "prompt.midi"
    json_prompt_to_midi(args.prompt, str(prompt_midi_path))

    remi_tokens_prompt = remi_tokenizer.midi_to_remi(
        str(prompt_midi_path), False, include_tempo=True, include_velocity=True
    )

    # Construct the input string from the ATTRIBUTE_PROMPT (settings for the generation)
    # the prompt that was used to create the generation, then finaly the generation itself.
    input_str = " ".join(
        ATTRIBUTE_PROMPT + ["<sep>"] + remi_tokens_prompt + remi_tokens_generation
    )
    input_ids = tokenizer(input_str, return_tensors="pt")["input_ids"].cuda()

    # Why is this done? Prepending the final token? Shouldn't this be handled by the tokenizer?
    input_ids = torch.cat([input_ids[:, -1:], input_ids[:, :-1]], dim=1)

    prompt_token_length = len(ATTRIBUTE_PROMPT + ["<sep>"] + remi_tokens_prompt) + 1

    labels = input_ids.clone()
    # Mask the prompt tokens
    labels[:, :prompt_token_length] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    avg_nll = outputs.loss

    number_generation_tokens = input_ids.size(1) - prompt_token_length
    cumulative_nll = avg_nll * number_generation_tokens

    print(f"Cumulative Negative Likelihood:\t{cumulative_nll}")


if __name__ == "__main__":
    main()
