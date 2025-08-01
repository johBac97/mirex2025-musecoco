import sys
import torch
import json
import argparse

from pathlib import Path
from music_json_convert import json_dict_to_note_matrix, note_matrix_to_midi
from utils_midi.utils_midi import RemiTokenizer

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
    pass
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=Path,
        help="The model root folder which contains both the model and the tokenizer as subfolders.",
    )

    parser.add_argument("generation", help="Generation JSON file.", type=Path)

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

    # Load the json formatted generation of the model
    with args.generation.open() as io:
        generation = json.load(io)

    # Convert the generation from json start/end/duration format into a note matrix (numpy)
    generation_notes = json_dict_to_note_matrix(generation, "generation")

    TEMP_MIDI_PATH = Path.cwd() / "temp.midi"

    # Convert the note matrix into a MIDI file
    note_matrix_to_midi(generation_notes, str(TEMP_MIDI_PATH))

    # Tokenize the MIDI file using the REMI Tokenizer
    remi_tokens = remi_tokenizer.midi_to_remi(
        str(TEMP_MIDI_PATH), False, include_tempo=True, include_velocity=True
    )

    # Construct the input string from the ATTRIBUTE_PROMPT (settings for the generation) and remi_tokens
    input_str = " ".join(ATTRIBUTE_PROMPT + ["<sep>"] + remi_tokens)
    input_ids = tokenizer(input_str, return_tensors="pt")["input_ids"].cuda()

    # Why is this done? Prepending the final token? Shouldn't this be handled by the tokenizer?
    input_ids = torch.cat([input_ids[:, -1:], input_ids[:, :-1]], dim=1)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

        # Averaged over all tokens
        ce_loss = outputs.loss

    nll_loss = ce_loss * input_ids.shape[-1]

    print(f"Cumulative Negative Likelihood:\t{nll_loss}")


if __name__ == "__main__":
    main()
