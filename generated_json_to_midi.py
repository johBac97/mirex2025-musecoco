import sys
import os
import re
from music_json_convert import read_json_dict, json_dict_to_note_matrix, note_matrix_to_midi, json_prompt_to_midi
import numpy as np


def merge_json_to_midi(prompt_json_fn, generation_json_fn, output_midi_fn, bpm=90, vel=80):
    prompt_json_dict = read_json_dict(prompt_json_fn)
    generation_json_dict = read_json_dict(generation_json_fn)

    prompt_note_matrix = json_dict_to_note_matrix(prompt_json_dict, 'prompt')
    generation_note_matrix = json_dict_to_note_matrix(generation_json_dict, 'generation')

    # Combine the two note matrices
    combined_note_matrix = np.concatenate([prompt_note_matrix, generation_note_matrix], axis=0)

    note_matrix_to_midi(combined_note_matrix, output_midi_fn, bpm=bpm, vel=vel)


def batch_merge_json_to_midi(prompt_json_fn, generated_json_folder, output_midi_folder, bpm=90, vel=80):
    os.makedirs(output_midi_folder, exist_ok=True)

    # Convert the prompt_json to midi
    json_prompt_to_midi(prompt_json_fn, os.path.join(output_midi_folder, 'prompt.mid'), bpm=bpm, vel=vel)

    # Match files named sample_XX.json
    pattern = re.compile(r'^sample_(\d+)\.json$')

    for filename in sorted(os.listdir(generated_json_folder)):
        match = pattern.match(filename)
        if match:
            sample_id = match.group(1)
            generation_json_fn = os.path.join(generated_json_folder, filename)
            output_midi_fn = os.path.join(output_midi_folder, f'sample_{sample_id}.mid')
            
            merge_json_to_midi(
                prompt_json_fn=prompt_json_fn,
                generation_json_fn=generation_json_fn,
                output_midi_fn=output_midi_fn,
                bpm=bpm,
                vel=vel
            )
            print(f"Converted {filename} → sample_{sample_id}.mid")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_json_folder_to_midi.py <prompt_json_fn> <generated_json_folder> <output_midi_folder>")
        sys.exit(1)

    prompt_json_fn = sys.argv[1]
    generated_json_folder = sys.argv[2]
    output_midi_folder = sys.argv[3]

    batch_merge_json_to_midi(prompt_json_fn, generated_json_folder, output_midi_folder)
