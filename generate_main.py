import sys
import os
import uuid
import numpy as np
sys.path.insert(0, os.path.abspath('./transformers/src'))
import torch
from transformers import MuseCocoLMHeadModel, MuseCocoConfig, MuseCocoTokenizer
from utils import jpath, read_json
from utils_midi.utils_midi import RemiTokenizer
from music_json_convert import json_prompt_to_midi, read_json_dict, write_json_dict, midi_to_note_matrix, note_matrix_to_json_dict, \
    json_dict_to_note_matrix, note_matrix_to_midi
import shutil


ATTRIBUTE_PROMPT = \
    ['I1s2_0_0', 'I1s2_1_2', 'I1s2_2_2', 'I1s2_3_2', 'I1s2_4_2', 'I1s2_5_2', 'I1s2_6_2', 'I1s2_7_2', 'I1s2_8_2', 
    'I1s2_9_2', 'I1s2_10_2', 'I1s2_11_2', 'I1s2_12_2', 'I1s2_13_2', 'I1s2_14_2', 'I1s2_15_2', 'I1s2_16_2', 'I1s2_17_2', 
    'I1s2_18_2', 'I1s2_19_2', 'I1s2_20_2', 'I1s2_21_2', 'I1s2_22_2', 'I1s2_23_2', 'I1s2_24_2', 'I1s2_25_2', 'I1s2_26_2', 
    'I1s2_27_2', 
    'I4_28', 'C1_4', 
    'R1_2', 'R3_3', 'S2s1_17', 
    'S4_0_2', 'S4_1_2', 'S4_2_2', 'S4_3_2', 'S4_4_2', 'S4_5_2', 
    'S4_6_2', 'S4_7_2', 'S4_8_2', 'S4_9_2', 'S4_10_2', 'S4_11_2', 'S4_12_2', 'S4_13_2', 'S4_14_2', 'S4_15_2', 'S4_16_2', 
    'S4_17_2', 'S4_18_2', 'S4_19_2', 'S4_20_2', 'S4_21_2', 
    'B1s1_3', 'TS1s1_0', 'K1_2', 'T1s1_1', 'P4_12', 
    'ST1_14', 'EM1_4', 
    'TM1_2']

MODEL_CKPT = '../model_and_tokenizer/1b/model'


def truncate_pickup_measure(json_dict, key="prompt", truncate_at=16):
    original_notes = json_dict[key]
    truncated_notes = []

    for note in original_notes:
        new_start = note["start"] - truncate_at
        if new_start >= 0:
            truncated_notes.append({
                "start": new_start,
                "pitch": note["pitch"],
                "duration": note["duration"]
            })
    return {key: truncated_notes}


def add_pickup_measure_back(json_dict, key="generation", shift_by=16):
    original_notes = json_dict[key]
    shifted_notes = []

    for note in original_notes:
        shifted_notes.append({
            "start": note["start"] + shift_by,
            "pitch": note["pitch"],
            "duration": note["duration"]
        })

    return {key: shifted_notes}

def generate_sample(prompt_json_fn, output_json_fn):
    config_fp = jpath(MODEL_CKPT, 'config.json')
    config = MuseCocoConfig.from_pretrained(config_fp)
    model = MuseCocoLMHeadModel.from_pretrained(MODEL_CKPT, config=config).cuda()
    tk_fp = '../model_and_tokenizer/1b/tokenizer'
    tk = MuseCocoTokenizer.from_pretrained(tk_fp)
    print("Model and tokenizer loaded.")

    unique_id = str(uuid.uuid4())
    temp_midi_folder = f"temp_midi_{unique_id}"
    os.makedirs(temp_midi_folder, exist_ok=True)

    # load json and store temporarily as midi file
    json_prompt_dict = read_json_dict(prompt_json_fn)
    json_prompt_dict = truncate_pickup_measure(json_prompt_dict, key='prompt', truncate_at=16)
    note_matrix = json_dict_to_note_matrix(json_prompt_dict, 'prompt')
    temp_midi_path = f'{temp_midi_folder}/temp_prompt.mid'
    note_matrix_to_midi(note_matrix, temp_midi_path)

    # load midi file as remi tokens
    remi_tokenizer = RemiTokenizer()
    prompt_remi_tokens = remi_tokenizer.midi_to_remi(temp_midi_path, False, include_tempo=True, include_velocity=True)

    # create musecoco input string
    input_str = ATTRIBUTE_PROMPT + ['<sep>'] + prompt_remi_tokens
    input_length = len(input_str)
    input_str = ' '.join(input_str)
    input_ids = tk(input_str, return_tensors='pt')['input_ids'].cuda()
    input_ids = torch.cat([input_ids[:,-1:], input_ids[:,:-1]], dim=1)

    # generate
    generate_kwargs = {'max_length': 8000,
                   'use_cache': False,
                   'do_sample':True,
                   'top_k': 15,
                   'temperature': 1.0
                   }
    output_ids = model.generate(input_ids, pad_token_id=tk.pad_token_id, **generate_kwargs)

    # convert output to remi tokens
    output_str = tk.batch_decode(output_ids)
    output_str = output_str[0].split(' ')
    output_str = [token for token in output_str if '-' in token]
    
    # save output as midi file
    temp_output_path = f'{temp_midi_folder}/temp_output.mid'
    remi_tokenizer.remi_to_midi(output_str, temp_output_path, ignore_velocity=True)
    
    # load midi file and save as json
    note_matrix = midi_to_note_matrix(temp_output_path)
    note_matrix = note_matrix[np.logical_and(note_matrix[:, 0] >= 4 * 4, note_matrix[:, 0] < 64)]  # remove the first five measures
    generation_json_dict = note_matrix_to_json_dict(note_matrix, 'generation')
    generation_json_dict = add_pickup_measure_back(generation_json_dict, key='generation', shift_by=16)
    write_json_dict(generation_json_dict, output_json_fn)

    if os.path.exists(temp_midi_folder):
        shutil.rmtree(temp_midi_folder)

def batch_generate(prompt_json_fn, output_folder, n_sample):
    os.makedirs(output_folder, exist_ok=True)
    for i in range(1, n_sample + 1):
        output_json_fn = os.path.join(output_folder, f"sample_{i:02d}.json")
        generate_sample(prompt_json_fn, output_json_fn)
        print(f"File generated:", output_json_fn)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_batch.py <input_json> <output_folder> <n_sample>")
        sys.exit(1)
    prompt_json_fn = sys.argv[1]
    output_folder = sys.argv[2]
    n_sample = int(sys.argv[3])
    batch_generate(prompt_json_fn, output_folder, n_sample)










# prompt_json_fn = 'test_musecoco_midi_demo/outputs/recon_from_nm.json'
# generation_json_fn = 'test_musecoco_midi_demo/outputs/generation_from_nm.json'

# print(len(ATTRIBUTE_PROMPT))
# main(prompt_json_fn, generation_json_fn)


# MODEL_CKPT = '../model_and_tokenizer/1b/model'
# config_fp = jpath(MODEL_CKPT, 'config.json')
# config = MuseCocoConfig.from_pretrained(config_fp)
# model = MuseCocoLMHeadModel.from_pretrained(pt_ckpt, config=config).cuda()
# tk_fp = '../model_and_tokenizer/1b/tokenizer'
# tk = MuseCocoTokenizer.from_pretrained(tk_fp)




# acc_midi_fn = 'test_musecoco_midi_demo/demo-song-wo-words.mid'
# remi_tokenizer = RemiTokenizer()
# acc_remi_tokens = remi_tokenizer.midi_to_remi(acc_midi_fn, False, include_tempo=True, include_velocity=True)
# print('fefefefefefefe')
# print(acc_remi_tokens)


# input_str = attibute_prompt + ['<sep>'] + acc_remi_tokens
# input_str = ' '.join(input_str)
# input_ids = tk(input_str, return_tensors='pt')['input_ids'].cuda()
# input_ids = torch.cat([input_ids[:,-1:], input_ids[:,:-1]], dim=1)
# print(input_ids)

# generate_kwargs = {'max_length': 8000, # 2000
#                    'use_cache': False,
#                    'do_sample':True,
#                    'top_k': 15,  # 3,
#                    'temperature': 1.0 # 1.2
#                    }

# output = model.generate(input_ids, pad_token_id=tk.pad_token_id, **generate_kwargs)
# print(output)
# print('---------------------------')
# out_str = tk.batch_decode(output)
# output = out_str[0].split(' ')  # ['</s>', 's-9', 'o-0', 'i-0', ..., '<sep>', ..., '</s>']
            
# print(output)
# print('-xxxxxxxx')

# # out_str =tk.batch_decode(output)
# # output = out_str[0].split(' ')
# output = [token for token in output if '-' in token]
# print(output)
# remi_tokenizer.remi_to_midi(output, 'test_musecoco_midi_demo/outputs/song-wo-word-2.mid', ignore_velocity=True)
