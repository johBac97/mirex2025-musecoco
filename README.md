# MuseCoco for Piano Music Continuation

## Installation

Install with a Python 3.11 environment. the `uv` command to create such an environment is

```
uv venv env --python 3.11
source env/bin/activate
python -m ensurepip
python -m pip install torch==2.4.1 wheel
python -m pip install --no-build-isolation -r requirements.txt
```

`pytorch-fast-transformers` requires torch for the build step which is the reason torch is installed separetly. The `--no-build-isolation` flag allows pip to use the environment to build the any packages that require building (`pytorch-fast-transformers`).


## Generation
*A baseline method for the Symbolic Music Generation task @ MIREX2025*

1. Download the checkpoint of MuseCoco at [this link](https://drive.google.com/file/d/1wTG4FhWocsJmDfncAp6j2lUBzPfHW90G/view?usp=sharing). Configure the path in `generate_main.py`. Install the required packages.
2. To convert a 4-measure prompt in MIDI format to the required JSON format, put MIDI files in `samples/raw_midi_upload`, and run
   ```
   python prompt_midi_to_json.py samples/raw_midi_upload samples/json_prompts
   ```
   JSON prompts will be generated and stored at `json_prompts`.
3. To generate piano music continuation, use the command `.generate.sh <input.json> <output_folder> <n_sample>`. For example:
   ```
   ./generate.sh samples/json_prompts/song_wo_words_prompt.json samples/json_outputs/song_wo_words 4
   ```
   4 continuation samples will be stored at `samples/json_outputs`. **IMPORTANT: Please follow the format of this command in the submission.**
4. To listen to the generated samples, run:
   ```
   python generated_json_to_midi.py samples/json_prompts/song_wo_words_prompt.json samples/json_outputs/song_wo_words samples/json_outputs/song_wo_words samples/midi_outputs/song_wo_words
   ```
   The prompt and 4 MIDI samples will be generated in the target folder.

$^*$ Attributes in MuseCoco are set to support 16-bar piano music generation as required by the MIREX challenge. For simplicity, the pickup measure is not utilized in the generation process. Thanks to [Longshen](https://www.oulongshen.xyz/), the original Fairseq-based code has been reorganized into a HuggingFace-like structure for easier integration. The original repo can be found at [this link](https://github.com/microsoft/muzic/tree/main/musecoco).

## Calculate NLL Of Generation

```
python calculate_cumulative_nll.py <PATH-TO-MODEL-ROOT> <PATH-TO-GENERATION-JSON> <PATH-TO-PROMPT-JSON>
```

Where `<PATH-TO-MODEL-ROOT>` is the path to the root of the checkpoint folder containing both the model and the tokenizer as subfolders, (`.../model_and_tokenizer/200m/`). `<PATH-TO-GENERATION-JSON>` is the path to the generated output from the model in json format. Finaly, the `<PATH-TO-PROMPT-JSON>` is the path to the prompt that was used to generate the generation.
