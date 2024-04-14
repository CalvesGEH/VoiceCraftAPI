# Generic imports
import sys
sys.path.append("./VoiceCraft") # Append VoiceCraft folder to fix python import
import os
import logging as logger

# VoiceCraft imports
import torch
import torchaudio
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
from models import voicecraft
import io
import numpy as np
import random
import uuid

# API imports
from fastapi import FastAPI, File, UploadFile, Form
from starlette.responses import StreamingResponse
import subprocess
import json
import shutil

# Configure logging
logger.basicConfig(level=logger.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logger.FileHandler('api.log'),
                        logger.StreamHandler()
                    ])

############################
#       API Variables      #
############################

app = FastAPI()

VOICES_PATH = os.getenv("VOICES_PATH", "./voices")

############################
#   VoiceCraft Variables   #
############################

DEMO_PATH = os.getenv("DEMO_PATH", "./VoiceCraft/demo")
TMP_PATH = os.getenv("TMP_PATH", "./VoiceCraft/demo/temp")
MODELS_PATH = os.getenv("MODELS_PATH", "./VoiceCraft/pretrained_models")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model, align_model, voicecraft_model = None, None, None

############################
#   VoiceCraft Functions   #
############################

def get_random_string():
    return "".join(str(uuid.uuid4()).split("-"))


def seed_everything(seed):
    if seed != -1:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class WhisperxAlignModel:
    def __init__(self):
        from whisperx import load_align_model
        self.model, self.metadata = load_align_model(language_code="en", device=device)

    def align(self, segments, audio_path):
        from whisperx import align, load_audio
        audio = load_audio(audio_path)
        return align(segments, self.model, self.metadata, audio, device, return_char_alignments=False)["segments"]


class WhisperModel:
    def __init__(self, model_name):
        from whisper import load_model
        self.model = load_model(model_name, device)

        from whisper.tokenizer import get_tokenizer
        tokenizer = get_tokenizer(multilingual=False)
        self.supress_tokens = [-1] + [
            i
            for i in range(tokenizer.eot)
            if all(c in "0123456789" for c in tokenizer.decode([i]).removeprefix(" "))
        ]

    def transcribe(self, audio_path):
        return self.model.transcribe(audio_path, suppress_tokens=self.supress_tokens, word_timestamps=True)["segments"]


class WhisperxModel:
    def __init__(self, model_name, align_model: WhisperxAlignModel):
        from whisperx import load_model
        possible_compute_types = ["float16", "float32", "int8"]
        for type in possible_compute_types:
            try:
                logger.debug(f"Trying to create a whisperx model using {type}.")
                self.model = load_model(model_name, device, compute_type=type, asr_options={"suppress_numerals": True, "max_new_tokens": None, "clip_timestamps": None, "hallucination_silence_threshold": None})
                logger.debug(f"Created whisperx model using {type}.")
                break
            except ValueError as err:
                logger.warning(f"Caught Exception while creating WhisperxModel: {err.args[0]}")
                if type == possible_compute_types[-1]:
                    # If we went through all types, forward error.
                    raise ValueError(err.args[0])
        self.align_model = align_model

    def transcribe(self, audio_path):
        segments = self.model.transcribe(audio_path, batch_size=8)["segments"]
        return self.align_model.align(segments, audio_path)

# Parameters:
#     whisper_backend_name: 'whisper' or 'whisperX'. 'whisperX' is default.
#     whisper_mode_name: 'None', 'base.en', 'small.en', 'medium.en', 'large'. 'base.en' is default.
#     alignment_mode_name: 'None'or 'whisperX'. 'whisperX' is default.
#     voicecraft_model:
#         - 'giga330M'
#         - 'giga830M' (default)
#         - 'giga330M_TTSEnhanced'
def load_models(whisper_backend_name, whisper_model_name, alignment_model_name, voicecraft_model_name):
    global transcribe_model, align_model, voicecraft_model

    if voicecraft_model_name == "giga330M_TTSEnhanced":
        voicecraft_model_name = "gigaHalfLibri330M_TTSEnhanced_max16s"

    if alignment_model_name is not None:
        align_model = WhisperxAlignModel()

    if whisper_model_name is not None:
        if whisper_backend_name == "whisper":
            transcribe_model = WhisperModel(whisper_model_name)
        else:
            if align_model is None:
                logger.error("Align model required for whisperx backend.")
                return False
            transcribe_model = WhisperxModel(whisper_model_name, align_model)

    voicecraft_name = f"{voicecraft_model_name}.pth"
    ckpt_fn = f"{MODELS_PATH}/{voicecraft_name}"
    encodec_fn = f"{MODELS_PATH}/encodec_4cb2048_giga.th"
    if not os.path.exists(ckpt_fn):
        logger.debug(f"Downloading {voicecraft_name} to {MODELS_PATH}.")
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/{voicecraft_name}\?download\=true")
        os.system(f"mv {voicecraft_name}\?download\=true {MODELS_PATH}/{voicecraft_name}")
    if not os.path.exists(encodec_fn):
        logger.debug(f"Downloading encodec_4cb2048_giga.th to {MODELS_PATH}.")
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
        os.system(f"mv encodec_4cb2048_giga.th {MODELS_PATH}/encodec_4cb2048_giga.th")

    ckpt = torch.load(ckpt_fn, map_location="cpu")
    model = voicecraft.VoiceCraft(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    voicecraft_model = {
        "ckpt": ckpt,
        "model": model,
        "text_tokenizer": TextTokenizer(backend="espeak"),
        "audio_tokenizer": AudioTokenizer(signature=encodec_fn)
    }
    logger.info("Succesfully loaded models.")
    return True


def get_transcribe_state(segments):
    words_info = [word_info for segment in segments for word_info in segment["words"]]
    return {
        "segments": segments,
        "transcript": " ".join([segment["text"] for segment in segments]),
        "words_info": words_info,
        "transcript_with_start_time": " ".join([f"{word['start']} {word['word']}" for word in words_info]),
        "transcript_with_end_time": " ".join([f"{word['word']} {word['end']}" for word in words_info]),
        "word_bounds": [f"{word['start']} {word['word']} {word['end']}" for word in words_info]
    }


def transcribe(seed, audio_path):
    if transcribe_model is None:
        logger.error("Transcription model not loaded. Please load models first.")
        return False
    seed_everything(seed)

    segments = transcribe_model.transcribe(audio_path)
    state = get_transcribe_state(segments)

    return state


def align_segments(transcript, audio_path, voice):
    from aeneas.executetask import ExecuteTask
    from aeneas.task import Task
    import json
    logger.debug("Aligning segments...")
    config_string = 'task_language=eng|os_task_file_format=json|is_text_type=plain'

    transcript_path = os.path.join(VOICES_PATH, f"{voice}.txt")
    sync_map_path = os.path.join(VOICES_PATH, f"{voice}.json")
    with open(transcript_path, "w") as f:
        f.write(transcript)

    task = Task(config_string=config_string)
    task.audio_file_path_absolute = os.path.abspath(audio_path)
    task.text_file_path_absolute = os.path.abspath(transcript_path)
    task.sync_map_file_path_absolute = os.path.abspath(sync_map_path)
    ExecuteTask(task).execute()
    task.output_sync_map_file()

    logger.debug("Finished aligning segments. Returning sync_map as json.")
    with open(sync_map_path, "r") as f:
        return json.load(f)


def align(seed, transcript, audio_path, voice):
    if align_model is None:
        logger.error("Align model not loaded")
        return False
    seed_everything(seed)

    fragments = align_segments(transcript, audio_path, voice)
    segments = [{
        "start": float(fragment["begin"]),
        "end": float(fragment["end"]),
        "text": " ".join(fragment["lines"])
    } for fragment in fragments["fragments"]]
    segments = align_model.align(segments, audio_path)
    state = get_transcribe_state(segments)

    return state


def get_output_audio(audio_tensors, codec_audio_sr):
    result = torch.cat(audio_tensors, 1)
    buffer = io.BytesIO()
    torchaudio.save(buffer, result, int(codec_audio_sr), format="wav")
    buffer.seek(0)
    return buffer.read()

# Parameters:
#     seed: Random seed
#     top_k:
#     top_p
#     temperature
#     stop_repetition
#     sample_batch_size
#     kvcache
#     audio_path
#     transcribe_state
#     transcript
#     smart_transcript: Whether or not to create the target transcript automatically. If disabled, you need to supply the transcript.
#     mode: The generation mode. Options: 'TTS', 'Long TTS'.
#         'Long TTS' will run the model on each and every newline because the model is not equipped to handle a large amount of text at once.
#     prompt_end_time: End time of the last message of the transcript. Float.
#     split_text: How to split the transcript. Only applies to 'Long TTS' mode. Options: 'Newline', 'Sentence'.
def generate(seed, top_k, top_p, temperature, stop_repetition, sample_batch_size, 
        kvcache, audio_path, transcribe_state, transcript, smart_transcript,
        mode, prompt_end_time,
        codec_audio_sr=16000, codec_sr=50, silence_tokens=[1388, 1898, 131] # Codec specific options
        ):
    if voicecraft_model is None:
        logger.error("VoiceCraft model not loaded.")
        return False
    if smart_transcript and (transcribe_state is None):
        logger.error("Can't use smart transcript: whisper transcript not found.")
        return False

    seed_everything(seed)
    if mode == "Long TTS":
        sentences = transcript.split('\n')
    else:
        sentences = [transcript.replace("\n", " ")]

    info = torchaudio.info(audio_path)
    audio_dur = info.num_frames / info.sample_rate

    audio_tensors = []
    inference_transcript = ""
    for sentence in sentences:
        decode_config = {"top_k": top_k, "top_p": top_p, "temperature": temperature, "stop_repetition": stop_repetition,
                         "kvcache": kvcache, "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr,
                         "silence_tokens": silence_tokens, "sample_batch_size": sample_batch_size}
        if mode == "TTS" or mode == "Long TTS":
            from inference_tts_scale import inference_one_sample

            if smart_transcript:
                target_transcript = ""
                for word in transcribe_state["words_info"]:
                    if word["end"] < prompt_end_time:
                        target_transcript += word["word"] + (" " if word["word"][-1] != " " else "")
                    elif (word["start"] + word["end"]) / 2 < prompt_end_time:
                        # include part of the word it it's big, but adjust prompt_end_time
                        target_transcript += word["word"] + (" " if word["word"][-1] != " " else "")
                        prompt_end_time = word["end"]
                        break
                    else:
                        break
                target_transcript += f" {sentence}"
            else:
                target_transcript = sentence

            inference_transcript += target_transcript + "\n"

            prompt_end_frame = int(min(audio_dur, prompt_end_time) * info.sample_rate)
            _, gen_audio = inference_one_sample(voicecraft_model["model"],
                                                voicecraft_model["ckpt"]["config"],
                                                voicecraft_model["ckpt"]["phn2num"],
                                                voicecraft_model["text_tokenizer"], voicecraft_model["audio_tokenizer"],
                                                audio_path, target_transcript, device, decode_config,
                                                prompt_end_frame)

        gen_audio = gen_audio[0].cpu()
        audio_tensors.append(gen_audio)

    output_audio = get_output_audio(audio_tensors, codec_audio_sr)
    sentences = [f"{idx}: {text}" for idx, text in enumerate(sentences)]
    return output_audio, inference_transcript, audio_tensors


def load_sentence(selected_sentence, codec_audio_sr, audio_tensors):
    if selected_sentence is None:
        return None
    colon_position = selected_sentence.find(':')
    selected_sentence_idx = int(selected_sentence[:colon_position])
    return get_output_audio([audio_tensors[selected_sentence_idx]], codec_audio_sr)


def update_bound_word(is_first_word, selected_word, edit_word_mode):
    if selected_word is None:
        return None

    word_start_time = float(selected_word.split(' ')[0])
    word_end_time = float(selected_word.split(' ')[-1])
    if edit_word_mode == "Replace half":
        bound_time = (word_start_time + word_end_time) / 2
    elif is_first_word:
        bound_time = word_start_time
    else:
        bound_time = word_end_time

    return bound_time


def update_bound_words(from_selected_word, to_selected_word, edit_word_mode):
    return [
        update_bound_word(True, from_selected_word, edit_word_mode),
        update_bound_word(False, to_selected_word, edit_word_mode),
    ]


############################
#       API Functions      #
############################

@app.post("/newvoice")
async def generate__or_update_voice(
    audio: UploadFile = File(...),
    transcript: UploadFile = None,
    top_k: int = Form(0),
    top_p: float = Form(0.8),
    temperature: float = Form(1.0),
    stop_repetition: int = Form(3),
    kvcache: int = Form(1),
    sample_batch_size: int = Form(4),
    seed: int = Form(-1) # Random Seed
):
    logger.info("Received request to generate new voice.")

    # Convert to all lower to keep consistency
    voice = os.path.splitext(audio.filename)[0].lower()

    # Create the voice folder
    voice_folder = f"{VOICES_PATH}/{voice}"
    logger.debug(f"Creating voice folder: {voice_folder}")
    os.makedirs(voice_folder, exist_ok=True)

    audio_fn = os.path.join(voice_folder, audio.filename)
    logger.debug(f'Saving audio file to {audio_fn}')
    with open(audio_fn, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    # If we were not given a transcript, we can transcribe using the whisper model.
    if transcript == None:
        logger.debug('Did not receive a transcription. Transcribing using whisper model.')
        transcribe_state = transcribe(seed, audio_fn)
    else:
        transcript_text = await transcript.read()
        transcript_text = transcript_text.decode('utf-8').replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        logger.debug(f"Aligning {voice} transcript.")
        transcribe_state = align(seed, transcript_text, audio_fn, voice)

    logger.debug(f"Saving {voice} alignment to {voice_folder}/{voice}_alignment.json")
    with open(f'{voice_folder}/{voice}_alignment.json', 'w') as alignment_file:
        json.dump(transcribe_state, alignment_file, indent=4)

    logger.info(f"Saving the voice settings to {voice}_options.json")
    options = {
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "stop_repetition": stop_repetition,
        "kvcache": kvcache,
        "sample_batch_size": sample_batch_size,
        "seed": seed
    }
    with open(f'{VOICES_PATH}/{voice}/{voice}_options.json', 'w') as options_file:
        json.dump(options, options_file, indent=4)

    return {"message": f"{voice} voice generated successfully."}

@app.get("/voicelist")
async def get_voices():
    # Read all of the folders from VOICES_PATH and return them in a list.
    retList = []
    for folder in os.listdir(VOICES_PATH):
        if os.path.isdir(os.path.join(VOICES_PATH, folder)):
            retList.append(folder)
    return {"voices": retList}

@app.post("/editvoice/{voice}")
async def edit_voice_config(
    voice: str,
    time: float = None,
    top_k: int = None,
    top_p: float = None,
    temperature: float = None,
    stop_repetition: int = None,
    kvcache: int = None,
    sample_batch_size: int = None,
    seed: int = None
):
    # Check to see if the voice exists
    if not os.path.exists(f'{VOICES_PATH}/{voice}'):
        return {"message": f"Given voice '{voice}' does not exist.", "status_code": 404}
    
    # Parse json, update values that aren't 'None', then write values back to json file
    options_file = f'{VOICES_PATH}/{voice}/{voice}_options.json'
    with open(options_file, 'r') as f:
        new_options = json.load(f)
    
    if time is not None:
        new_options["time"] = time
    if top_k is not None:
        new_options["top_k"] = top_k
    if top_p is not None:
        new_options["top_p"] = top_p
    if temperature is not None:
        new_options["temperature"] = temperature
    if stop_repetition is not None:
        new_options["stop_repetition"] = stop_repetition
    if kvcache is not None:
        new_options["kvcache"] = kvcache
    if sample_batch_size is not None:
        new_options["sample_batch_size"] = sample_batch_size
    if seed is not None:
        new_options["seed"] = seed

    # Re-write the JSON file
    with open(options_file, "w") as f:
        json.dump(new_options, f, indent=4)
    

@app.post("/generateaudio/{voice}")
async def generate_voice_audio(
    voice: str,
    target_text: str = Form(""),
    device: str = Form(None),
):
    if not os.path.exists('{VOICES_PATH}/{voice}/{voice}.wav'):
        return {"message": "Missing the {voice}.wav file! Please recreate the voice using the /newvoice endpoint.", "status_code": 500}
    if not os.path.exists('{VOICES_PATH}/{voice}/{voice}_options.json'):
        return {"message": "Missing the {voice}_options.json file! Please recreate the voice using the /newvoice endpoint.", "status_code": 500}
    if not os.path.exists('{VOICES_PATH}/{voice}/{voice}_alignment.json'):
        return {"message": "Missing the {voice}_alignment.json file! Please recreate the voice using the /newvoice endpoint.", "status_code": 500}
    if not os.path.exists('{VOICES_PATH}/{voice}/{voice}.json'):
        return {"message": "Missing the sync_map {voice}.json file! Please recreate the voice using the /newvoice endpoint.", "status_code": 500}
    
    # Load in transcribe_state, voice_options and sync_map
    with open(f'{VOICES_PATH}/{voice}/{voice}_options.json', 'r') as f:
        options = json.load(f)
    with open(f'{VOICES_PATH}/{voice}/{voice}_alignment.json', 'r') as f:
        transcribe_state = json.load(f)
    with open(f'{VOICES_PATH}/{voice}/{voice}.json', 'r') as f:
        sync_map = json.load(f) # The sync_map only ever has a single fragment in it because we replace all newlines with spaces.
    output_audio, inference_transcript, audio_tensors = generate(options.seed, options.top_k,
                                                                 options.top_p, options.temperature,
                                                                 options.stop_repetition, options.sample_batch_size,
                                                                 options.kvcache, '{VOICES_PATH}/{voice}/{voice}.wav',
                                                                 transcribe_state, target_text, "TTS", sync_map.fragments[0].end)

    # Serve the generated audio as bytes
    audio_bytes = io.BytesIO()
    torchaudio.save(audio_bytes, audio_tensors[0], 16000, format="wav")
    audio_bytes.seek(0)
    
    # Serve the generated audio as bytes
    return StreamingResponse(audio_bytes, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    logger.info("Loading models...")
    load_models('whisperX', 'base.en', 'whisperX', 'giga830M')

    logger.info("Starting uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8245)
