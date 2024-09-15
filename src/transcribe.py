import asyncio
import io
import logging
import pathlib
import re
import tempfile
import time
from typing import Iterator
import uuid

import modal
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import subprocess
import requests

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID)

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "https://github.com/openai/whisper/archive/v20230314.tar.gz",
        "ffmpeg-python",
        "pytube @ git+https://github.com/felipeucelli/pytube",
        "transformers[torch]",
        "hf-transfer",
        "huggingface_hub",
        "requests",
        "context_cite",
    )
    .pip_install(
        "flash-attn", extra_options="--no-build-isolation"
    )
    .env(  # hf-transfer: faster downloads, but fewer comforts
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    )
    .run_function(  # download the model
        download_model,
        timeout=20 * 60, # in seconds
        secrets=[modal.Secret.from_name("my-huggingface-secret")],
    )
)
app = modal.App(name="example-whisper-streaming-testing", image=image)
vol = modal.Volume.from_name("ems-storage")
web_app = FastAPI()
CHARLIE_CHAPLIN_DICTATOR_SPEECH_URL = (
    "https://www.youtube.com/watch?v=J7GY1Xg6X20"
)

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PROMPT_TEMPLATE = '''Here's an example of an EMT call summary: 
# Dispatch Information:
# Dispatched for a possible cardiac arrest. It is an unresponsive patient who is not breathing at a residence. PD is responding and EMS has been notified.
# Subjective Information:
# Upon our arrival to the scene, the patient's wife directed us to the third floor apartment, as this is the patient's residence. The wife hysterically explained that she could not get EMS because she couldn't get the patient to the floor. The patient's wife explained that she awoke and noticed that the patient was not breathing. She tried to wake him up, but could not. She then called 911. The wife stated that the patient had chest pain recently and had an appointment to see his doctor tomorrow.
# Objective Information:
# The patient, a 58-year-old male, was found on his bed in cardiac arrest. EMS was updated on the patient's status. The patient has no palpable pulses or spontaneous respirations at this time and has a gray/white skin color. The patient was moved to floor so effective CPR could be performed. Once the patient was one the floor CPR was started and an AED was placed on the patient. CPR is being performed with rotation of the compressors every 2 minutes. An OPA was placed and the patient was ventilated with a BVM connected to high flow oxygen. The BVM is effective at producing adequate chest rise with ventilations and there seems to be no obstructions noted. The AED has advised responders that there is "No shock advised" and CPR was resumed.
# EMS arrived and report was given to the paramedics. EMS directed us to continue assisting with compressions/CPR and EVAC of the patient.
# Assessment Information:
# Cardiac Arrest - Suspected/possible cardiac in nature
# Plan Information:
# Arrived on scene, moved patient from bed to floor, began CPR, attached AED, BVM, Oxygen, OPA utilized. Report to EMS when they arrived. Assisted with CPR/compressions and EVAC during resuscitation. 

# The following is a transcript for an EMT call:

# {transcript}

# Provide a summary of this call in the same format as the example. Be specific, detailed, and professional.'''

# PROMPT_TEMPLATE = '''Here's an example of an EMT call summary: 
    
# <BEGIN EXAMPLE>
# Dispatch Information:
# Dispatched for a possible cardiac arrest. It is an unresponsive patient who is not breathing at a residence. PD is responding and EMS has been notified.
# Subjective Information:
# Upon our arrival to the scene, the patient's wife directed us to the third floor apartment, as this is the patient's residence. The wife hysterically explained that she could not get EMS because she couldn't get the patient to the floor. The patient's wife explained that she awoke and noticed that the patient was not breathing. She tried to wake him up, but could not. She then called 911. The wife stated that the patient had chest pain recently and had an appointment to see his doctor tomorrow.
# Objective Information:
# The patient, a 58-year-old male, was found on his bed in cardiac arrest. EMS was updated on the patient's status. The patient has no palpable pulses or spontaneous respirations at this time and has a gray/white skin color. The patient was moved to floor so effective CPR could be performed. Once the patient was one the floor CPR was started and an AED was placed on the patient. CPR is being performed with rotation of the compressors every 2 minutes. An OPA was placed and the patient was ventilated with a BVM connected to high flow oxygen. The BVM is effective at producing adequate chest rise with ventilations and there seems to be no obstructions noted. The AED has advised responders that there is "No shock advised" and CPR was resumed.
# EMS arrived and report was given to the paramedics. EMS directed us to continue assisting with compressions/CPR and EVAC of the patient.
# Assessment Information:
# Cardiac Arrest - Suspected/possible cardiac in nature
# Plan Information:
# Arrived on scene, moved patient from bed to floor, began CPR, attached AED, BVM, Oxygen, OPA utilized. Report to EMS when they arrived. Assisted with CPR/compressions and EVAC during resuscitation. 
# <END EXAMPLE>

# The following is a transcript for an EMT call:

# <BEGIN TRANSCRIPT>
# {transcript}
# <END TRANSCRIPT>

# Provide a summary of this call in the same format as the example. Be detailed, specific, and professional.  Most importantly, do not make up any information that is not included in the trasncript. Do NOT use information from the example summary in your summary, that was only an example. If the trasncript is limited or missing, it is OKAY to provide a shorter summary. Do not include <BEGIN/END SUMMARY> tags.'''

PROMPT_TEMPLATE = '''Here's an example of an EMT call summary: 
    
<BEGIN EXAMPLE>
Dispatch Information:
Dispatched for a possible cardiac arrest. It is an unresponsive patient who is not breathing at a residence. PD is responding and EMS has been notified.
Subjective Information:
Upon our arrival to the scene, the patient's wife directed us to the third floor apartment, as this is the patient's residence. The wife hysterically explained that she could not get EMS because she couldn't get the patient to the floor. The patient's wife explained that she awoke and noticed that the patient was not breathing. She tried to wake him up, but could not. She then called 911. The wife stated that the patient had chest pain recently and had an appointment to see his doctor tomorrow.
Objective Information:
The patient, a 58-year-old male, was found on his bed in cardiac arrest. EMS was updated on the patient's status. The patient has no palpable pulses or spontaneous respirations at this time and has a gray/white skin color. The patient was moved to floor so effective CPR could be performed. Once the patient was one the floor CPR was started and an AED was placed on the patient. CPR is being performed with rotation of the compressors every 2 minutes. An OPA was placed and the patient was ventilated with a BVM connected to high flow oxygen. The BVM is effective at producing adequate chest rise with ventilations and there seems to be no obstructions noted. The AED has advised responders that there is "No shock advised" and CPR was resumed.
EMS arrived and report was given to the paramedics. EMS directed us to continue assisting with compressions/CPR and EVAC of the patient.
Assessment Information:
Cardiac Arrest - Suspected/possible cardiac in nature
Plan Information:
Arrived on scene, moved patient from bed to floor, began CPR, attached AED, BVM, Oxygen, OPA utilized. Report to EMS when they arrived. Assisted with CPR/compressions and EVAC during resuscitation. 
<END EXAMPLE>

The following is a transcript for an EMT call:

<BEGIN TRANSCRIPT>
{context}
<END TRANSCRIPT>

{query}'''

SUMMARY_QUERY = '''Provide a summary of this call in the same format as the example. Be detailed, specific, and professional.  Most importantly, do not make up any information that is not included in the trasncript. Do NOT use information from the example summary in your summary, that was only an example. If the trasncript is limited or missing, it is OKAY to provide a shorter summary. Do not include <BEGIN/END SUMMARY> tags.'''

# MODEL_ID = "8w6yyp2q" # Replace with your model ID
# BASETEN_API_KEY = "YMKFudUr.FcjOTi13DlaR3ZtCbBIumoXeqFJy25yx" # [Optional] Replace with your deployment ID

def load_audio(data: bytes, start=None, end=None, sr: int = 16000):
    import ffmpeg
    import numpy as np

    try:
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        fp.write(data)
        fp.close()
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        if start is None and end is None:
            out, _ = (
                ffmpeg.input(fp.name, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(
                    cmd=["ffmpeg", "-nostdin"],
                    capture_stdout=True,
                    capture_stderr=True,
                )
            )
        else:
            out, _ = (
                ffmpeg.input(fp.name, threads=0)
                .filter("atrim", start=start, end=end)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(
                    cmd=["ffmpeg", "-nostdin"],
                    capture_stdout=True,
                    capture_stderr=True,
                )
            )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def split_silences(
    path: str, min_segment_length: float = 30.0, min_silence_length: float = 0.8
) -> Iterator[tuple[float, float]]:
    """
    Split audio file into contiguous chunks using the ffmpeg `silencedetect` filter.
    Yields tuples (start, end) of each chunk in seconds.

    Parameters
    ----------
    path: str
        path to the audio file on disk.
    min_segment_length : float
        The minimum acceptable length for an audio segment in seconds. Lower values
        allow for more splitting and increased parallelizing, but decrease transcription
        accuracy. Whisper models expect to transcribe in 30 second segments, so this is the
        default minimum.
    min_silence_length : float
        Minimum silence to detect and split on, in seconds. Lower values are more likely to split
        audio in middle of phrases and degrade transcription accuracy.
    """
    import ffmpeg

    silence_end_re = re.compile(
        r" silence_end: (?P<end>[0-9]+(\.?[0-9]*)) \| silence_duration: (?P<dur>[0-9]+(\.?[0-9]*))"
    )

    metadata = ffmpeg.probe(path)
    duration = float(metadata["format"]["duration"])

    reader = (
        ffmpeg.input(str(path))
        .filter("silencedetect", n="-10dB", d=min_silence_length)
        .output("pipe:", format="null")
        .run_async(pipe_stderr=True)
    )

    cur_start = 0.0
    num_segments = 0

    while True:
        line = reader.stderr.readline().decode("utf-8")
        if not line:
            break
        match = silence_end_re.search(line)
        if match:
            silence_end, silence_dur = match.group("end"), match.group("dur")
            split_at = float(silence_end) - (float(silence_dur) / 2)

            if (split_at - cur_start) < min_segment_length:
                continue

            yield cur_start, split_at
            cur_start = split_at
            num_segments += 1

    # silencedetect can place the silence end *after* the end of the full audio segment.
    # Such segments definitions are negative length and invalid.
    if duration > cur_start and (duration - cur_start) > min_segment_length:
        yield cur_start, duration
        num_segments += 1
    if num_segments == 0:
        yield cur_start, duration
        num_segments += 1
    print(f"Split {path} into {num_segments} segments")


@app.function()
def download_mp3_from_youtube(youtube_url: str) -> bytes:
    from pytube import YouTube

    logging.getLogger("pytube").setLevel(logging.INFO)
    yt = YouTube(youtube_url)
    video = yt.streams.filter(only_audio=True).first()
    buffer = io.BytesIO()
    video.stream_to_buffer(buffer)
    buffer.seek(0)
    return buffer.read()


@app.function(gpu="a10g")
def transcribe_segment(
    start: float,
    end: float,
    audio_data: bytes,
    model: str,
):
    import torch
    import whisper

    print(
        f"Transcribing segment {start:.2f} to {end:.2f} ({end - start:.2f}s duration)"
    )

    t0 = time.time()
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    model = whisper.load_model(model, device=device)
    np_array = load_audio(audio_data, start=start, end=end)
    result = model.transcribe(np_array, language="en", fp16=use_gpu)  # type: ignore
    print(
        f"Transcribed segment {start:.2f} to {end:.2f} ({end - start:.2f}s duration) in {time.time() - t0:.2f} seconds."
    )

    # Add back offsets.
    for segment in result["segments"]:
        segment["start"] += start
        segment["end"] += start

    return result


async def stream_whisper(audio_data: bytes):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(audio_data)
        f.flush()
        segment_gen = split_silences(f.name)

    async for result in transcribe_segment.starmap(
        segment_gen, kwargs=dict(audio_data=audio_data, model="tiny.en")
    ):
        # Must cooperatively yield here otherwise `StreamingResponse` will not iteratively return stream parts.
        # see: https://github.com/python/asyncio/issues/284#issuecomment-154162668
        await asyncio.sleep(0)
        yield result["text"], result["segments"]

def get_summary(transcript):
    messages = [
        {"role": "system", "content": "You are a trained EMT."},
        {"role": "user", "content": PROMPT_TEMPLATE.format(transcript=transcript)},
    ]

    payload = {
        "messages": messages,
        "stream": False,
        "max_tokens": 16096,
        "temperature": 0.9
    }

    # Call model endpoint
    res = requests.post(
        f"https://model-{MODEL_ID}.api.baseten.co/production/predict",
        headers={"Authorization": f"Api-Key {BASETEN_API_KEY}"},
        json=payload,
        stream=False
    )

    return res.text

@app.function(
    gpu="a100-80gb",
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    timeout=20000,
    volumes={"/cache/": vol}
)
def get_summary_modal(transcript, filename, volumes={"/cache/": vol}):
    vol.reload()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from context_cite import ContextCiter
    import nltk
    nltk.download('punkt_tab')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    cc = ContextCiter.from_pretrained(
        MODEL_ID, transcript, SUMMARY_QUERY,
        device="cuda",
        prompt_template=PROMPT_TEMPLATE,
        generate_kwargs=dict(
            max_length=8192,
            pad_token_id=tokenizer.eos_token_id
        )
    )

    # messages = [
    #     {"role": "system", "content": "You are a trained EMT."},
    #     {"role": "user", "content": PROMPT_TEMPLATE.format(transcript=transcript)},
    # ]

    # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_ID,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    # ).cuda()

    # tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()

    # outputs = model.generate(tokens, max_length=8192, pad_token_id=tokenizer.eos_token_id)

    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cc._compute_masks_and_logit_probs()
    cc.response
    cc.model = None

    os.makedirs("/cache/cc", exist_ok=True)
    cache_filename = f"/cache/cc/{filename}.pt"
    torch.save(cc, cache_filename)

    vol.commit()

    return cc.response

# @app.function(cpu=8, volumes={"/cache/": vol})
@app.function(cpu=8, volumes={"/cache/": vol})
@modal.web_endpoint(method="GET")
async def cite(filename: str, start_idx: int, end_idx: int):
    from context_cite import ContextCiter
    import numpy as np
    import torch

    vol.reload()
    cache_filename = f"/cache/cc/{filename}.pt"
    cc = torch.load(cache_filename)
    attrib = cc.get_attributions(start_idx, end_idx)
    top_indices = np.argsort(attrib)[::-1][:5]
    lens = [len(part.split()) for part in cc.partitioner.parts]
    indexes = [0] + list(np.cumsum(lens))
    indexes = [indexes[ind] for ind in top_indices]

    transcript_cache = f"/cache/transcript/{filename}"
    with open(f"{transcript_cache}_timings.npy", "rb") as f:
        timings = np.load(f)

    times = [timings[ind] for ind in indexes]
    text = [cc.partitioner.get_source(ind) for ind in top_indices]

    return JSONResponse(content=dict(times=times, text=text), status_code=200)


# @app.function(cpu=8, volumes={"/cache/": vol})
@app.function(volumes={"/cache/": vol})
@modal.web_endpoint(method="GET")
async def get_timestamp(filename:str, word_idx: int):
    import numpy as np

    transcript_cache = f"/cache/transcript/{filename}"
    with open(f"{transcript_cache}_timings.npy", "rb") as f:
        timings = np.load(f)

    return JSONResponse(content=dict(timestamp=timings[word_idx]), status_code=200)


# @web_app.post("/transcribe", volumes={"/cache/": vol})
@app.function(cpu=8, volumes={"/cache/": vol}, timeout=20000)
async def transcribe_function(audio_data, name):
    """
    Usage:

    ```sh
    curl --no-buffer \
        https://modal-labs--example-whisper-streaming-web.modal.run/transcribe?url=https://www.youtube.com/watch?v=s_LncVnecLA"
    ```

    This endpoint will stream back the Youtube's audio transcription as it makes progress.

    Some example Youtube videos for inspiration:

    1. Churchill's 'We shall never surrender' speech - https://www.youtube.com/watch?v=s_LncVnecLA
    2. Charlie Chaplin's final speech from The Great Dictator - https://www.youtube.com/watch?v=J7GY1Xg6X20
    """
    import numpy as np
    vol.reload()

    os.makedirs("/cache/audio", exist_ok=True)

    filename = f"/cache/audio/{name}"

    filename_webm = f"{filename}.webm"
    filename_wav = f"{filename}.wav"
    if True: # not os.path.isfile(filename_wav):

        with open(filename_webm, "wb") as f:
            f.write(audio_data)

        subprocess.run(["ffmpeg", "-y", "-i", filename_webm, "-c:a", "pcm_s16le", filename_wav])


    os.makedirs("/cache/transcript", exist_ok=True)
    transcript_cache = f"/cache/transcript/{name}"
    if True: #not os.path.isfile(transcript_cache):
        
        filepath = pathlib.Path(filename_wav)
        audio_data = filepath.read_bytes()
        result = ""
        timings = []
        async for text, segment in stream_whisper(audio_data):
            # result += text + "\n"
            for seg in segment:
                result += seg["text"]
                num_words = len(seg["text"].split())
                timings.extend(np.linspace(seg["start"], seg["end"], num_words, endpoint=False))

        with open(f"{transcript_cache}_timings.npy", "wb") as f:
            np.save(f, np.array(timings))

        with open(transcript_cache, "w") as f:
            f.write(result)
    vol.commit()

    with open(transcript_cache, "r") as f:
        result = f.read()

    print("TRANSCRIPT", result)

    raw_llama_output = get_summary_modal.remote(result, name)

    vol.commit()
    
    return JSONResponse(content=dict(transcript=result, raw_llama_output=raw_llama_output), status_code=200)

# @modal.web_endpoint(method="POST")
@web_app.post("/transcribe")
async def transcribe(audio: UploadFile):
    audio_data = await audio.read()

    call = transcribe_function.spawn(audio_data, audio.filename)
    return {"call_id": call.object_id}

@web_app.get("/result/{call_id}")
async def poll_results(call_id: str):
    function_call = modal.functions.FunctionCall.from_id(call_id)
    try:
        return function_call.get(timeout=0)
    except TimeoutError:
        http_accepted_code = 202
        return JSONResponse({}, status_code=http_accepted_code)

@app.function()
@modal.asgi_app()
def web():
    return web_app


@app.function(timeout=20000, volumes={"/cache/": vol})
async def transcribe_cli(audio_data: bytes, name: str):
    import numpy as np
    # async for result in stream_whisper(data):
    #     print(result)
    os.makedirs("/cache/transcript", exist_ok=True)
    transcript_cache = f"/cache/transcript/{name}"

    result = ""
    timings = []
    async for text, segment in stream_whisper(audio_data):
        # result += text + "\n"
        for seg in segment:
            result += seg["text"]
            num_words = len(seg["text"].split())
            timings.extend(np.linspace(seg["start"], seg["end"], num_words, endpoint=False))

    with open(f"{transcript_cache}_timings.npy", "wb") as f:
        np.save(f, np.array(timings))
    vol.commit()

    print("TRANSCRIPT", result)

    raw_llama_output = get_summary_modal.remote(result, name)
    print(raw_llama_output)

    print(cite.remote(name, 25, 150))
    breakpoint()


@app.local_entrypoint()
def main(path: str, name: str):
    if path.startswith("https"):
        data = download_mp3_from_youtube.remote(path)
        suffix = ".mp3"
    else:
        filepath = pathlib.Path(path)
        data = filepath.read_bytes()
        suffix = filepath.suffix
    transcribe_cli.remote(
        data,
        name
    )