import whisper
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import math
import soundfile as sf
import numpy as np
import librosa

from transformers import SpeechEncoderDecoderModel
from transformers import AutoFeatureExtractor, AutoTokenizer, GenerationConfig
import torchaudio
import torch
import requests

model_path = 'nguyenvulebinh/wav2vec2-bartpho'
model_wav = SpeechEncoderDecoderModel.from_pretrained(model_path).eval()
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if torch.cuda.is_available():
    model_wav = model_wav.cuda()

# DEVICE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# WHISPER
model_whisper_2 = whisper.load_model("large-v2", device=device)


def remove_silence(input_file):
    audio, sr = librosa.load(input_file)
    non_silent_intervals = librosa.effects.split(audio, top_db=40)

    non_silent_audio = []
    for interval in non_silent_intervals:
        non_silent_audio.extend(audio[interval[0]:interval[1]])

    non_silent_audio = np.array(non_silent_audio)
    output_file = "/media/brainx/Data/hdx/CBrain/ai-service-installer/audios/tmp/audio_remove_silence.wav"
    sf.write(output_file, non_silent_audio, sr)
    return output_file


def decode_tokens(token_ids, skip_special_tokens=True):
    """
    This function decodes a sequence of token IDs into a text string, optionally skipping special tokens.

    Args:
    token_ids (list): A list of token IDs to be decoded.
    skip_special_tokens (bool): Whether to skip special tokens during decoding. Defaults to True.

    Returns:
    str: The decoded text string.
    """
    timestamp_begin = tokenizer.vocab_size
    outputs = []
    for token in token_ids:
        if token < timestamp_begin:
            outputs.append(token)

    text = tokenizer.decode(outputs, skip_special_tokens=skip_special_tokens)[1:-1]
    return text


def speech_file_to_array_fn(path: str):
    """
    This function loads an audio file from a given path and converts it to a numpy array with a specified sampling rate.

    Args:
    path (str): Path to the input audio file.

    Returns:
    numpy.ndarray: The audio data as a numpy array with a sampling rate of 16000 Hz.
    """
    batch = {"file": path}
    speech_array, sampling_rate = torchaudio.load(batch["file"], normalize=True)
    if sampling_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_array = transform(speech_array)
    speech_array = speech_array[0]
    return speech_array


def get_transcript_vi(audio_wavs, prefix="", start_time=0, duration=20):
    device = next(model_wav.parameters()).device
    input_values = feature_extractor.pad(
        [{"input_values": feature} for feature in audio_wavs],
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors="pt",
    )

    output_beam_ids = model_wav.generate(
        input_values['input_values'].to(device),
        attention_mask=input_values['attention_mask'].to(device),
        decoder_input_ids=tokenizer.batch_encode_plus([prefix] * len(audio_wavs), return_tensors="pt")['input_ids'][...,
                          :-1].to(device),
        generation_config=GenerationConfig(decoder_start_token_id=tokenizer.bos_token_id),
        max_length=250,
        num_beams=25,
        no_repeat_ngram_size=4,
        num_return_sequences=1,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
    )

    output_text = [decode_tokens(sequence) for sequence in output_beam_ids.sequences]

    # Track the start and end time of each segment
    end_time = start_time + duration  # Duration per batch
    transcript_with_timestamps = [{
        "start_time": start_time,
        "end_time": end_time,
        "text": " ".join(output_text)
    }]
    return transcript_with_timestamps


def get_transcript_ml(speech_array):
    result = model_whisper_2.transcribe(speech_array, task="transcribe")
    return result["text"], result["segments"]


def get_language(audio_path):
    """
    This function detects the language spoken in an audio file using a pre-trained Whisper model.

    Args:
    audio_path (str): Path to the input audio file.

    Returns:
    str: The detected language.
    """
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model_whisper_2.device)

    _, probs = model_whisper_2.detect_language(mel)
    detected_language = max(probs, key=probs.get)

    return detected_language


def transcribe_2(audio_path, start_time=None, end_time=None, sample_rate=16000):
    """
    This function transcribes the speech in an audio file into text, identifying the language
    spoken in the audio file and choosing the appropriate transcription model accordingly.

    Args:
    audio_path (str): Path to the input audio file.

    Returns:
    tuple: A tuple containing the transcript of the audio and the detected language.
    """
    audio = whisper.load_audio(audio_path)
    if start_time is not None and end_time is not None:
        audio = audio[int(start_time * sample_rate): int(end_time * sample_rate)]
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model_whisper_2.device)

    _, probs = model_whisper_2.detect_language(mel)
    detected_language = max(probs, key=probs.get)

    output_path = remove_silence(audio_path)
    speech_array = speech_file_to_array_fn(output_path)
    if start_time is not None and end_time is not None:
        speech_array = speech_array[int(start_time * sample_rate): int(end_time * sample_rate)]

    transcript = ""
    timestamps = []
    # detected_language = "vi"
    if detected_language == "vi":
        batch_duration = 20  # Each batch of audio covers 20 seconds
        samples_per_batch = 16000 * batch_duration
        total_batches = math.ceil(len(speech_array) / samples_per_batch)
        audio_wavs = [speech_array[i * samples_per_batch: (i + 1) * samples_per_batch] for i in range(total_batches)]

        current_time = 0  # Track current time for timestamps
        for i in range(0, len(audio_wavs), 1):
            result = get_transcript_vi(audio_wavs[i: (i + 1)], start_time=current_time, duration=batch_duration)
            transcript += result[0]["text"] + " "
            timestamps.extend(result)
            current_time += batch_duration  # Increment start time by the batch duration
    else:
        # Whisper model handles timestamping internally
        transcript, segments = get_transcript_ml(speech_array)
        timestamps = segments

    return transcript, detected_language, timestamps


def translate(text, dest_lang):
    url = "http://localhost:8000/translate"
    data = {
        "text": text,
        "dest": dest_lang
    }

    translated_text = ""
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            translated_text = result['translated_text']
        else:
            print("Failed to translate. Status code:", response.status_code)
            print("Error message:", response.text)
    except Exception as e:
        print("An error occurred:", str(e))

    return translated_text


def transcribe_3(audio_path, target_lang, start_time=None, end_time=None, sample_rate=16000):
    """
    This function transcribes the speech in an audio file into text, identifying the language
    spoken in the audio file and choosing the appropriate transcription model accordingly.

    Args:
    audio_path (str): Path to the input audio file.

    Returns:
    tuple: A tuple containing the transcript of the audio and the detected language.
    """
    audio = whisper.load_audio(audio_path)
    if start_time is not None and end_time is not None:
        audio = audio[int(start_time * sample_rate): int(end_time * sample_rate)]
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model_whisper_2.device)

    _, probs = model_whisper_2.detect_language(mel)
    detected_language = max(probs, key=probs.get)

    output_path = remove_silence(audio_path)
    speech_array = speech_file_to_array_fn(output_path)
    if start_time is not None and end_time is not None:
        speech_array = speech_array[int(start_time * sample_rate): int(end_time * sample_rate)]

    transcript = ""
    timestamps = []

    if detected_language == "vi":
        batch_duration = 20  # Each batch of audio covers 20 seconds
        samples_per_batch = 16000 * batch_duration
        total_batches = math.ceil(len(speech_array) / samples_per_batch)
        audio_wavs = [speech_array[i * samples_per_batch: (i + 1) * samples_per_batch] for i in range(total_batches)]

        current_time = 0  # Track current time for timestamps
        for i in range(0, len(audio_wavs), 1):
            result = get_transcript_vi(audio_wavs[i: (i + 1)], start_time=current_time, duration=batch_duration)
            transcript += result[0]["text"] + " "
            timestamps.extend(result)
            current_time += batch_duration  # Increment start time by the batch duration
    else:
        # Whisper model handles timestamping internally
        transcript, segments = get_transcript_ml(speech_array)
        timestamps = segments

    for i in range(len(timestamps)):
        timestamps[i]['translated_text'] = translate(timestamps[i]['text'], target_lang)
        pass

    translated_text = translate(transcript, target_lang)
    return transcript, detected_language, timestamps, translated_text


app = FastAPI()


class QueryRequest_Audio(BaseModel):
    audio_path: str


class QueryRequest_AudioVersion3(BaseModel):
    audio_path: str
    target_lang: str


@app.post("/s2t_version1")
async def query_handler_2(request: QueryRequest_Audio):
    audio_path = request.audio_path
    transcript, detected_language, timestamps = transcribe_2(audio_path)

    return {
        "transcript": str(transcript),
        "language": str(detected_language),
        "timestamps": timestamps
    }


@app.post("/language")
async def query_handler_2(request: QueryRequest_Audio):
    audio_path = request.audio_path
    detected_language = get_language(audio_path)

    return {
        "language": str(detected_language),
    }


@app.post("/s2t_version3")
async def query_handler_2(request: QueryRequest_AudioVersion3):
    audio_path = request.audio_path
    target_lang = request.target_lang
    transcript, detected_language, timestamps, translated_text = transcribe_3(audio_path, target_lang)

    return {
        "transcript": str(transcript),
        "translated_text": str(translated_text),
        "language": str(detected_language),
        "timestamps": timestamps,

    }


# uvicorn --reload api_version1:app  --host 0.0.0.0 --port 8011
# nohup uvicorn --reload api_version1:app --host 0.0.0.0 --port 8011 > output.log 2>&1 &
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8011)