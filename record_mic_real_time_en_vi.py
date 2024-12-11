import os
import sys
import queue
import sounddevice as sd
import torch
import numpy as np
import time
import torchaudio
from transformers import (
    SpeechEncoderDecoderModel,
    AutoFeatureExtractor,
    AutoTokenizer,
    GenerationConfig,
    WhisperProcessor,
    WhisperForConditionalGeneration
)

#=========================
# TẢI MÔ HÌNH WHISPER LARGE V2
#=========================
model_path_en = "openai/whisper-large-v2"
processor_en = WhisperProcessor.from_pretrained(model_path_en)
model_en = WhisperForConditionalGeneration.from_pretrained(model_path_en).eval().to("cuda" if torch.cuda.is_available() else "cpu")

#=========================
# TẢI MÔ HÌNH WAV2VEC2-BARTPHO (TIẾNG VIỆT)
#=========================
model_path_vi = 'nguyenvulebinh/wav2vec2-bartpho'
model_vi = SpeechEncoderDecoderModel.from_pretrained(model_path_vi).eval()
feature_extractor_vi = AutoFeatureExtractor.from_pretrained(model_path_vi)
tokenizer_vi = AutoTokenizer.from_pretrained(model_path_vi)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_vi = model_vi.to(device)

def decode_tokens(token_ids, skip_special_tokens=True, time_precision=0.02):
    timestamp_begin = tokenizer_vi.vocab_size
    outputs = [[]]
    for token in token_ids:
        if token >= timestamp_begin:
            timestamp = f" |{(token - timestamp_begin) * time_precision:.2f}| "
            outputs.append(timestamp)
            outputs.append([])
        else:
            outputs[-1].append(token)
    outputs = [
        s if isinstance(s, str) else tokenizer_vi.decode(s, skip_special_tokens=skip_special_tokens) for s in outputs
    ]
    return "".join(outputs).replace("< |", "<|").replace("| >", "|>")

def decode_wav_vi(audio_wavs, asr_model, prefix=""):
    audio_wavs_cpu = [wav.cpu() for wav in audio_wavs]
    input_values = feature_extractor_vi.pad(
        [{"input_values": feature.detach().cpu().numpy()} for feature in audio_wavs_cpu],
        padding=True,
        return_tensors="pt",
    )
    input_values = input_values.to(device)

    output_beam_ids = asr_model.generate(
        input_values['input_values'],
        attention_mask=input_values['attention_mask'],
        decoder_input_ids=tokenizer_vi.batch_encode_plus([prefix] * len(audio_wavs_cpu), return_tensors="pt")['input_ids'][..., :-1].to(device),
        generation_config=GenerationConfig(decoder_start_token_id=tokenizer_vi.bos_token_id),
        max_length=250,
        num_beams=5,
        no_repeat_ngram_size=4,
        num_return_sequences=1,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
    )

    output_text = [decode_tokens(sequence) for sequence in output_beam_ids.sequences]
    return output_text

def transcribe_whisper(audio_waveform, model, processor):
    audio_waveform_cpu = audio_waveform.cpu().numpy()
    input_features = processor(audio_waveform_cpu, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    predicted_ids = model.generate(
        input_features,
        max_length=225,
        num_beams=5,
        temperature=0.0,
        no_repeat_ngram_size=4,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=False
    )
    tokens = predicted_ids.sequences[0]
    transcription = processor.tokenizer.decode(tokens, skip_special_tokens=False)
    return transcription, tokens

#=========================
# CẤU HÌNH GHI ÂM & LOGIC TÁCH THEO KHOẢNG LẶNG
#=========================
sample_rate = 16000
channels = 1

frame_duration = 0.5  # Mỗi frame 0.5 giây
frame_samples = int(sample_rate * frame_duration)

audio_queue = queue.Queue()
buffer = np.zeros((0,), dtype=np.float32)

# Ngưỡng năng lượng để phân biệt tiếng nói và im lặng
energy_threshold = 0.001
silence_needed = 2.0  # số giây im lặng cần để kết thúc chunk
silence_frames = int(silence_needed / frame_duration)

silence_count = 0
speaking_started = False
chunk_audio = []

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

stream = sd.InputStream(
    samplerate=sample_rate,
    channels=channels,
    dtype='float32',
    callback=audio_callback
)
stream.start()

print("Bắt đầu thu âm...")

def process_chunk(audio_array):
    # Hàm xử lý chunk âm thanh khi gặp khoảng lặng
    # audio_array: np.ndarray
    if len(audio_array) == 0:
        return
    audio_wav = torch.from_numpy(audio_array).float().to(device)
    raw_transcription, tokens = transcribe_whisper(audio_wav, model_en, processor_en)

    if "<|en|>" in raw_transcription:
        lang = "en"
        final_text = raw_transcription.replace("<|en|>", "").strip()
    else:
        lang = "vi"
        # Nhận dạng tiếng Việt
        audio_wav_cpu = audio_wav.cpu()
        transcription_vi = decode_wav_vi([audio_wav_cpu], model_vi)[0]
        final_text = transcription_vi

    print(f"[Ngôn ngữ: {lang}] Kết quả cuối cùng: {final_text}")

try:
    while True:
        # Lấy dữ liệu âm thanh từ queue
        while not audio_queue.empty():
            data = audio_queue.get()
            data = data.flatten()
            buffer = np.concatenate((buffer, data))

        # Xử lý theo từng frame
        if len(buffer) >= frame_samples:
            frame_data = buffer[:frame_samples]
            buffer = buffer[frame_samples:]
            rms = np.sqrt(np.mean(frame_data**2))

            if rms < energy_threshold:
                # Frame im lặng
                if speaking_started:
                    silence_count += 1
                # Nếu chưa bắt đầu nói (speaking_started = False), ta chỉ bỏ qua
            else:
                # Frame có tiếng nói
                speaking_started = True
                silence_count = 0
                chunk_audio.extend(frame_data)

            # Kiểm tra nếu im lặng đủ lâu sau khi đã nói
            if speaking_started and silence_count >= silence_frames:
                # Kết thúc chunk
                full_audio = np.array(chunk_audio, dtype=np.float32)
                process_chunk(full_audio)

                # Reset chunk
                chunk_audio = []
                speaking_started = False
                silence_count = 0

        time.sleep(0.01)

except KeyboardInterrupt:
    print("Dừng ghi âm...")
    # Xử lý chunk cuối cùng nếu còn
    if speaking_started and len(chunk_audio) > 0:
        process_chunk(np.array(chunk_audio, dtype=np.float32))
    stream.stop()
    stream.close()
except Exception as e:
    print("Lỗi:", e)
    if stream:
        stream.stop()
        stream.close()


#ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"