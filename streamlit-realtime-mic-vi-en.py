import os
import sys
import queue
import sounddevice as sd
import torch
import numpy as np
import time
import cv2
import subprocess
import soundfile as sf

from transformers import (
    SpeechEncoderDecoderModel,
    AutoFeatureExtractor,
    AutoTokenizer,
    GenerationConfig,
    WhisperProcessor,
    WhisperForConditionalGeneration
)

#=========================
# CÀI ĐẶT MÔ HÌNH WHISPER LARGE V2 (EN)
#=========================
model_path_en = "openai/whisper-large-v2"
processor_en = WhisperProcessor.from_pretrained(model_path_en)
model_en = WhisperForConditionalGeneration.from_pretrained(model_path_en).eval().to("cuda" if torch.cuda.is_available() else "cpu")

#=========================
# CÀI ĐẶT MÔ HÌNH WAV2VEC2-BARTPHO (VI)
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
    transcription = processor_en.tokenizer.decode(tokens, skip_special_tokens=False)
    return transcription, tokens

#=========================
# CẤU HÌNH GHI ÂM & VIDEO & LOGIC TÁCH KHOẢNG LẶNG
#=========================
sample_rate = 16000
channels = 1

frame_duration = 0.5  # Mỗi frame audio 0.5 giây
frame_samples = int(sample_rate * frame_duration)

audio_queue = queue.Queue()
buffer = np.zeros((0,), dtype=np.float32)

# Ngưỡng năng lượng để phân biệt tiếng nói và im lặng
energy_threshold = 0.001
silence_needed = 2.0  # số giây im lặng để kết thúc chunk
silence_frames = int(silence_needed / frame_duration)

silence_count = 0
speaking_started = False
chunk_audio = []

recognized_segments = []
chunk_start_time = None

# Mảng lưu toàn bộ âm thanh để sau này xuất ra file WAV
all_audio = []

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

# Khởi động stream audio
stream = sd.InputStream(
    samplerate=sample_rate,
    channels=channels,
    dtype='float32',
    callback=audio_callback
)
stream.start()

print("Bắt đầu thu âm và video... Nhấn 'q' trong cửa sổ video để dừng.")

# Thiết lập quay video
cap = cv2.VideoCapture(0)  # dùng camera laptop
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video_path = "output.mp4"
out = cv2.VideoWriter(out_video_path, fourcc, 20.0, (640,480))

start_recording_time = time.time()

def process_chunk(audio_array, start_t, end_t):
    if len(audio_array) == 0:
        return
    audio_wav = torch.from_numpy(audio_array).float().to(device)
    raw_transcription, tokens = transcribe_whisper(audio_wav, model_en, processor_en)

    if "<|en|>" in raw_transcription:
        lang = "en"
        final_text = raw_transcription.replace("<|en|>", "").strip()
    else:
        lang = "vi"
        audio_wav_cpu = audio_wav.cpu()
        transcription_vi = decode_wav_vi([audio_wav_cpu], model_vi)[0]
        final_text = transcription_vi

    print(f"[{lang}] {start_t:.2f} --> {end_t:.2f}: {final_text}")
    recognized_segments.append((start_t, end_t, final_text))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Video', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Kết thúc
            break

        # Xử lý âm thanh
        while not audio_queue.empty():
            data = audio_queue.get()
            data = data.flatten()
            buffer = np.concatenate((buffer, data))
            # Lưu toàn bộ audio
            all_audio.extend(data)

        # Xử lý frame audio
        if len(buffer) >= frame_samples:
            frame_data = buffer[:frame_samples]
            buffer = buffer[frame_samples:]
            rms = np.sqrt(np.mean(frame_data**2))
            current_time = time.time() - start_recording_time

            if rms < energy_threshold:
                # Im lặng
                if speaking_started:
                    silence_count += 1
            else:
                # Có tiếng nói
                if not speaking_started:
                    chunk_start_time = current_time
                speaking_started = True
                silence_count = 0
                chunk_audio.extend(frame_data)

            if speaking_started and silence_count >= silence_frames:
                # Kết thúc chunk
                chunk_end_time = current_time
                full_audio = np.array(chunk_audio, dtype=np.float32)
                process_chunk(full_audio, chunk_start_time, chunk_end_time)
                chunk_audio = []
                speaking_started = False
                silence_count = 0
                chunk_start_time = None

        time.sleep(0.001)

except KeyboardInterrupt:
    print("Dừng bởi KeyboardInterrupt")

finally:
    # Dừng video
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    stream.stop()
    stream.close()

    # Xử lý chunk cuối nếu còn
    if speaking_started and len(chunk_audio) > 0:
        chunk_end_time = time.time() - start_recording_time
        process_chunk(np.array(chunk_audio, dtype=np.float32), chunk_start_time, chunk_end_time)

    # Bạn có thể điều chỉnh offset để phụ đề sớm hoặc muộn hơn.
    # Nếu phụ đề đến muộn hơn hình, giảm offset. Ví dụ giảm 2 giây:
    offset = -1.0
    new_segments = []
    for (start_t, end_t, txt) in recognized_segments:
        new_segments.append((start_t - offset, end_t - offset, txt))
    recognized_segments = new_segments

    # Xuất phụ đề ra file SRT
    def srt_time(sec):
        # Tránh thời gian âm
        if sec < 0:
            sec = 0
        hours = int(sec//3600)
        minutes = int((sec%3600)//60)
        seconds = int(sec%60)
        milliseconds = int((sec - int(sec))*1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    srt_path = "subtitles.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (start_t, end_t, txt) in enumerate(recognized_segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{srt_time(start_t)} --> {srt_time(end_t)}\n")
            f.write(f"{txt}\n\n")

    # Ghi toàn bộ audio ra file WAV
    all_audio = np.array(all_audio, dtype=np.float32)
    wav_path = "output_audio.wav"
    sf.write(wav_path, all_audio, samplerate=sample_rate)

    # Kết hợp video và âm thanh
    ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
    output_av = "output_av.mp4"
    cmd_combine = [
        ffmpeg_path,
        "-i", out_video_path,
        "-i", wav_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        output_av,
        "-y"
    ]
    subprocess.run(cmd_combine, check=True)

    # Dùng ffmpeg để nhúng phụ đề vào video
    output_final = "output_subtitled.mp4"
    cmd_sub = [
        ffmpeg_path,
        "-i", output_av,
        "-vf", f"subtitles={srt_path}",
        "-c:a", "copy",
        output_final,
        "-y"
    ]
    subprocess.run(cmd_sub, check=True)

    print("Quá trình kết thúc, video với phụ đề được lưu ở:", output_final)
