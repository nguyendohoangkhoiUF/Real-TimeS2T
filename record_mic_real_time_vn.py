import os
import sys
import queue
import sounddevice as sd
import torch
import numpy as np
import time
import torchaudio
from transformers import SpeechEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer, GenerationConfig

#=========================
# TẢI MÔ HÌNH
#=========================
model_path = 'nguyenvulebinh/wav2vec2-bartpho'
model = SpeechEncoderDecoderModel.from_pretrained(model_path).eval()
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

#=========================
# HÀM TIỆN ÍCH
#=========================
def decode_tokens(token_ids, skip_special_tokens=True, time_precision=0.02):
    timestamp_begin = tokenizer.vocab_size
    outputs = [[]]
    for token in token_ids:
        if token >= timestamp_begin:
            timestamp = f" |{(token - timestamp_begin) * time_precision:.2f}| "
            outputs.append(timestamp)
            outputs.append([])
        else:
            outputs[-1].append(token)
    outputs = [
        s if isinstance(s, str) else tokenizer.decode(s, skip_special_tokens=skip_special_tokens) for s in outputs
    ]
    return "".join(outputs).replace("< |", "<|").replace("| >", "|>")

def decode_wav(audio_wavs, asr_model, prefix=""):
    device = next(asr_model.parameters()).device
    input_values = feature_extractor.pad(
        [{"input_values": feature} for feature in audio_wavs],
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors="pt",
    )

    output_beam_ids = asr_model.generate(
        input_values['input_values'].to(device),
        attention_mask=input_values['attention_mask'].to(device),
        decoder_input_ids=tokenizer.batch_encode_plus([prefix] * len(audio_wavs), return_tensors="pt")['input_ids'][..., :-1].to(device),
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
    return output_text


#=========================
# CẤU HÌNH GHI ÂM
#=========================
sample_rate = 16000        # Tần số mẫu
channels = 1               # Số kênh, 1 là mono
chunk_duration = 3.0       # Thời gian mỗi chunk (giây)
chunk_samples = int(sample_rate * chunk_duration)

# Hàng đợi để lưu trữ data audio từ callback
audio_queue = queue.Queue()

# Biến lưu trữ tạm thời tín hiệu audio
buffer = np.zeros((0,), dtype=np.float32)

#=========================
# CALLBACK GHI ÂM
#=========================
def audio_callback(indata, frames, time_info, status):
    # indata là mảng numpy chứa dữ liệu âm thanh thu được
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

#=========================
# BẮT ĐẦU GHI ÂM TỪ MICRO
#=========================
stream = sd.InputStream(
    samplerate=sample_rate,
    channels=channels,
    dtype='float32',
    callback=audio_callback
)
stream.start()

print("Bắt đầu thu âm... Nói vào micro...")

try:
    while True:
        # Lấy dữ liệu âm thanh từ hàng đợi
        while not audio_queue.empty():
            data = audio_queue.get()
            data = data.flatten()
            buffer = np.concatenate((buffer, data))

        # Khi đã đủ chunk_samples, lấy một chunk ra để nhận dạng
        if len(buffer) >= chunk_samples:
            # Lấy chunk
            chunk_data = buffer[:chunk_samples]
            # Cắt chunk ra khỏi buffer
            buffer = buffer[chunk_samples:]

            # Chuyển dữ liệu thành tensor
            audio_wav = torch.from_numpy(chunk_data).float()

            # Thực hiện nhận dạng
            transcription = decode_wav([audio_wav], model)
            print("Transcription:", transcription[0])

        # Chờ một chút trước khi lặp lại
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Dừng ghi âm...")
    stream.stop()
    stream.close()
except Exception as e:
    print("Lỗi:", e)
    stream.stop()
    stream.close()
