import os
import shutil  # Added to help with file operations
from transformers import SpeechEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer, GenerationConfig
import torchaudio
import torchaudio.transforms as T
import torch
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import re

# Load the model and tokenizer
model_path = 'nguyenvulebinh/wav2vec2-bartpho'
model = SpeechEncoderDecoderModel.from_pretrained(model_path).eval()
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Chuyển mô hình sang GPU nếu khả dụng
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Hàm chuyển đổi tần số lấy mẫu
def resample_audio(audio_wav, original_sample_rate, target_sample_rate):
    if original_sample_rate != target_sample_rate:
        resampler = T.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        audio_wav = resampler(audio_wav)
    return audio_wav

# Hàm chia âm thanh theo khoảng lặng và độ dài
def split_audio_on_silence_and_length(waveform, sample_rate, silence_threshold=-40, min_silence_len=300, chunk_padding=100, max_chunk_length=10):
    # Chuyển waveform thành AudioSegment
    waveform_np = waveform.numpy()
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.mean(axis=0)
    waveform_np = (waveform_np * (2**15)).astype(np.int16)  # Chuyển từ float [-1,1] sang int16
    audio_segment = AudioSegment(
        data=waveform_np.tobytes(),
        sample_width=2,  # 2 bytes cho int16
        frame_rate=sample_rate,
        channels=1
    )

    # Sử dụng pydub để chia âm thanh theo khoảng lặng
    chunks = split_on_silence(
        audio_segment,
        min_silence_len=min_silence_len,  # Đơn vị: milliseconds
        silence_thresh=silence_threshold,  # Đơn vị: dBFS
        keep_silence=chunk_padding  # Đơn vị: milliseconds
    )

    # Tiếp tục chia các đoạn dài hơn max_chunk_length thành các đoạn nhỏ hơn
    max_ms = max_chunk_length * 1000  # Chuyển sang milliseconds
    all_chunks = []
    for chunk in chunks:
        if len(chunk) > max_ms:
            # Chia nhỏ đoạn này thành các đoạn nhỏ hơn
            for i in range(0, len(chunk), max_ms):
                sub_chunk = chunk[i:i+max_ms]
                all_chunks.append(sub_chunk)
        else:
            all_chunks.append(chunk)

    # Chuyển các AudioSegment chunks trở lại thành Tensors
    tensor_chunks = []
    for i, chunk in enumerate(all_chunks):
        samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / (2**15)
        tensor_chunk = torch.from_numpy(samples)
        print(f"Đoạn {i+1}: độ dài {tensor_chunk.shape[-1]/sample_rate:.2f} giây")
        tensor_chunks.append(tensor_chunk)

    return tensor_chunks

# Hàm lưu các file chunk vào thư mục
def save_chunks_to_folder(chunks, sample_rate, output_folder="chunks_output"):
    # Nếu thư mục tồn tại và có file bên trong, xóa trắng thư mục
    if os.path.exists(output_folder):
        # Xóa tất cả các file trong thư mục
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Xóa file hoặc symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Xóa thư mục con
            except Exception as e:
                print(f'Không thể xóa {file_path}. Lỗi: {e}')
    else:
        os.makedirs(output_folder)

    # Lưu từng đoạn âm thanh
    for i, chunk in enumerate(chunks):
        file_path = os.path.join(output_folder, f"chunk_{i+1}.wav")
        torchaudio.save(file_path, chunk.unsqueeze(0), sample_rate)
        print(f"Lưu đoạn {i+1} tại: {file_path}")

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

# Đường dẫn file âm thanh
mp3_path = 'Demo23_9S2T/Bac3.mp3'

# Đọc file MP3 và chuyển đổi tần số lấy mẫu nếu cần
waveform, sample_rate = torchaudio.load(mp3_path)
waveform = resample_audio(waveform, original_sample_rate=sample_rate, target_sample_rate=16000)
sample_rate = 16000  # Cập nhật tần số lấy mẫu sau khi resample

# **Thêm đoạn code này để kiểm tra và xóa thư mục chunks_output trước khi tiếp tục**
output_folder = "chunks_output"
if os.path.exists(output_folder):
    # Xóa tất cả các file trong thư mục
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Xóa file hoặc symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Xóa thư mục con
        except Exception as e:
            print(f'Không thể xóa {file_path}. Lỗi: {e}')
else:
    os.makedirs(output_folder)

# Chia nhỏ âm thanh theo các khoảng lặng và độ dài tối đa
audio_chunks = split_audio_on_silence_and_length(
    waveform,
    sample_rate,
    silence_threshold=-40,  # Tăng ngưỡng để nhạy hơn với khoảng lặng
    min_silence_len=300,    # Giảm độ dài tối thiểu của khoảng lặng để phát hiện nhiều khoảng lặng hơn
    chunk_padding=100,      # Thêm một chút khoảng lặng vào đầu và cuối đoạn
    max_chunk_length=10     # Độ dài tối đa của mỗi đoạn (giây)
)

print(f"Số lượng đoạn âm thanh được chia: {len(audio_chunks)}")

# Lưu các đoạn âm thanh vào thư mục
save_chunks_to_folder(audio_chunks, sample_rate, output_folder=output_folder)

print("\nLưu tất cả các đoạn âm thanh thành công!")

# Directory containing the WAV files
wav_directory = output_folder

# Get a list of all WAV files in the directory
wav_files = [f for f in os.listdir(wav_directory) if f.endswith('.wav')]

# Function to extract numerical index from filename
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

# Sort the filenames based on the extracted number
wav_files.sort(key=extract_number)

# Process each WAV file in sorted order
for wav_file in wav_files:
    # Construct the full path to the audio file
    audio_path = os.path.join(wav_directory, wav_file)

    # Load the audio file
    audio_wav = torchaudio.load(audio_path)[0].squeeze()

    # Transcribe the audio
    transcription = decode_wav([audio_wav], model)

    # Print out the transcription immediately
    print(f"File: {wav_file}\nTranscription: {transcription[0]}\n")
