import pyaudio
import wave
import keyboard
import time
from faster_whisper import WhisperModel
from googletrans import Translator

# Thiết lập mô hình Whisper
model_size = "large-v3"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAVE_OUTPUT_FILENAME = "temp_audio.wav"

p = pyaudio.PyAudio()

cumulative_text = ""  # Biến toàn cục lưu tất cả text thu được
translator = Translator()

def record_audio():
    """Ghi âm từ mic cho đến khi nhả phím Space."""
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    print("Đang ghi âm... (Giữ phím Space để nói, nhả ra để kết thúc)")
    while True:
        if keyboard.is_pressed('space'):
            data = stream.read(CHUNK)
            frames.append(data)
        else:
            # Khi người dùng nhả Space, thoát khỏi vòng lặp ghi âm
            if len(frames) > 0:
                break
        # Nếu nhấn 'q', thoát luôn
        if keyboard.is_pressed('q'):
            frames = []
            break

    stream.stop_stream()
    stream.close()

    # Nếu không có dữ liệu ghi âm thì trả về None
    if len(frames) == 0:
        return None

    # Lưu ra file WAV tạm
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return WAVE_OUTPUT_FILENAME

def transcribe_audio(audio_file):
    """Transcribe file audio bằng WhisperModel."""
    segments, info = model.transcribe(audio_file, beam_size=5)
    print("Ngôn ngữ dự đoán: '%s' (xác suất: %f)" % (info.language, info.language_probability))
    text_content = ""
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        text_content += segment.text.strip() + " "
    return text_content.strip()

def main():
    global cumulative_text
    print("Nhấn và giữ phím Space để ghi âm, nhả ra để chuyển voice sang text.")
    print("Nhấn 'q' để thoát.")

    while True:
        # Chờ cho đến khi Space được nhấn
        while True:
            if keyboard.is_pressed('space'):
                break
            if keyboard.is_pressed('q'):
                return

        # Ghi âm cho đến khi nhả Space
        audio_file = record_audio()
        if audio_file is None:
            # Không có âm thanh thu được
            print("Không có âm thanh!")
        else:
            # Sau khi ghi âm xong, transcribe
            transcribed_text = transcribe_audio(audio_file)
            # Thêm text này vào cumulative_text
            cumulative_text += " " + transcribed_text if cumulative_text else transcribed_text

            # In cumulative text
            print("\n[Cumulative Text]:", cumulative_text)

            # Dịch cumulative_text sang tiếng Việt
            translated = translator.translate(cumulative_text, dest='vi')
            print("[Cumulative Text Dịch]:", translated.text)

            print("\nNhấn và giữ Space để ghi tiếp, hoặc nhấn 'q' để thoát.\n")

        if keyboard.is_pressed('q'):
            break

if __name__ == "__main__":
    main()

p.terminate()