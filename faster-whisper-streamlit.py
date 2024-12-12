import pyaudio
import wave
import keyboard
import time
from faster_whisper import WhisperModel
from googletrans import Translator
import tkinter as tk
from tkinter import scrolledtext
import threading

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
cumulative_translated_text = ""  # Biến toàn cục lưu tất cả text đã dịch
translator = Translator()

# Lock for thread-safe operations
text_lock = threading.Lock()


# Setup Tkinter GUI
class TextDisplay:
    def __init__(self, root):
        self.root = root
        self.root.title("Cumulative Text and Translation Display")
        self.root.geometry("1200x600")  # Increased width for side-by-side display

        # Create a PanedWindow to hold two ScrolledText widgets side by side
        paned_window = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5)
        paned_window.pack(fill=tk.BOTH, expand=1)

        # Frame for Cumulative Text
        frame1 = tk.Frame(paned_window)
        paned_window.add(frame1)

        # Label for Cumulative Text
        label1 = tk.Label(frame1, text="Cumulative Text", font=("Helvetica", 16, "bold"))
        label1.pack(pady=10)

        # ScrolledText widget for Cumulative Text
        self.text_area1 = scrolledtext.ScrolledText(frame1, wrap=tk.WORD, font=("Helvetica", 14))
        self.text_area1.pack(expand=True, fill='both')
        self.text_area1.configure(state='disabled')  # Make it read-only

        # Frame for Translated Text
        frame2 = tk.Frame(paned_window)
        paned_window.add(frame2)

        # Label for Translated Text
        label2 = tk.Label(frame2, text="Translated Text (Vietnamese)", font=("Helvetica", 16, "bold"))
        label2.pack(pady=10)

        # ScrolledText widget for Translated Text
        self.text_area2 = scrolledtext.ScrolledText(frame2, wrap=tk.WORD, font=("Helvetica", 14))
        self.text_area2.pack(expand=True, fill='both')
        self.text_area2.configure(state='disabled')  # Make it read-only

    def update_texts(self, new_text, new_translated_text):
        # Update Cumulative Text
        self.text_area1.configure(state='normal')
        self.text_area1.delete(1.0, tk.END)  # Clear existing text
        self.text_area1.insert(tk.END, new_text)
        self.text_area1.configure(state='disabled')

        # Update Translated Text
        self.text_area2.configure(state='normal')
        self.text_area2.delete(1.0, tk.END)  # Clear existing text
        self.text_area2.insert(tk.END, new_translated_text)
        self.text_area2.configure(state='disabled')


def gui_thread(app):
    root = tk.Tk()
    app_instance = app(root)

    # Periodically update the text areas every 500 ms
    def periodic_update():
        with text_lock:
            app_instance.update_texts(cumulative_text, cumulative_translated_text)
        root.after(500, periodic_update)

    periodic_update()
    root.mainloop()


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
        time.sleep(0.01)  # Small delay to prevent high CPU usage

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


def main(app):
    global cumulative_text, cumulative_translated_text
    print("Nhấn và giữ phím Space để ghi âm, nhả ra để chuyển voice sang text.")
    print("Nhấn 'q' để thoát.")

    while True:
        # Chờ cho đến khi Space được nhấn
        while True:
            if keyboard.is_pressed('space'):
                break
            if keyboard.is_pressed('q'):
                return
            time.sleep(0.01)  # Small delay to prevent high CPU usage

        # Ghi âm cho đến khi nhả Space
        audio_file = record_audio()
        if audio_file is None:
            # Không có âm thanh thu được
            print("Không có âm thanh!")
        else:
            # Sau khi ghi âm xong, transcribe
            transcribed_text = transcribe_audio(audio_file)
            # Thêm text này vào cumulative_text
            with text_lock:
                cumulative_text += " " + transcribed_text if cumulative_text else transcribed_text

                # Dịch cumulative_text sang tiếng Việt
                translated = translator.translate(cumulative_text, dest='vi')
                cumulative_translated_text = translated.text

                # Debug prints (optional)
                print("\n[Cumulative Text]:", cumulative_text)
                print("[Cumulative Text Dịch]:", cumulative_translated_text)
                print("\nNhấn và giữ Space để ghi tiếp, hoặc nhấn 'q' để thoát.\n")

        if keyboard.is_pressed('q'):
            break


if __name__ == "__main__":
    # Initialize the GUI
    app = TextDisplay
    gui = threading.Thread(target=gui_thread, args=(app,), daemon=True)
    gui.start()

    # Run the main loop
    try:
        main(app)
    except KeyboardInterrupt:
        pass
    finally:
        p.terminate()
