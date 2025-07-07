# Fully Offline Whisper-based Russian-to-English Real-time Translator

import sounddevice as sd
import numpy as np
import webrtcvad
import whisper
import queue
import threading
import time
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import scrolledtext, messagebox
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# Default Config
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
CHANNELS = 1
VAD_MODE = 2
MAX_SEGMENT_DURATION = 30  # seconds
MODEL_SIZE = "small"
SOURCE_LANG = "ru"
TARGET_LANG = "en"

DEVICE_INDEX = None

# Load Whisper and translation models
vad = webrtcvad.Vad(VAD_MODE)
model = whisper.load_model(MODEL_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
translator_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
translator_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(device)

def offline_translate(text, src_lang, tgt_lang):
    translator_tokenizer.src_lang = src_lang
    encoded = translator_tokenizer(text, return_tensors="pt").to(device)
    generated = translator_model.generate(**encoded, forced_bos_token_id=translator_tokenizer.get_lang_id(tgt_lang))
    return translator_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

audio_q = queue.Queue()
text_queue = queue.Queue()
running_flag = threading.Event()

stats = {
    "segments": 0,
    "last_duration": 0,
    "last_length": 0,
    "last_volume": 0.0,
}

def list_devices():
    return [(i, d['name']) for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]

def frame_generator(stream):
    while running_flag.is_set():
        data, _ = stream.read(FRAME_SIZE)
        yield data[:, 0].copy()

def is_speech(frame):
    pcm = (frame * 32768).astype(np.int16).tobytes()
    return vad.is_speech(pcm, SAMPLE_RATE)

def record_segments(progress_callback):
    global DEVICE_INDEX
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32',
                        blocksize=FRAME_SIZE, device=DEVICE_INDEX) as stream:
        frames = []
        speech_started = False
        silence_count = 0
        max_frames = int(MAX_SEGMENT_DURATION * 1000 / FRAME_DURATION)
        start_time = time.time()

        for i, frame in enumerate(frame_generator(stream)):
            if not running_flag.is_set():
                break

            if is_speech(frame):
                frames.append(frame)
                speech_started = True
                silence_count = 0
            elif speech_started:
                silence_count += 1
                frames.append(frame)
                if silence_count > 10:
                    break

            if len(frames) > max_frames:
                break

            progress = int((len(frames) / max_frames) * 100)
            progress_callback(progress)

        if frames:
            duration = time.time() - start_time
            volume = np.mean(np.abs(np.concatenate(frames)))
            stats["last_volume"] = volume
            stats["last_duration"] = round(duration, 2)
            audio = np.concatenate(frames)
            audio_q.put(audio)
        progress_callback(0)

def transcribe_translate():
    while running_flag.is_set():
        try:
            audio = audio_q.get(timeout=1)
        except queue.Empty:
            continue
        try:
            result = model.transcribe(audio, language=SOURCE_LANG, task='transcribe', fp16=False)
            ru = result['text'].strip()
            if ru:
                en = offline_translate(ru, SOURCE_LANG, TARGET_LANG)
                stats["segments"] += 1
                stats["last_length"] = len(ru)
                text_queue.put((ru, en))
        except Exception as e:
            text_queue.put((f"[Error] {str(e)}", ""))

def build_gui():
    global DEVICE_INDEX, vad, model, translator_tokenizer, translator_model

    app = tb.Window(themename="darkly")
    app.title("🎙️ Offline Russian-English Translator")
    app.geometry("1100x800")

    device_frame = tb.Frame(app)
    device_frame.pack(pady=5)
    tb.Label(device_frame, text="🎧 Input Device:").pack(side=LEFT, padx=(0, 5))
    device_list = list_devices()
    device_var = tb.StringVar()
    device_combo = tb.Combobox(device_frame, values=[f"{i}: {name}" for i, name in device_list],
                               textvariable=device_var, width=60)
    device_combo.pack(side=LEFT)

    status_var = tb.StringVar(value="🟢 Idle")
    status_label = tb.Label(app, textvariable=status_var, bootstyle=INFO, font=("Helvetica", 12))
    status_label.pack(pady=(5, 0))

    progress = tb.IntVar(value=0)
    progress_bar = tb.Progressbar(app, variable=progress, maximum=100, length=500)
    progress_bar.pack(pady=(5, 10))

    text_frame = tb.Frame(app)
    text_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
    ru_box = scrolledtext.ScrolledText(text_frame, height=12, width=70, font=("Consolas", 10))
    en_box = scrolledtext.ScrolledText(text_frame, height=12, width=70, font=("Consolas", 10))
    ru_box.pack(side=LEFT, expand=True, fill=BOTH, padx=(0, 5))
    en_box.pack(side=LEFT, expand=True, fill=BOTH)
    ru_box.insert('end', "🇷🇺 Source Language\n")
    en_box.insert('end', "🌍 Translated Language\n")
    ru_box.config(state='disabled')
    en_box.config(state='disabled')

    stats_frame = tb.Labelframe(app, text="📊 Translation Stats")
    stats_frame.pack(fill=X, padx=10, pady=(0, 10))
    seg_label = tb.Label(stats_frame, text="Total Segments: 0", font=("Consolas", 10))
    dur_label = tb.Label(stats_frame, text="Last Duration: 0.0 sec", font=("Consolas", 10))
    len_label = tb.Label(stats_frame, text="Last Transcript Length: 0 chars", font=("Consolas", 10))
    vol_label = tb.Label(stats_frame, text="Estimated Clarity: Good", font=("Consolas", 10))
    seg_label.pack(anchor=W)
    dur_label.pack(anchor=W)
    len_label.pack(anchor=W)
    vol_label.pack(anchor=W)

    def record_loop():
        while running_flag.is_set():
            record_segments(progress.set)

    def start():
        global DEVICE_INDEX
        if not device_var.get():
            status_var.set("❌ Please select a device")
            return
        DEVICE_INDEX = int(device_var.get().split(":")[0])
        running_flag.set()
        status_var.set("🎤 Listening...")
        threading.Thread(target=transcribe_translate, daemon=True).start()
        threading.Thread(target=record_loop, daemon=True).start()

    def stop():
        running_flag.clear()
        status_var.set("⏹️ Stopped")

    button_frame = tb.Frame(app)
    button_frame.pack(pady=10)
    tb.Button(button_frame, text="▶ Start", command=start, bootstyle=SUCCESS).pack(side=LEFT, padx=10)
    tb.Button(button_frame, text="⏹ Stop", command=stop, bootstyle=DANGER).pack(side=LEFT, padx=10)

    def update_gui():
        while not text_queue.empty():
            ru, en = text_queue.get_nowait()
            ru_box.config(state='normal')
            en_box.config(state='normal')
            ru_box.insert('end', ru + "\n")
            en_box.insert('end', en + "\n")
            ru_box.see('end')
            en_box.see('end')
            ru_box.config(state='disabled')
            en_box.config(state='disabled')

        seg_label.config(text=f"Total Segments: {stats['segments']}")
        dur_label.config(text=f"Last Duration: {stats['last_duration']} sec")
        len_label.config(text=f"Last Transcript Length: {stats['last_length']} chars")
        clarity = "Poor" if stats['last_volume'] < 0.01 else "Good" if stats['last_volume'] > 0.03 else "Fair"
        vol_label.config(text=f"Estimated Clarity: {clarity}")

        app.after(500, update_gui)

    update_gui()
    app.mainloop()

if __name__ == "__main__":
    build_gui()
