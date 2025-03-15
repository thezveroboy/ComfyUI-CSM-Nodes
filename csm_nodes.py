# custom_nodes/csm_nodes.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io.wavfile import write as wav_write

# Полная реализация класса CSM
class CSM(nn.Module):
    def __init__(self, vocab_size=256, hidden_dim=512, audio_dim=80, sample_rate=22050):
        super(CSM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.audio_dim = audio_dim
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, audio_dim)
        )
        self.to(self.device)

    def load(self, model_path):
        """Загрузка весов модели"""
        if not os.path.exists(model_path):
            raise ValueError(f"Модель по пути {model_path} не найдена!")
        state_dict = torch.load(model_path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.eval()
        return self

    def text_to_indices(self, text):
        """Преобразование текста в индексы"""
        return torch.tensor([ord(c) % self.vocab_size for c in text], dtype=torch.long).to(self.device)

    def generate(self, text, speaker_id="default", sample_rate=22050):
        """Генерация аудио из текста"""
        with torch.no_grad():
            input_ids = self.text_to_indices(text).unsqueeze(0)
            embedded = self.embedding(input_ids)
            encoded, _ = self.encoder(embedded)
            mel_spec = self.decoder(encoded)
            audio = self.mel_to_audio(mel_spec, sample_rate)
            return audio.squeeze(0)

    def mel_to_audio(self, mel_spec, sample_rate):
        """Упрощённый вокодер"""
        mel_spec = mel_spec.transpose(1, 2)
        audio = torch.tanh(mel_spec.mean(dim=1)) * 32767
        return audio.to(torch.int16)

# Узел для ComfyUI
class CSMTextToSpeech:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Привет, мир!", "multiline": True}),
                "model_path": ("STRING", {"default": "./ckpt.pt"}),  # Указано для Hugging Face
                "output_dir": ("STRING", {"default": "./output"}),
                "sample_rate": ("INT", {"default": 22050, "min": 8000, "max": 48000}),
            },
            "optional": {
                "speaker_id": ("STRING", {"default": "default"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "generate_speech"
    CATEGORY = "Audio/CSM"

    def generate_speech(self, text, model_path, output_dir, sample_rate, speaker_id="default"):
        model = CSM(vocab_size=256, hidden_dim=512, audio_dim=80, sample_rate=sample_rate)
        model = model.load(model_path)
        audio_data = model.generate(text, speaker_id, sample_rate)
        audio_data = audio_data.cpu().numpy()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "output.wav")
        wav_write(output_path, sample_rate, audio_data)
        return (output_path,)

# Регистрация ноды
NODE_CLASS_MAPPINGS = {"CSMTextToSpeech": CSMTextToSpeech}
NODE_DISPLAY_NAME_MAPPINGS = {"CSMTextToSpeech": "CSM Text-to-Speech"}
