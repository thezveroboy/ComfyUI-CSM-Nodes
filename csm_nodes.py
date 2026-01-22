import os
import torch
import torch.nn as nn
import torchaudio
from dataclasses import dataclass
from typing import List, Tuple, Optional
import folder_paths
from transformers import AutoTokenizer, MimiModel, AutoFeatureExtractor
from tokenizers.processors import TemplateProcessing
import torchtune
from torchtune.models import llama3_2

# Глобальный кеш для моделей и токенизаторов
_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}

# Регистрация папок для моделей
sesame_path = os.path.join(folder_paths.models_dir, "sesame")
folder_paths.add_model_folder_path("sesame", sesame_path)

tokenizer_path = os.path.join(folder_paths.models_dir, "sesame_tokenizer")
folder_paths.add_model_folder_path("sesame_tokenizer", tokenizer_path)

audio_path = os.path.join(folder_paths.models_dir, "audio")
folder_paths.add_model_folder_path("audio", audio_path)  # Регистрация папки audio

# Вспомогательные функции для моделей LLaMA
def llama3_2_1B() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )

def llama3_2_100M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )

FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-100M": llama3_2_100M,
}

def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim

def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    return mask[input_pos, :]

def _multinomial_sample_one_no_sync(probs):
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature
    filter_value = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)
    return _multinomial_sample_one_no_sync(probs)

@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[args.backbone_flavor]())
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[args.decoder_flavor]())
        self.text_embeddings = nn.Embedding(args.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(args.audio_vocab_size * args.audio_num_codebooks, backbone_dim)
        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, args.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(args.audio_num_codebooks - 1, decoder_dim, args.audio_vocab_size))

    def setup_caches(self, max_batch_size: int):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        with device:
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=self.args.audio_num_codebooks)
        self.register_buffer("backbone_causal_mask", _create_causal_mask(self.backbone.max_seq_len, device))
        self.register_buffer("decoder_causal_mask", _create_causal_mask(self.args.audio_num_codebooks, device))

    def generate_frame(self, tokens: torch.Tensor, tokens_mask: torch.Tensor, input_pos: torch.Tensor, temperature: float, topk: int) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()
        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)
        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)
        self.decoder.reset_caches()
        for i in range(1, self.args.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask).to(dtype=dtype)
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)
            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1
        return curr_sample

    def reset_caches(self):
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.args.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)
        audio_tokens = tokens[:, :, :-1] + (
            self.args.audio_vocab_size * torch.arange(self.args.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.args.audio_num_codebooks, -1
        )
        return torch.cat([audio_embeds, text_embeds], dim=-2)

@dataclass
class Segment:
    speaker: int
    text: str
    audio: torch.Tensor

class Generator:
    def __init__(self, model: Model, tokenizer_path: str):
        self._model = model
        self._model.setup_caches(1)
        self._text_tokenizer = self.load_local_tokenizer(tokenizer_path)
        device = next(model.parameters()).device
        self._audio_tokenizer = MimiModel.from_pretrained("kyutai/mimi").to(device)
        self._feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        self.sample_rate = self._feature_extractor.sampling_rate  # Обычно 24 кГц
        print("Mimi загружен через transformers из kyutai/mimi")
        self.device = device

    def load_local_tokenizer(self, tokenizer_path: str):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            bos = tokenizer.bos_token
            eos = tokenizer.eos_token
            tokenizer._tokenizer.post_processor = TemplateProcessing(
                single=f"{bos}:0 $A:0 {eos}:0",
                pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
                special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
            )
            return tokenizer
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить токенизатор из {tokenizer_path}: {str(e)}")

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True
        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))
        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if audio.dim() == 3 and audio.size(1) == 2:
            audio = audio.mean(dim=1, keepdim=True)
        elif audio.dim() == 2 and audio.size(0) == 2:
            audio = audio.mean(dim=0, keepdim=True)

        if audio.dim() == 3 and audio.size(1) != 1:
            raise ValueError(f"Ожидается моно-аудио, но получено {audio.size(1)} каналов")
        elif audio.dim() == 2 and audio.size(0) != 1:
            raise ValueError(f"Ожидается моно-аудио, но получено {audio.size(0)} каналов")

        if audio.dim() == 3:
            audio = audio.squeeze(0).squeeze(0)
        elif audio.dim() == 2:
            audio = audio.squeeze(0)

        audio = audio.to(self.device)
        inputs = self._feature_extractor(
            raw_audio=audio.cpu().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            encoder_outputs = self._audio_tokenizer.encode(inputs["input_values"])
            audio_tokens = encoder_outputs.audio_codes.squeeze(0)

        frame_tokens = []
        frame_masks = []
        eos_frame = torch.zeros(audio_tokens.size(0), 1, dtype=torch.long).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)
        audio_frame = torch.zeros(audio_tokens.size(1), 33, dtype=torch.long).to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33, dtype=torch.bool).to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True
        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)
        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(self, text: str, speaker: int, context: List[Segment], max_audio_length_ms: float = 90000, temperature: float = 0.5, topk: int = 10) -> torch.Tensor:
        self._model.reset_caches()
        max_audio_frames = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)
        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)
        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        max_seq_len = 2048 - max_audio_frames
        if curr_tokens.size(1) >= max_seq_len:
            raise ValueError(f"Входные данные слишком длинные, должны быть меньше {max_seq_len}")
        for _ in range(max_audio_frames):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break
            samples.append(sample)
            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1
        audio_tokens = torch.stack(samples).permute(1, 2, 0)
        with torch.no_grad():
            dummy_audio = torch.zeros(max_audio_frames * (self.sample_rate // 80), dtype=torch.float32)
            inputs = self._feature_extractor(raw_audio=dummy_audio.cpu().numpy(), sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)
            audio = self._audio_tokenizer.decode(audio_tokens, inputs["padding_mask"])[0]
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)
        return audio

def load_csm_1b(ckpt_path: str, tokenizer_path: str, device: str = "cuda") -> 'Generator':
    if not os.path.isfile(ckpt_path):
        raise ValueError(f"Путь к чекпоинту {ckpt_path} не является файлом!")
    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    model = Model(model_args).to(device=device, dtype=torch.bfloat16)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return Generator(model, tokenizer_path)

# Ноды
class LoadCSMCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):
        checkpoint_files = folder_paths.get_filename_list("sesame") or ["ckpt.pt"]
        return {"required": {"checkpoint": (checkpoint_files,)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "zveroboy/CSM"  # Изменено с Audio на zveroboy

    def load_checkpoint(self, checkpoint):
        model_path = folder_paths.get_full_path("sesame", checkpoint)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Чекпоинт {checkpoint} не найден в ComfyUI/models/sesame/")
        print(f"Загрузка чекпоинта из: {model_path}")
        return (model_path,)

class LoadCSMTokenizer:
    @classmethod
    def INPUT_TYPES(cls):
        tokenizer_dirs = folder_paths.get_folder_paths("sesame_tokenizer")
        available_dirs = [d for d in os.listdir(tokenizer_dirs[0]) if os.path.isdir(os.path.join(tokenizer_dirs[0], d))] if tokenizer_dirs and os.path.exists(tokenizer_dirs[0]) else []
        if not available_dirs:
            raise FileNotFoundError("Не найдено директорий токенизаторов в ComfyUI/models/sesame_tokenizer/")
        print(f"Найденные директории токенизатора: {available_dirs}")
        return {"required": {"tokenizer_dir": (available_dirs, {"default": available_dirs[0]})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tokenizer_path",)
    FUNCTION = "load_tokenizer"
    CATEGORY = "zveroboy/CSM"  # Изменено с Audio на zveroboy

    def load_tokenizer(self, tokenizer_dir):
        base_path = folder_paths.get_folder_paths("sesame_tokenizer")[0]
        full_path = os.path.join(base_path, tokenizer_dir)
        if not os.path.exists(full_path) or not os.path.isdir(full_path):
            raise FileNotFoundError(f"Директория токенизатора {full_path} не найдена")
        print(f"Загрузка токенизатора из: {full_path}")
        return (full_path,)

class LoadAudioWithTranscription:
    @classmethod
    def INPUT_TYPES(cls):
        audio_files = folder_paths.get_filename_list("audio") or []
        if not audio_files:
            print("Предупреждение: Аудиофайлы не найдены в ComfyUI/models/audio/. Убедитесь, что папка существует и содержит файлы.")
        return {
            "required": {
                "audio_file": (audio_files if audio_files else ["No audio files found"],),
                "transcription": ("STRING", {"default": "", "multiline": True}),
                "speaker": ("INT", {"default": 0, "min": 0, "max": 10}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "INT")
    RETURN_NAMES = ("audio", "transcription", "speaker")
    FUNCTION = "load_audio"
    CATEGORY = "zveroboy/CSM"  # Изменено с Audio на zveroboy

    def load_audio(self, audio_file, transcription, speaker):
        audio_path = folder_paths.get_full_path("audio", audio_file)
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Аудиофайл {audio_file} не найден в ComfyUI/models/audio/")
        
        # Загрузка аудио
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 24000:  # Приведение к 24 кГц, как ожидает Mimi
            resampler = torchaudio.transforms.Resample(sample_rate, 24000)
            waveform = resampler(waveform)
            sample_rate = 24000
        
        # Приведение к моно, если стерео
        if waveform.size(0) == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Приведение к 3D формату [1, 1, samples]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        print(f"Загружен аудиофайл: {audio_path}, транскрипция: {transcription}, спикер: {speaker}")
        return ({"waveform": waveform, "sample_rate": sample_rate}, transcription, speaker)

class CSMTextToSpeech:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING",),
                "model_path": ("STRING",),
                "tokenizer_path": ("STRING",),
                "sample_rate": ("INT", {"default": 24000, "min": 8000, "max": 48000}),
            },
            "optional": {
                "speaker": ("INT", {"default": 0, "min": 0, "max": 10}),
                "max_audio_length_ms": ("FLOAT", {"default": 90000.0, "min": 1000.0, "max": 300000.0}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0}),
                "topk": ("INT", {"default": 100, "min": 1, "max": 200}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "zveroboy/CSM"  # Изменено с Audio на zveroboy

    def generate_speech(self, text, model_path, tokenizer_path, sample_rate, speaker=0, max_audio_length_ms=90000.0, temperature=0.7, topk=100):
        if not model_path or not os.path.isfile(model_path):
            raise ValueError(f"model_path {model_path} некорректен!")
        if not tokenizer_path or not os.path.isdir(tokenizer_path):
            raise ValueError(f"tokenizer_path {tokenizer_path} некорректен!")
        if not text:
            raise ValueError("Текст пуст!")

        global _MODEL_CACHE, _TOKENIZER_CACHE
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_path not in _MODEL_CACHE:
            print(f"Загрузка модели из {model_path}")
            _MODEL_CACHE[model_path] = load_csm_1b(model_path, tokenizer_path, device=device)
        generator = _MODEL_CACHE[model_path]

        if tokenizer_path not in _TOKENIZER_CACHE:
            print(f"Загрузка токенизатора из {tokenizer_path}")
            _TOKENIZER_CACHE[tokenizer_path] = generator._text_tokenizer

        audio = generator.generate(
            text=text,
            speaker=speaker,
            context=[],
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk
        )

        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)

        return ({"waveform": audio, "sample_rate": sample_rate},)

class CSMTextToSpeechWithContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING",),
                "model_path": ("STRING",),
                "tokenizer_path": ("STRING",),
                "sample_rate": ("INT", {"default": 24000, "min": 8000, "max": 48000}),
                "audio_context": ("AUDIO",),
                "context_transcription": ("STRING",),
                "context_speaker": ("INT", {"default": 0, "min": 0, "max": 10}),
            },
            "optional": {
                "speaker": ("INT", {"default": 0, "min": 0, "max": 10}),
                "max_audio_length_ms": ("FLOAT", {"default": 90000.0, "min": 1000.0, "max": 300000.0}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0}),
                "topk": ("INT", {"default": 100, "min": 1, "max": 200}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech_with_context"
    CATEGORY = "zveroboy/CSM"  # Изменено с Audio на zveroboy

    def generate_speech_with_context(self, text, model_path, tokenizer_path, sample_rate, audio_context, context_transcription, context_speaker, speaker=0, max_audio_length_ms=90000.0, temperature=0.7, topk=100):
        if not model_path or not os.path.isfile(model_path):
            raise ValueError(f"model_path {model_path} некорректен!")
        if not tokenizer_path or not os.path.isdir(tokenizer_path):
            raise ValueError(f"tokenizer_path {tokenizer_path} некорректен!")
        if not text:
            raise ValueError("Текст пуст!")
        if not audio_context:
            raise ValueError("Аудио-контекст пуст!")
        if not context_transcription:
            raise ValueError("Транскрипция контекста пуста! Укажите транскрипцию в LoadAudioWithTranscription.")

        global _MODEL_CACHE, _TOKENIZER_CACHE
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_path not in _MODEL_CACHE:
            print(f"Загрузка модели из {model_path}")
            _MODEL_CACHE[model_path] = load_csm_1b(model_path, tokenizer_path, device=device)
        generator = _MODEL_CACHE[model_path]

        if tokenizer_path not in _TOKENIZER_CACHE:
            print(f"Загрузка токенизатора из {tokenizer_path}")
            _TOKENIZER_CACHE[tokenizer_path] = generator._text_tokenizer

        # Подготовка аудио-контекста
        audio_tensor = audio_context["waveform"]
        input_sample_rate = audio_context["sample_rate"]
        if input_sample_rate != 24000:
            resampler = torchaudio.transforms.Resample(input_sample_rate, 24000)
            audio_tensor = resampler(audio_tensor)
        if audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)

        context_segment = Segment(
            speaker=context_speaker,
            text=context_transcription,
            audio=audio_tensor
        )

        audio = generator.generate(
            text=text,
            speaker=speaker,
            context=[context_segment],
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk
        )

        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)

        return ({"waveform": audio, "sample_rate": sample_rate},)

# Регистрация нод
NODE_CLASS_MAPPINGS = {
    "LoadCSMCheckpoint": LoadCSMCheckpoint,
    "LoadCSMTokenizer": LoadCSMTokenizer,
    "LoadAudioWithTranscription": LoadAudioWithTranscription,
    "CSMTextToSpeech": CSMTextToSpeech,
    "CSMTextToSpeechWithContext": CSMTextToSpeechWithContext
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadCSMCheckpoint": "Load CSM Checkpoint",
    "LoadCSMTokenizer": "Load CSM Tokenizer",
    "LoadAudioWithTranscription": "Load Audio with Transcription",
    "CSMTextToSpeech": "CSM Text-to-Speech",
    "CSMTextToSpeechWithContext": "CSM Text-to-Speech with Context"
}

print("Зарегистрированы ноды CSM: LoadCSMCheckpoint, LoadCSMTokenizer, LoadAudioWithTranscription, CSMTextToSpeech, CSMTextToSpeechWithContext")
