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

# Global cache for models and tokenizers
_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}

# Register model folders
sesame_path = os.path.join(folder_paths.models_dir, "sesame")
folder_paths.add_model_folder_path("sesame", sesame_path)

tokenizer_path = os.path.join(folder_paths.models_dir, "sesame_tokenizer")
folder_paths.add_model_folder_path("sesame_tokenizer", tokenizer_path)

# Helper functions for LLaMA models
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
        self.sample_rate = self._feature_extractor.sampling_rate  # Usually 24 kHz
        print(f"Mimi loaded via transformers from kyutai/mimi (sample rate: {self.sample_rate} Hz)")
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
            raise RuntimeError(f"Failed to load tokenizer from {tokenizer_path}: {str(e)}")

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
        audio = audio.to(self.device)
        inputs = self._feature_extractor(raw_audio=audio.cpu().numpy(), sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)
        with torch.no_grad():
            encoder_outputs = self._audio_tokenizer.encode(inputs["input_values"])
            audio_tokens = encoder_outputs.audio_codes.squeeze(0)  # [num_codebooks, T]
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
    def generate(self, text: str, speaker: int, context: List[Segment], max_audio_length_ms: float = 90000, temperature: float = 0.9, topk: int = 50) -> torch.Tensor:
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
            raise ValueError(f"Input too long, must be less than {max_seq_len}")
        for _ in range(max_audio_frames):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break
            samples.append(sample)
            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1
        audio_tokens = torch.stack(samples).permute(1, 2, 0)  # [B, num_codebooks, T]
        with torch.no_grad():
            dummy_audio = torch.zeros(max_audio_frames * (self.sample_rate // 80), dtype=torch.float32)
            inputs = self._feature_extractor(raw_audio=dummy_audio.cpu().numpy(), sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)
            audio = self._audio_tokenizer.decode(audio_tokens, inputs["padding_mask"])[0]
        
        # Convert audio to [1, 1, samples] format (3D tensor)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)  # [1, 1, samples]
        return audio

def load_csm_1b(ckpt_path: str, tokenizer_path: str, device: str = "cuda") -> 'Generator':
    if not os.path.isfile(ckpt_path):
        raise ValueError(f"Checkpoint path {ckpt_path} is not a file!")
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

# Nodes
class LoadCSMCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):
        checkpoint_files = folder_paths.get_filename_list("sesame") or ["ckpt.pt"]
        return {"required": {"checkpoint": (checkpoint_files,)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "Audio/CSM"

    def load_checkpoint(self, checkpoint):
        model_path = folder_paths.get_full_path("sesame", checkpoint)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint} not found in ComfyUI/models/sesame/")
        print(f"Loading checkpoint from: {model_path}")
        return (model_path,)

class LoadCSMTokenizer:
    @classmethod
    def INPUT_TYPES(cls):
        tokenizer_dirs = folder_paths.get_folder_paths("sesame_tokenizer")
        available_dirs = [d for d in os.listdir(tokenizer_dirs[0]) if os.path.isdir(os.path.join(tokenizer_dirs[0], d))] if tokenizer_dirs and os.path.exists(tokenizer_dirs[0]) else []
        if not available_dirs:
            raise FileNotFoundError("No tokenizer directories found in ComfyUI/models/sesame_tokenizer/")
        print(f"Found tokenizer directories: {available_dirs}")
        return {"required": {"tokenizer_dir": (available_dirs, {"default": available_dirs[0]})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tokenizer_path",)
    FUNCTION = "load_tokenizer"
    CATEGORY = "Audio/CSM"

    def load_tokenizer(self, tokenizer_dir):
        base_path = folder_paths.get_folder_paths("sesame_tokenizer")[0]
        full_path = os.path.join(base_path, tokenizer_dir)
        if not os.path.exists(full_path) or not os.path.isdir(full_path):
            raise FileNotFoundError(f"Tokenizer directory {full_path} not found")
        print(f"Loading tokenizer from: {full_path}")
        return (full_path,)

class LoadAudioFile:
    """Node to load an audio file and convert it to 24kHz mono"""
    @classmethod
    def INPUT_TYPES(cls):
        # Get input folder for ComfyUI
        input_folder = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
        
        return {
            "required": {
                "audio_file": (sorted(files),),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load_audio"
    CATEGORY = "Audio/CSM"

    def load_audio(self, audio_file):
        input_folder = folder_paths.get_input_directory()
        audio_path = os.path.join(input_folder, audio_file)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print(f"Converted stereo to mono")
        
        # Resample to 24kHz if needed
        target_sample_rate = 24000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
            print(f"Resampled from {sample_rate}Hz to {target_sample_rate}Hz")
            sample_rate = target_sample_rate
        
        # Ensure proper shape [1, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        print(f"Loaded audio: {audio_file}, shape: {waveform.shape}, sample_rate: {sample_rate}Hz")
        
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)

class CreateConversationContext:
    """Node to create a conversation context entry from audio and text"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "text": ("STRING", {"multiline": True}),
                "speaker": ("INT", {"default": 0, "min": 0, "max": 1}),
            },
            "optional": {
                "previous_context": ("CSM_CONTEXT",),
            }
        }

    RETURN_TYPES = ("CSM_CONTEXT",)
    RETURN_NAMES = ("context",)
    FUNCTION = "create_context"
    CATEGORY = "Audio/CSM"

    def create_context(self, audio, text, speaker, previous_context=None):
        # Extract waveform from audio dict
        waveform = audio["waveform"]
        
        # Remove batch and channel dimensions to get [samples]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0).squeeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        
        # Create segment
        segment = Segment(speaker=speaker, text=text, audio=waveform)
        
        # Add to existing context or create new list
        if previous_context is not None:
            context = previous_context + [segment]
        else:
            context = [segment]
        
        print(f"Created context entry: speaker={speaker}, text='{text[:50]}...', audio_shape={waveform.shape}")
        print(f"Total context entries: {len(context)}")
        
        return (context,)

class CSMTextToSpeechWithContext:
    """Enhanced CSM TTS node with conversation history support"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "model_path": ("STRING",),
                "tokenizer_path": ("STRING",),
                "sample_rate": ("INT", {"default": 24000, "min": 8000, "max": 48000}),
            },
            "optional": {
                "context": ("CSM_CONTEXT",),
                "speaker": ("INT", {"default": 0, "min": 0, "max": 1}),
                "max_audio_length_ms": ("FLOAT", {"default": 90000.0, "min": 1000.0, "max": 300000.0}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0}),
                "topk": ("INT", {"default": 50, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "Audio/CSM"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Return a random value to force re-execution every time
        import random
        return random.random()

    def generate_speech(self, text, model_path, tokenizer_path, sample_rate, context=None, speaker=0, max_audio_length_ms=90000.0, temperature=0.9, topk=50):
        if not model_path:
            raise ValueError("Model path is empty! Connect LoadCSMCheckpoint.")
        if not os.path.isfile(model_path):
            raise ValueError(f"model_path {model_path} is not a file!")
        if not tokenizer_path:
            raise ValueError("Tokenizer path is empty! Connect LoadCSMTokenizer.")
        if not os.path.isdir(tokenizer_path):
            raise ValueError(f"tokenizer_path {tokenizer_path} is not a directory!")
        if not text:
            raise ValueError("Text is empty!")

        # Model caching
        global _MODEL_CACHE, _TOKENIZER_CACHE
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model if not cached
        if model_path not in _MODEL_CACHE:
            print(f"Loading model from {model_path}")
            _MODEL_CACHE[model_path] = load_csm_1b(model_path, tokenizer_path, device=device)
        generator = _MODEL_CACHE[model_path]

        # Load tokenizer if not cached
        if tokenizer_path not in _TOKENIZER_CACHE:
            print(f"Loading tokenizer from {tokenizer_path}")
            _TOKENIZER_CACHE[tokenizer_path] = generator._text_tokenizer

        # Use context if provided, otherwise empty list
        conversation_context = context if context is not None else []
        
        if conversation_context:
            print(f"Using conversation context with {len(conversation_context)} entries")
            for idx, seg in enumerate(conversation_context):
                print(f"  Entry {idx}: speaker={seg.speaker}, text='{seg.text[:50]}...', audio_shape={seg.audio.shape}")
        else:
            print("No conversation context provided - generating without voice cloning")

        # Generate audio
        audio = generator.generate(
            text=text,
            speaker=speaker,
            context=conversation_context,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk
        )
        print(f"Generated audio shape: {audio.shape}")

        # Ensure proper shape [1, 1, samples]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)

        return ({"waveform": audio, "sample_rate": sample_rate},)

# Node registration
NODE_CLASS_MAPPINGS = {
    "LoadCSMCheckpoint": LoadCSMCheckpoint,
    "LoadCSMTokenizer": LoadCSMTokenizer,
    "LoadAudioFile": LoadAudioFile,
    "CreateConversationContext": CreateConversationContext,
    "CSMTextToSpeechWithContext": CSMTextToSpeechWithContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadCSMCheckpoint": "Load CSM Checkpoint",
    "LoadCSMTokenizer": "Load CSM Tokenizer",
    "LoadAudioFile": "Load Audio File (24kHz Mono)",
    "CreateConversationContext": "Create Conversation Context",
    "CSMTextToSpeechWithContext": "CSM Text-to-Speech (with Context)",
}

print("Registered enhanced CSM nodes with conversation context support")
