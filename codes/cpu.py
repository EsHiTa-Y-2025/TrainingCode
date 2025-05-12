#!/usr/bin/env python3
import os, sys

# 1) Point to your local repos
sys.path.insert(0, os.path.expanduser("~/TrainingCode/ultravox/ultravox/model"))
sys.path.insert(0, os.path.expanduser("~/TrainingCode/csm"))

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from moshi.models import loaders
from ultravox_model import UltravoxModel, UltravoxConfig
from models import Model as CSMModel, ModelArgs   # from your csm repo

class Audio2CodebooksCPU(nn.Module):
    def __init__(
        self,
        ul_cfg_or_path: str,
        csm_repo: str,
        *,
        text_vocab_size: int,
        audio_vocab_size: int,
        audio_num_codebooks: int,
        csm_backbone_flavor: str = "llama-1B",
        csm_decoder_flavor: str = "llama-100M",
    ):
        super().__init__()

        # --- 1) MIMI codec & feature extractor ---
        mimi_weight = hf_hub_download(
            repo_id=loaders.DEFAULT_REPO,
            filename=loaders.MIMI_NAME,
        )
        self.mimi = loaders.get_mimi(mimi_weight, device=torch.device('cpu'))
        self.mimi.set_num_codebooks(audio_num_codebooks)

        # --- 2) UltravoxModel (LLM + projector) ---
        self.uv_cfg = UltravoxConfig.from_pretrained(ul_cfg_or_path)
        self.ul     = UltravoxModel.from_pretrained(
                          ul_cfg_or_path,
                          config=self.uv_cfg,
                          load_in_8bit=False,
                          device_map=None,
                      )
        self.ul.tie_weights()
        self.projector      = self.ul.multi_modal_projector
        self.language_model = self.ul.language_model

        # --- 3) CSM decoder (with both flavors) ---
        args = ModelArgs(
            backbone_flavor     = csm_backbone_flavor,
            decoder_flavor      = csm_decoder_flavor,
            text_vocab_size     = text_vocab_size,
            audio_vocab_size    = audio_vocab_size,
            audio_num_codebooks = audio_num_codebooks,
        )
        self.csm = CSMModel.from_pretrained(csm_repo)
        self.csm.backbone = self.language_model

        # --- 4) Tokenizer & caches (optional) ---
        # self.tok = AutoTokenizer.from_pretrained(tokenizer_id)
        # self.csm.setup_caches(max_batch_size=1)
        # self.ul.reset_caches()

        # --- 5) Freeze everything ---
        for p in self.parameters():
            p.requires_grad = False

        # gating layers for fusion
        D = self.ul.config.hidden_size
        self.gate = nn.Linear(2*D, D)
        self.sig  = nn.Sigmoid()

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        # ensure shape (B, 1, L)
        wav = raw_audio.unsqueeze(1) if raw_audio.dim() == 2 else raw_audio

        # === 1) MIMI encode → semantic + acoustic codes ===

        # (a) CNN encoder → (B, hidden, T1)
        cnn_emb = self.mimi.encoder(wav)

        # (b) Transformer encoder → (B, T1, hidden)
        trans_out = self.mimi.encoder_transformer(
            cnn_emb.transpose(1, 2),
            return_dict=True
        )
        sem = trans_out.last_hidden_state                   # (B, T1, hidden)

        # (c) to channel-first for downsample: (B, hidden, T1)
        sem_cf = sem.transpose(1, 2)

        # (d) optional downsample → (B, hidden, T2)
        if self.mimi.downsample is not None:
            quant_in = self.mimi.downsample(sem_cf)
        else:
            quant_in = sem_cf

        # (e) quantize with all codebooks → (Q, B, T2)
        all_codes = self.mimi.quantizer.encode(
            quant_in,
            self.mimi.config.num_quantizers
        )

        # (f) drop first codebook → (Q-1, B, T2)
        acoustic_codes = all_codes[1:]

        # (g) permute to (B, Q-1, T2)
        codes = acoustic_codes.permute(1, 0, 2)

        # sem is (B, T1, hidden), codes is (B, Q-1, T2)

        # === 2) Acoustic embedding → (B, T2, hidden) ===
        flat   = codes.reshape(-1)
        emb_ac = self.csm.audio_embeddings(flat).view(*codes.shape, -1)
        ac     = emb_ac.mean(dim=2)                   # (B, Q-1, hidden) → mean→ (B, hidden)

        # but we need (B, T1, hidden) to match sem: up/down-sample ac to T1 if needed
        # here we simply unsqueeze time dim to broadcast
        ac = ac.unsqueeze(1).expand(-1, sem.size(1), -1)  # (B, T1, hidden)

        # === 3) Gating fusion ===
        cat   = torch.cat([sem, ac], dim=-1)         # (B, T1, 2*hidden)
        g     = self.sig(self.gate(cat))            # (B, T1, hidden)
        fused = g * sem + (1 - g) * ac              # (B, T1, hidden)

        # === 4) Project to LLM space & 5) Ultravox LLM forward ===
        proj = self.projector(fused)
        with torch.no_grad():
            out = self.language_model(
                inputs_embeds=proj,
                return_dict=True,
                output_hidden_states=True,
            )
        h = out.hidden_states[-1]                  # (B, T1, hidden)

        # === 6) CSM generate audio codes & 7) Decode to waveform ===
        B = h.size(0)
        tokens = torch.zeros(
            B, 1, self.csm.config.audio_num_codebooks + 1,
            dtype=torch.long, device=h.device
        )
        mask  = torch.ones_like(tokens, dtype=torch.float)
        pos   = torch.zeros(B, 1, dtype=torch.long, device=h.device)
        frame_codes = self.csm.generate_frame(
            tokens=tokens,
            tokens_mask=mask,
            input_pos=pos,
            temperature=1.0,
            topk=5,
        ).view(B, -1, self.csm.config.audio_num_codebooks)

        with torch.no_grad():
            waveform = self.mimi.decode(frame_codes)

        return waveform

if __name__ == "__main__":
    # Initialize on CPU
    device = torch.device('cpu')
    model = Audio2CodebooksCPU(
        ul_cfg_or_path       = "fixie-ai/ultravox-v0_2",
        csm_repo             = "sesame/csm-1b",
        text_vocab_size      = 32000,
        audio_vocab_size     = 1024,
        audio_num_codebooks  = 32,
    )

    # Inference on first 5 samples of Expresso
    ds = load_dataset("ylacombe/expresso", split="train[:5]")
    print(f"Loaded {len(ds)} examples for inference.")

    from itertools import islice
    for i, ex in enumerate(islice(ds, 5)):
        raw = torch.tensor(ex["audio"]["array"], dtype=torch.float32).unsqueeze(0)
        out = model(raw)
        print(f"Example {i+1}: output shape = {out.shape}")

    print("\nDone.")
