import os, sys

# Add your local repos to the path
#sys.path.insert(0, os.path.expanduser("~/TrainingCode/ultravox/model"))
#sys.path.insert(0, os.path.expanduser("~/TrainingCode/csm"))
sys.path.append("/home/eshita_24bcs10024/TrainingCode/ultravox/ultravox/model")
sys.path.append("/home/eshita_24bcs10024/TrainingCode/csm")

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
        tokenizer_id: str,
        *,
        text_vocab_size: int,
        audio_vocab_size: int,
        audio_num_codebooks: int,
        csm_decoder_flavor: str = "llama-100M",
        csm_backbone_flavor: str = "llama,
    ):
        super().__init__()

        # 1) MIMI codec & feature extractor
        mimi_weight = hf_hub_download(
            repo_id=loaders.DEFAULT_REPO,
            filename=loaders.MIMI_NAME,
        )
        self.mimi = loaders.get_mimi(mimi_weight, device=torch.device('cpu'))
        self.mimi.set_num_codebooks(audio_num_codebooks)

        # 2) UltravoxModel (loads both LLM and its projector)
        self.uv_cfg = UltravoxConfig.from_pretrained(ul_cfg_or_path)
        self.ul = UltravoxModel.from_pretrained(
            ul_cfg_or_path,
            config=self.uv_cfg,
            load_in_8bit=False,
            device_map=None,
        )
        self.ul.tie_weights()
        # extract the projector and language_model submodules
#        self.projector      = self.ul.projector
#        self.language_model = self.ul.language_model
        self.projector      = self.ul.multi_modal_projector
        self.language_model = self.ul.language_model


        # 3) CSM decoder
        args = ModelArgs(
            backbone_flavor=csm_decoder_flavor,
            decoder_flavor=csm_decoder_flavor,
            text_vocab_size=text_vocab_size,
            audio_vocab_size=audio_vocab_size,
            audio_num_codebooks=audio_num_codebooks,
        )
        self.csm = CSMModel.from_pretrained(csm_repo, config=args)
        # swap in the Ultravox LLM as backbone
        self.csm.backbone = self.language_model

        uv_dim  = self.uv_cfg.text_config.hidden_size
        dec_dim = self.csm.decoder.tok_embeddings.embedding_dim
        self.csm.projection = nn.Linear(uv_dim, dec_dim, bias=False)
        self.csm.tie_weights()

        # tokenizer & caches
        self.tok = AutoTokenizer.from_pretrained(tokenizer_id)
        self.csm.setup_caches(max_batch_size=1)
        self.ul.reset_caches()

        # 4) Freeze *all* parameters
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        # raw_audio: (1, samples) on CPU
        wav = raw_audio.unsqueeze(1) if raw_audio.dim() == 2 else raw_audio
        # 1) MIMI encode
        eo   = self.mimi.encode(wav)
        sem  = eo.last_hidden_state            # (B, T, D)
        codes= eo.audio_codes                  # (B, T, num_codebooks)

        # 2) Acoustic embedding
        flat  = codes.view(-1)
        emb_ac= self.csm.audio_embeddings(flat).view(*codes.shape, -1)
        ac    = emb_ac.mean(dim=2)             # (B, T, D)

        # 3) FIXED fusion
        fused = (sem + ac) / 2

        # 4) Project to LLM space
        proj = self.projector(fused)

        # 5) Ultravox (LLM) forward
        with torch.no_grad():
            out = self.language_model(
                inputs_embeds=proj,
                return_dict=True,
                output_hidden_states=True,
            )
        h = out.hidden_states[-1]              # (B, T, D)

        # 6) CSM generate codes
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
        )
        frame_codes = frame_codes.view(B, -1, self.csm.config.audio_num_codebooks)

        # 7) Decode back to waveform
        with torch.no_grad():
            waveform = self.mimi.decode(frame_codes)

        return waveform

if __name__ == "__main__":
    device = torch.device('cpu')
    model = Audio2CodebooksCPU(
        ul_cfg_or_path="fixie-ai/ultravox-v0_2",
        csm_repo="sesame/csm-1b",
        tokenizer_id="meta-llama/Llama-3.2-1B",
        text_vocab_size=32000,
        audio_vocab_size=1024,
        audio_num_codebooks=32,
    ).to(device)

    # Inference on a few Expresso examples
    ds = load_dataset("ylacombe/expresso", split="train[:5]")
    print(f"Loaded {len(ds)} examples.")

    for i, ex in enumerate(ds):
        raw = torch.tensor(ex["audio"]["array"], dtype=torch.float32, device=device).unsqueeze(0)
        print(f"\nExample {i+1}, length {raw.shape[-1]} samples:")
        out = model(raw)
        print(" ➜ Output shape:", out.shape)

    print("\nDone.")











































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
        tokenizer_id: str,
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
        # extract the already-loaded projector & LLM
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
        # swap in our Ultravox LLM as the backbone
        self.csm.backbone = self.language_model
        self.csm.tie_weights()

        # --- 4) Tokenizer & caches ---
        self.tok = AutoTokenizer.from_pretrained(tokenizer_id)
        self.csm.setup_caches(max_batch_size=1)
        self.ul.reset_caches()

        # --- 5) Freeze everything ---
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        wav = raw_audio.unsqueeze(1) if raw_audio.dim() == 2 else raw_audio  # (B, 1, sampl>

    # 1) MIMI encode → semantic + audio codes
        eo    = self.mimi.encode(wav)
        sem   = eo.last_hidden_state           # (B, T, D)
        codes = eo.audio_codes                 # (B, T, num_codebooks)

    # 2) Acoustic embedding → (B, T, D)
        flat   = codes.view(-1)
        emb_ac = self.csm.audio_embeddings(flat).view(*codes.shape, -1)
        ac     = emb_ac.mean(dim=2)
    # 3) Gating fusion (learned)
    #    assumes self.gate: nn.Linear(2*D, D) and self.sig: Sigmoid()
        cat   = torch.cat([sem, ac], dim=-1)    # (B, T, 2D)
        g     = self.sig(self.gate(cat))       # (B, T, D)
        fused = g * sem + (1 - g) * ac         # (B, T, D)


        # 4) Project to LLM space
        proj = self.projector(fused)

        # 5) Ultravox LLM forward
        with torch.no_grad():
            out = self.language_model(
                inputs_embeds=proj,
                return_dict=True,
                output_hidden_states=True,
            )
        h = out.hidden_states[-1]             # (B, T, D)

        # 6) CSM generate audio codes
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

        # 7) Decode to waveform
        with torch.no_grad():
            waveform = self.mimi.decode(frame_codes)

        return waveform

if __name__ == "__main__":
    # Initialize on CPU
    device = torch.device('cpu')
    model = Audio2CodebooksCPU(
        ul_cfg_or_path       = "fixie-ai/ultravox-v0_2",
        csm_repo             = "sesame/csm-1b",
        tokenizer_id         = "meta-llama/Llama-3.2-1B",
        text_vocab_size      = 32000,
        audio_vocab_size     = 1024,
        audio_num_codebooks  = 32,
    ).to(device)

    # Inference on first 5 samples of Expresso
    ds = load_dataset("ylacombe/expresso", split="train[:5]")
    print(f"Loaded {len(ds)} examples for inference.")

    for i, ex in enumerate(ds):
        raw = torch.tensor(ex["audio"]["array"], dtype=torch.float32, device=device).unsque>
        print(f"\nExample {i+1}: input length {raw.shape[-1]} samples")
        out = model(raw)
        print(f"  ➜ Output waveform shape: {out.shape}")

    print("\nDone.")
