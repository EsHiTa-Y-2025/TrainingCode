#!/usr/bin/env python3
import os, sys

# 1) Point to your local repos
sys.path.insert(0, os.path.expanduser("~/TrainingCode/ultravox/ultravox/model"))
sys.path.insert(0, os.path.expanduser("~/TrainingCode/csm"))


import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from moshi.models import loaders
from ultravox_model import UltravoxModel, UltravoxConfig
from models import Model as CSMModel, ModelArgs   # from your csm repo
from transformers import MimiConfig
from csm.models import sample_topk



class FrozenBackbone(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden

    def forward(self, *args, **kwargs):
        return self.hidden

    def caches_are_enabled(self):
        return True




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
        self.mimi_config = MimiConfig.from_pretrained("kyutai/mimi")
        self.num_quantizers = audio_num_codebooks

        h_mimi = 512
        self.ac_proj = nn.Linear(2048, 6144, bias=False)
        self.sem_proj = nn.Linear(512, 6144, bias=False)
#        self.gate = nn.Linear(sem_dim + ac_dim, hidden_dim)
        self.fused_proj = nn.Linear(512, 262144)  # or whatever `fused.shape[-1]` is → 6144 i>


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
        self.projector._pad_and_stack = nn.Identity()
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
        self.h_to_csm_backbone = nn.Linear(4096, 2048, bias=False)
#        self.csm.setup_caches(max_batch_size=1)
        self._csm_backbone_orig = self.csm.backbone
        backbone_dim = self.csm.projection.in_features
        self.ultra_to_csm_back = nn.Linear(4096, backbone_dim, bias=False)


        # --- 5) Freeze everything ---
        for p in self.parameters():
            p.requires_grad = False


        # gating layers for fusion
#        D = self.ul.config.hidden_size
        self.gate = nn.Linear(2 * 6144, 6144)
        self.sig  = nn.Sigmoid()
        max_frames_to_generate = 30  # Or whatever number of frames you want to generate


    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        # ensure shape (B, 1, L)
        wav = raw_audio.unsqueeze(1) if raw_audio.dim() == 2 else raw_audio

        # === 1) MIMI encode → semantic + acoustic codes ===

        # (a) CNN encoder → (B, hidden, T1)
        cnn_emb = self.mimi.encoder(wav)
        print("cnn_emb:", cnn_emb.shape)
        cnn_time = cnn_emb.transpose(1, 2)
#        cnn_time = cnn_time.squeeze(0)
        print("cnn_emb.shape:", cnn_emb.shape)
        print("cnn_time:", cnn_time.shape)
        print("completed cnn_time")

        # (b) Transformer encoder → tuple: (hidden_states, past_key_values)
        print("starting trans_out")
        trans_out = self.mimi.encoder_transformer(cnn_emb)
        print("completed trans_out")

#        trans_out = self.mimi.encoder_transformer(cnn_emb)
#        trans_out = self.mimi.encoder_transformer(cnn_emb.transpose(1, 2))
        sem = trans_out[0]                   # (B, T1, hidden)
#       print("cnn_emb:", cnn_emb.shape)      # should be [B, 512, T1]
#       print("cnn_time:", cnn_time.shape)    # should be [B, T1, 512]
#       print("sem_cf:", sem_cf.shape)        # should be [B, 512, T1]
        # (c) to channel-first for downsample: (B, hidden, T1)
#        cnn_time = cnn_time.unsqueeze(0)
#        sem_cf = sem.transpose(1, 2)
#        print("sem_cf:", sem_cf.shape)
        # (d) optional downsample → (B, hidden, T2)

#        sem = sem.transpose(1, 2)
        print("sem.shape:", sem.shape)


        print("quant_in started")
        quant_in = self.mimi.downsample(sem) if self.mimi.downsample is not None else sem_cf
        print("quant_in completed")
        # (e) quantize with all codebooks → (Q, B, T2)
#        all_codes = self.mimi.quantizer.encode(
#            quant_in,
#            self.mimi_config.num_quantizers
#        )


#        n_q = self.mimi.quantizer.config.num_quantizers
#        all_codes = self.mimi.quantizer.encode(
#            quant_in,
#            n_q
#        )
        print("all_codes started")
        all_codes = self.mimi.quantizer.encode(quant_in)
        acoustic_codes = all_codes[:, 1:, :]
        codes = acoustic_codes
        B, Q, T2 = acoustic_codes.shape
        print("all_codes completed")
#        H = emb_ac.shape[-1] if emb_ac.dim() == 2 else emb_ac.size(-1)


        # (f) drop first codebook → (Q-1, B, T2)
#        acoustic_codes = all_codes[1:]

        # (g) permute to (B, Q-1, T2)
#        codes = acoustic_codes.permute(1, 0, 2)

        # === 2) Acoustic embedding → (B, T2, hidden) ===
        flat   = codes.reshape(-1)
#        emb_ac = self.csm.audio_embeddings(flat).view(*codes.shape, -1)
#        ac     = emb_ac.mean(dim=2)                   # (B, hidden)

        print("emb started")
        emb_ac = self.csm.audio_embeddings(flat)
        H = emb_ac.shape[-1] if emb_ac.dim() == 2 else emb_ac.size(-1)

        emb_ac = emb_ac.view(B, Q, T2, H)
        print("emb completed")


        # broadcast to match sem time dimension
#        ac = ac.unsqueeze(1).expand(-1, sem.size(1), -1)  # (B, T1, hidden)
        ac = emb_ac.mean(dim=1)


        print("sem.shape:", sem.shape)  # expect (..., 512)
        print("ac.shape:",  ac.shape)   # expect (..., 512)

        ac_hp = self.ac_proj(ac)
        sem = sem.permute(0, 2, 1)
        sem_hp = self.sem_proj(sem)



#        ac_cf = ac.transpose(1, 2)
 #       if self.mimi.upsample is not None:
#            ac_up = self.mimi.upsample(ac_cf)
 #           ac_cf_proj = self.ac_proj(ac_cf.transpose(1, 2)).transpose(1, 2)  # [B, 512, T]
 #           ac_up = self.mimi.upsample(ac_cf_proj)
 #       else:
            # Option B: linear interpolate from 69 → 138
#            import torch.nn.functional as F
#            ac_up = F.interpolate(ac_cf, size=sem.size(1), mode="linear", align_corners=Fals>

            # Back to time-first:
#        ac = ac_up.transpose(1, 2)      # ➜ (B, T1, hidden)

#        sem = sem.permute(0, 2, 1)  # From [B, H, T] → [B, T, H]
#        sem = sem.transpose(1, 2)
        print("gatingsem.shape:", sem.shape)
        print("gatingac.shape: ", ac.shape)


        # === 3) Gating fusion ===
#        cat   = torch.cat([sem, ac], dim=-1)         # (B, T1, 2*hidden)
#        g     = self.sig(self.gate(cat))            # (B, T1, hidden)
#        fused = g * sem + (1 - g) * ac              # (B, T1, hidden)

        import torch.nn.functional as F
        ac_up = F.interpolate(
            ac_hp.transpose(1, 2),      # to (B, 4096, 69)
            size=sem_hp.size(1),         # 138
            mode="linear",
            align_corners=False
        ).transpose(1, 2)

        print("gating started")

#        if ac.shape[1] != sem.shape[1]:
#            import torch.nn.functional as F
#            ac = F.interpolate(ac.transpose(1, 2), size=sem.shape[1], mode="linear", align_c>

        print("sem_hp: ", sem_hp.shape)
        print("ac_up: ", ac_up.shape)

        print("cat before")
        cat = torch.cat([sem_hp, ac_up], dim=-1)
        print("cate completed")
        g     = self.sig(self.gate(cat))
        print("g completd")
        fused = g * sem_hp + (1 - g) * ac_up
        print("fused completd")
        print("gating completed")


 # cat = torch.cat([sem, ac], dim=-1)
#        g = self.sig(self.gate(cat))
 #       fused = g * sem + (1 - g) * ac
        print("beforefused.shape: ", fused.shape)
#        proj_input = self.fused_proj(fused)  # [B, T, 4096]
        proj = self.projector(fused)
        print("fused.shape: ", fused.shape)


        # === 4) Project to LLM space & 5) Ultravox LLM forward ===
        print("projector started")
#        proj_input = self.fused_proj(fused)  # [B, T, 4096]
 #       proj = self.projector(proj_input)
        print("fused.shape: ", fused.shape)
#        proj = self.projector(fused)
        print("projector completed")
        print("LLM put into")
        with torch.no_grad():
            out = self.language_model(
                inputs_embeds=proj,
                return_dict=True,
                output_hidden_states=True,
            )
        h = out.hidden_states[-1]                  # (B, T1, hidden)
        print("hidden states computed")
        print(h)

        # === 6) CSM generate audio codes & 7) Decode to waveform ===
        # 1) Project Ultravox features to CSM decoder space
        # 1) Project h to match decoder input dim
# Map h_csm → decoder embedding space
        print("into the decoder")
        h_csm = self.h_to_csm_backbone(h)
        h_backbone = h_csm
        h_dec = self.csm.projection(h_csm)   # (B, T, decoder_dim)

        # “Prime” the decoder’s KV caches by feeding in the full context:
        # (Note: this uses positional caching inside `self.csm.decoder` itself,
        #  so no need for a backbone or causal mask here.)
        _ = self.csm.decoder(
            h_dec,
            input_pos=torch.arange(0, h_dec.size(1), device=h_dec.device).unsqueeze(0),
            mask=None,            # uses its internal causal mask
        )
        print("decoder completed")

        topk       = 100
        temperature = 1.0

        print("codebook sampling started")
        # ——— AUTOREGRESSIVE CODEBOOK SAMPLING ———
        B   = h_dec.size(0)
        Q   = self.csm.config.audio_num_codebooks
        pos = torch.arange(0, 1, device=h_dec.device).unsqueeze(0)  # starting position [B,1]
        tokens = []  # will collect each frame’s Q tokens

        curr_h = None
        last_h = None

        for i in range(Q):
            if i == 0:
                # 1) sample codebook 0 from backbone hidden
                last_back = h_backbone[:, -1, :]                       # (B, backbone_dim)
                logits0   = self.csm.codebook0_head(last_back)         # (B, audio_vocab)
                sample0   = sample_topk(logits0, topk, temperature)    # (B, 1)
                tokens.append(sample0)

                # 2) embed into backbone_dim, *then* project to decoder_dim
                curr_h = self.csm._embed_audio(0, sample0)             # (B, 1, backbone_dim)
                curr_h = self.csm.projection(curr_h)                   # (B, 1, decoder_dim)

                # 3) set last_h for next decoder step
                last_h = h_dec[:, -1, :]                               # (B, decoder_dim)

            else:
                # 4) run a single decoder step on [last_h, curr_h]
                inp     = torch.cat([last_h.unsqueeze(1), curr_h], dim=1)  # (B,2,decoder_dim)
                seq_len = inp.size(1)
                pos = torch.arange(0, seq_len, device=inp.device).unsqueeze(0)
                dec_out = self.csm.decoder(inp, input_pos=pos, mask=None)  # (B,2,decoder_dim)
                h_i     = dec_out[:, -1, :]                                # (B, decoder_dim)

                # 5) sample codebook i
                logits_i = h_i @ self.csm.audio_head[i - 1]                 # (B, audio_vocab)
                sample_i = sample_topk(logits_i, topk, temperature)        # (B, 1)
                tokens.append(sample_i)

                # 6) embed & project for next step
                curr_h = self.csm._embed_audio(i, sample_i)                # (B, 1, backbone_>
                curr_h = self.csm.projection(curr_h)                       # (B, 1, decoder_d>

                # 7) update last_h
                last_h = h_i

            # 8) move position forward
            pos = pos + 1



        print("codebook sampling completed")

        # Stack Q samples into (B, Q, 1) and then decode
        frame_codes = torch.cat(tokens, dim=1).unsqueeze(2)  # (B, Q, 1)
        waveform   = self.mimi.decode(frame_codes)          # (B,1,samples)

        return waveform


if __name__ == "__main__":
    device = torch.device('cpu')
    model = Audio2CodebooksCPU(
        ul_cfg_or_path       = "fixie-ai/ultravox-v0_2",
        csm_repo             = "sesame/csm-1b",
        text_vocab_size      = 32000,
        audio_vocab_size     = 1024,
        audio_num_codebooks  = 32,
    )

    ds = load_dataset("ylacombe/expresso", split="train[:5]")
    print(f"Loaded {len(ds)} examples for inference.")

    from itertools import islice
    for i, ex in enumerate(islice(ds, 5)):
        raw = torch.tensor(ex["audio"]["array"], dtype=torch.float32).unsqueeze(0)
        out = model(raw)
        print(f"Example {i+1}: output shape = {out.shape}")

    print("\nDone.")