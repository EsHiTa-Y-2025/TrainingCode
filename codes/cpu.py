import sys
sys.path.append("/home/eshita_24bcs10024/TrainingCode/ultravox/ultravox/model")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from ultravox_model import UltravoxModel, UltravoxConfig, UltravoxProjector
sys.path.append("/home/eshita_24bcs10024/TrainingCode/csm")
from models import Model as CSMModel, ModelArgs
sys.path.append("/home/eshita_24bcs10024/TrainingCode/ultravox/ultravox")

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
    ):
        super().__init__()
        # 1) MIMI codec & feature extractor
        mimi_weight = hf_hub_download(
            repo_id=loaders.DEFAULT_REPO,
            filename=loaders.MIMI_NAME,
        )
        self.mimi = loaders.get_mimi(mimi_weight, device=torch.device('cpu'))
        self.mimi.set_num_codebooks(audio_num_codebooks)


        # 2) Ultravox + projector
        self.uv_cfg = UltravoxConfig.from_pretrained(ul_cfg_or_path)
        self.ul = UltravoxModel.from_pretrained(
            ul_cfg_or_path,
            config=self.uv_cfg,
            load_in_8bit=False,
            device_map=None,
        )
        self.ul.tie_weights()
        self.projector = UltravoxProjector(
            ul_cfg_or_path,
            config=self.uv_cfg,
        )

        # 3) CSM decoder
        args = ModelArgs(
            decoder_flavor=csm_decoder_flavor,
            text_vocab_size=text_vocab_size,
            audio_vocab_size=audio_vocab_size,
            audio_num_codebooks=audio_num_codebooks,
        )
        self.csm = CSMModel.from_pretrained(csm_repo, config=args)
        self.csm.backbone = self.ul.language_model
        uv_dim = self.uv_cfg.text_config.hidden_size
        dec_dim = self.csm.decoder.tok_embeddings.embedding_dim
        self.csm.projection = nn.Linear(uv_dim, dec_dim, bias=False)
        self.csm.tie_weights()

        # tokenizer & cache setup
        self.tok = AutoTokenizer.from_pretrained(tokenizer_id)
        self.csm.setup_caches(max_batch_size=1)
        self.ul.reset_caches()

        # Freeze *all* parameters
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        """
        raw_audio: (1, samples) on CPU
        returns: waveform decoded by MIMI on CPU
        """
        # 1) MIMI encode
        wav = raw_audio.unsqueeze(1) if raw_audio.dim() == 2 else raw_audio
        eo = self.mimi.encode(wav)
        sem = eo.last_hidden_state           # (B, T, D)
        codes = eo.audio_codes               # (B, T, num_codebooks)

        # 2) Acoustic embedding & aggregation
        flat = codes.view(-1)
        emb_ac = self.csm.audio_embeddings(flat).view(*codes.shape, -1)
        ac = emb_ac.mean(dim=2)              # (B, T, D)

        # 3) FIXED fusion: simple average
        fused = (sem + ac) / 2

        # 4) Project to LLM space
        proj = self.projector(fused)

        # 5) Ultravox forward (frozen)
        with torch.no_grad():
            out = self.ul(
                inputs_embeds=proj,
                return_dict=True,
                return_dict=True,
                output_hidden_states=True,
            )
        h = out.hidden_states[-1]            # (B, T, D)

        # 6) CSM generate codes
        B = h.size(0)
        tokens = torch.zeros(
            B, 1, self.csm.config.audio_num_codebooks + 1,
            device=h.device, dtype=torch.long
        )
        mask = torch.ones_like(tokens, dtype=torch.float)
        pos  = torch.zeros(B, 1, dtype=torch.long, device=h.device)
        frame_codes = self.csm.generate_frame(
            tokens=tokens,
            tokens_mask=mask,
            input_pos=pos,
            temperature=1.0,
            topk=5,
        )
        frame_codes = frame_codes.view(
            B, -1, self.csm.config.audio_num_codebooks
        )

        # 7) Decode back to waveform
        with torch.no_grad():
            waveform = self.mimi.decode(frame_codes)

        return waveform

if __name__ == "__main__":
    # 1) Initialize model on CPU
    device = torch.device('cpu')
    model = Audio2CodebooksCPU(
        ul_cfg_or_path="fixie-ai/ultravox-v0_2",
        csm_repo="sesame/csm-1b",
        tokenizer_id="meta-llama/Llama-3.2-1B",
        text_vocab_size=32000,
        audio_vocab_size=1024,
        audio_num_codebooks=32,
    ).to(device)

    # 2) Load a small slice of expresso for inference
    ds = load_dataset("ylacombe/expresso", split="train[:5]")  # first 5 exampl>
    print(f"Loaded {len(ds)} examples for inference")

    # 3) Run inference loop
    for i, example in enumerate(ds):
        raw = torch.tensor(example["audio"]["array"], dtype=torch.float32, devi>
        print(f"\nExample {i+1}: waveform length = {raw.shape[-1]} samples")
        out_wav = model(raw)
        print(f"  ➜ Generated waveform shape: {out_wav.shape}")

    print("\nInference complete.")