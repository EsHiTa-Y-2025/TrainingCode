import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from moshi.models import loaders

from ultravox_model import UltravoxModel, UltravoxConfig, UltravoxProjector
from csm_model import Model as CSMModel, ModelArgs

class Audio2Codebooks(nn.Module):
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
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1) MIMI codec & feature extractor
        mimi_weight = hf_hub_download(
            repo_id=loaders.DEFAULT_REPO,
            filename=loaders.MIMI_NAME,
        )
        self.mimi = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(audio_num_codebooks)
        self.feat = self.mimi.feature_extractor

        # 2) Ultravox config + LLM + projector
        self.uv_cfg = UltravoxConfig.from_pretrained(ul_cfg_or_path)
        self.ul = UltravoxModel.from_pretrained(
            ul_cfg_or_path,
            config=self.uv_cfg,
            load_in_8bit=True,
            device_map="auto",
            llm_int8_enable_fp32_cpu_offload=True,
        )
        self.ul.tie_weights()
        self.projector = UltravoxProjector.from_pretrained(
            ul_cfg_or_path,
            config=self.uv_cfg,
        ).to(self.device).eval()

        # 3) CSM decoder with swapped Ultravox backbone
        args = ModelArgs(
            decoder_flavor=csm_decoder_flavor,
            text_vocab_size=text_vocab_size,
            audio_vocab_size=audio_vocab_size,
            audio_num_codebooks=audio_num_codebooks,
        )
        self.csm = CSMModel.from_pretrained(csm_repo, config=args).to(self.device).eval()
        self.csm.backbone = self.ul.language_model
        uv_dim = self.uv_cfg.text_config.hidden_size
        dec_dim = self.csm.decoder.tok_embeddings.embedding_dim
        self.csm.projection = nn.Linear(uv_dim, dec_dim, bias=False).to(self.device)
        self.csm.tie_weights()

        # 4) fusion gate (trainable)
        self.gate = nn.Linear(2 * uv_dim, uv_dim).to(self.device)
        self.sig = nn.Sigmoid()

        # 5) tokenizer (optional)
        self.tok = AutoTokenizer.from_pretrained(tokenizer_id)

        # prepare caches
        self.csm.setup_caches(max_batch_size=1)
        self.ul.reset_caches()

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        # raw_audio: (B, samples)
        wav = raw_audio.unsqueeze(1) if raw_audio.dim() == 2 else raw_audio
        eo = self.mimi.encode(wav)
        sem = eo.last_hidden_state
        codes = eo.audio_codes

        # re-embed codes
        flat = codes.view(-1)
        emb_ac = self.csm.audio_embeddings(flat).view(*codes.shape, -1)
        ac = emb_ac.mean(dim=2)

        # gated fusion
        cat = torch.cat([sem, ac], dim=-1)
        g = self.sig(self.gate(cat))
        fused = g * sem + (1 - g) * ac

        # project to LLM space
        proj = self.projector(fused)

        # Ultravox forward (frozen)
        with torch.no_grad():
            out = self.ul(inputs_embeds=proj, return_dict=True, output_hidden_states=True)
        h = out.hidden_states[-1]

        # CSM generate next audio codes
        B = h.size(0)
        tokens = torch.zeros(B, 1, self.csm.config.audio_num_codebooks + 1,
                             device=self.device, dtype=torch.long)
        mask = torch.ones_like(tokens, dtype=torch.float)
        pos = torch.zeros(B, 1, dtype=torch.long, device=self.device)
        frame_codes = self.csm.generate_frame(
            tokens=tokens,
            tokens_mask=mask,
            input_pos=pos,
            temperature=1.0,
            topk=5,
        )
        frame_codes = frame_codes.view(B, -1, self.csm.config.audio_num_codebooks)

        # decode to waveform
        with torch.no_grad():
            waveform = self.mimi.decode(frame_codes)
        return waveform


# Dataset for input/target pairs
torch.manual_seed(0)
class AudioFrameDataset(Dataset):
    def __init__(self, waveforms, frame_size):
        self.pairs = []
        for w in waveforms:
            for i in range(0, w.shape[-1] - 2 * frame_size, frame_size):
                inp = w[i : i + frame_size]
                tgt = w[i + frame_size : i + 2 * frame_size]
                self.pairs.append((inp, tgt))

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]


if __name__ == "__main__":
    # --- 1) Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Audio2Codebooks(
        ul_cfg_or_path="fixie-ai/ultravox-v0_2",
        csm_repo="sesame/csm-1b",
        tokenizer_id="meta-llama/Llama-3.2-1B",
        text_vocab_size=32000,
        audio_vocab_size=1024,
        audio_num_codebooks=32,
        device=device,
    ).to(device)

    # freeze all but gate
    for p in model.parameters(): p.requires_grad = False
    for p in model.gate.parameters(): p.requires_grad = True

    optimizer = torch.optim.Adam(model.gate.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # load dataset (replace with real loading)
    from datasets import load_dataset
    ds = load_dataset("ylacombe/expresso", split="train")
    raw_waveforms = [torch.tensor(x["audio"]["array"]) for x in ds]
    frame_size = model.feat.sampling_rate // 10
    dataset = AudioFrameDataset(raw_waveforms, frame_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # --- 2) Training loop ---
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inp, tgt in loader:
            inp = inp.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()
            pred = model(inp)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, loss: {total_loss/len(loader):.6f}")