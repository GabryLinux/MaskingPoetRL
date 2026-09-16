import math
import os
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from Configuration import TRANSFORMER_CONFIGURATION
from Tokenizers.tokenizer import BasePoetryTokenizer


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.d = d
        self.heads = heads
        self.d_k = d // heads

        assert d % heads == 0, "Embedding dimension 'd' must be divisible by 'heads'."

        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # -1e9 previene l'insorgere di NaN nella Softmax rispetto a float('-inf')
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask == 0, min_value)

        attention_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d)

        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d, heads)
        self.norm1 = nn.LayerNorm(d)
        
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d)
        )
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        att_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(att_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x


class PoetEmbedder(nn.Module):
    def __init__(self, tokenizer: BasePoetryTokenizer, d: int, n: int, heads: int, max_seq_len: int, dropout: float, device: torch.device):
        super().__init__()
        self.d = d
        self.max_seq_len = max_seq_len
        self.layers_outputs : List[torch.Tensor] = []
        self.tokenizer = tokenizer
        vocab_size = len(tokenizer.vocab)
        self.token_embedding = nn.Embedding(vocab_size, d, padding_idx=0)
        self.positional_embedding = nn.Embedding(max_seq_len, d)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d, heads, dropout) for _ in range(n)
        ])

        self.device = device

        self.final_norm = nn.LayerNorm(d)
        self.fc_out = nn.Linear(d, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def from_config(tokenizer: BasePoetryTokenizer, WEIGHT_PATH: str, device: torch.device) -> "PoetEmbedder":
        d = TRANSFORMER_CONFIGURATION.PARAMETERS()["VECTOR_DIMENSION"]
        n = TRANSFORMER_CONFIGURATION.PARAMETERS()["N_ATTENTION_LAYERS"]
        heads = TRANSFORMER_CONFIGURATION.PARAMETERS()["N_HEADS_PER_LAYER"]
        max_seq_len = TRANSFORMER_CONFIGURATION.PARAMETERS()["MAX_SEQ_LEN"]
        dropout = TRANSFORMER_CONFIGURATION.PARAMETERS()["DROPOUT"]
        embedder = PoetEmbedder(tokenizer=tokenizer, d=d, n=n, heads=heads, max_seq_len=max_seq_len, dropout=dropout, device=device)
        if os.path.exists(WEIGHT_PATH):
            try:
                embedder.load_state_dict(torch.load(WEIGHT_PATH, map_location=device)['model_state_dict'])
                print(f"[INFO] Caricato modello pre-addestrato da '{WEIGHT_PATH}'")
            except Exception as e:
                print(f"[ERROR] Errore durante il caricamento dei pesi da '{WEIGHT_PATH}': {e}")
                raise e
        else:
            print(f"[WARNING] Nessun file di pesi trovato in '{WEIGHT_PATH}'. Inizializzazione casuale del modello.")
            
        return embedder.to(device)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self, 
        x: torch.Tensor, 
        padding_mask: torch.Tensor = None, 
    ):
        batch_size, seq_len = x.shape
        
        vocab_max = self.token_embedding.num_embeddings - 1
        x = torch.clamp(x, min=0, max=vocab_max)

        max_pos = self.positional_embedding.num_embeddings - 1
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        positions = torch.clamp(positions, min=0, max=max_pos)

        out = self.token_embedding(x) + self.positional_embedding(positions)
        out = self.emb_dropout(out)

        if padding_mask is not None:
            padding_mask = padding_mask.to(x.device)
            if len(padding_mask.shape) == 2:
                padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)

        self.layers_outputs.clear()
        for idx, layer in enumerate(self.layers):
            out = layer(out, padding_mask)
            self.layers_outputs.append(out)

        self.last_attention_output = out
        out_norm = self.final_norm(out)
        logits = self.fc_out(out_norm)

        return logits

    def get_layer_output(self, layer_idx: int) -> torch.Tensor:
        return self.layers_outputs[layer_idx]


    def forward_from_text(self, text: str) -> torch.Tensor:
        token_ids = self.tokenizer.tokenize(text)
        pad_id = self.tokenizer.vocab.get("[PAD]", 0)

        seq_len = len(token_ids)
        if seq_len < self.max_seq_len:
            token_ids = token_ids + [pad_id] * (self.max_seq_len - seq_len)
            padding_mask = [1] * seq_len + [0] * (self.max_seq_len - seq_len)
        else:
            token_ids = token_ids[:self.max_seq_len]
            padding_mask = [1] * self.max_seq_len

        device = next(self.parameters()).device
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        padding_mask_tensor = torch.tensor([padding_mask], dtype=torch.long, device=device)
        return self.forward(input_tensor, padding_mask=padding_mask_tensor)