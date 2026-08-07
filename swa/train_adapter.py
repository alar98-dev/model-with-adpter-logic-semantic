import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from swa.adapter import SemanticWarpAdapter, triplet_loss


class TripletDataset(Dataset):
    def __init__(self, path):
        self.samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append((obj['anchor'], obj['positive'], obj['negative']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    anchors, positives, negatives = zip(*batch)
    return list(anchors), list(positives), list(negatives)


def encode_batch(texts, tokenizer, device):
    tok = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    tok = {k: v.to(device) for k, v in tok.items()}
    return tok


def compute_embeddings(model, tokenizer, texts, device):
    tok = encode_batch(texts, tokenizer, device)
    with torch.no_grad():
        out = model(**tok, output_hidden_states=True)
        emb = out.last_hidden_state.mean(dim=1)
    return emb


def train(args):
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir).to(device)
    model.eval()

    # init adapters (local + macro)
    d = model.config.hidden_size
    adapter_local = SemanticWarpAdapter(d=d, k=args.k_local, ortho_lambda=args.lambda_ortho, low_rank=args.low_rank, node=0).to(device)
    adapter_macro = SemanticWarpAdapter(d=d, k=args.k_macro, ortho_lambda=args.lambda_ortho, low_rank=args.low_rank, node=0).to(device)

    optim = torch.optim.Adam(list(adapter_local.parameters()) + list(adapter_macro.parameters()), lr=args.lr, weight_decay=1e-5)

    dataset = TripletDataset(args.triplets)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for anchors, positives, negatives in loader:
            # compute embeddings via base model
            a_emb = compute_embeddings(model, tokenizer, anchors, device)
            p_emb = compute_embeddings(model, tokenizer, positives, device)
            n_emb = compute_embeddings(model, tokenizer, negatives, device)

            fa = adapter_local(a_emb)
            fp = adapter_local(p_emb)
            fn = adapter_local(n_emb)

            # aggregate to macro (simple mean per batch as prototype)
            fa_agg = adapter_macro(fa.mean(dim=0, keepdim=True)).repeat(fa.size(0), 1)
            fp_agg = adapter_macro(fp.mean(dim=0, keepdim=True)).repeat(fp.size(0), 1)
            fn_agg = adapter_macro(fn.mean(dim=0, keepdim=True)).repeat(fn.size(0), 1)

            loss_trip = triplet_loss(fa_agg, fp_agg, fn_agg, margin=args.margin)
            loss_ortho = adapter_local.ortho_loss() + adapter_macro.ortho_loss()
            loss = loss_trip + loss_ortho * args.lambda_ortho

            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item()

        avg = epoch_loss / len(loader) if len(loader) > 0 else 0.0
        print(f"Epoch {epoch}/{args.epochs} avg_loss={avg:.6f}")

        # checkpoint
        torch.save({'local': adapter_local.state_dict(), 'macro': adapter_macro.state_dict()}, save_dir / f'adapter_epoch{epoch}.pt')

    print("Training finished. Latest adapters saved to", save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='models/gemma3-4b')
    parser.add_argument('--triplets', type=str, default='data/sample_triplets.jsonl')
    parser.add_argument('--save-dir', type=str, default='swa/checkpoints')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--k-local', dest='k_local', type=int, default=64)
    parser.add_argument('--k-macro', dest='k_macro', type=int, default=128)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--lambda-ortho', dest='lambda_ortho', type=float, default=1e-3)
    parser.add_argument('--low-rank', dest='low_rank', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
