# entity_wwm_801010.py
import json, argparse, random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForMaskedLM,
    Trainer, TrainingArguments, DataCollatorWithPadding
)

# ---------- Dataset (JSON / JSONL) ----------
class EntityMaskDataset(Dataset):
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if txt.startswith("["):
                self.data = json.loads(txt)
            else:
                self.data = [json.loads(line) for line in txt.splitlines()]
        for ex in self.data:
            ex.setdefault("entities", [])
            for ent in ex["entities"]:
                s, e = ent["span"]
                ent["span"] = (int(s), int(e))
                ent["type"] = str(ent.get("type", "UNK"))

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

# ---------- Build type -> token-id pool from training JSON ----------
def build_type_token_pool(train_dataset: EntityMaskDataset, tokenizer, max_per_type: int = 200000) -> Dict[str, List[int]]:
    pool = defaultdict(list)
    for ex in train_dataset.data:
        text = ex["text"]
        for ent in ex["entities"]:
            s, e = ent["span"]
            etype = ent["type"]
            piece = text[s:e]
            enc = tokenizer(
                piece,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False
            )
            # append all subword ids (skip empty)
            ids = enc["input_ids"]
            if ids:
                pool[etype].extend(ids)
                # bound size to avoid memory blow-up
                if len(pool[etype]) > max_per_type:
                    pool[etype] = random.sample(pool[etype], max_per_type)
    return dict(pool)

# ---------- Preprocess: tokenization + offsets + entity-type-per-token ----------
def build_preprocess_fn(tokenizer, max_length=512):
    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        text = example["text"]
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        offsets = enc["offset_mapping"]
        special = enc["special_tokens_mask"]

        # Build token-wise entity flags and types
        tok_in_entity = [False] * len(offsets)
        tok_entity_type: List[Optional[str]] = [None] * len(offsets)

        for ent in example.get("entities", []):
            es, ee = ent["span"]
            etype = ent["type"]
            for ti, (cs, ce) in enumerate(offsets):
                if special[ti] == 1:
                    continue
                if max(cs, es) < min(ce, ee):  # overlap
                    tok_in_entity[ti] = True
                    tok_entity_type[ti] = etype

        enc["entity_token_mask"] = tok_in_entity
        enc["entity_token_type"] = tok_entity_type  # strings or None
        return enc
    return preprocess

# ---------- Utilities ----------
def contiguous_true_spans(mask: torch.Tensor) -> List[Tuple[int,int]]:
    # returns list of [start, end) for runs of True
    spans = []
    start = None
    for i, v in enumerate(mask.tolist()):
        if v and start is None:
            start = i
        if (not v) and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans

# ---------- Collator: Entity-level WWM with 80/10/10 ----------
@dataclass
class DataCollatorForEntityWWM801010:
    tokenizer: Any
    type_token_pool: Dict[str, List[int]]
    mask_prob: float = 1.0   # probability of selecting an entity span at all (usually 1.0 for “always”)
    rep_prob: float = 0.10   # within selected spans: 10% replace
    keep_prob: float = 0.10  # within selected spans: 10% keep (i.e., unchanged)
    # remaining (1 - rep_prob - keep_prob) is mask (typically 0.80)

    def __call__(self, features: List[Dict[str,Any]]) -> Dict[str, torch.Tensor]:
        # pull and remove aux fields before padding
        ent_masks = [torch.tensor(f.pop("entity_token_mask"), dtype=torch.bool) for f in features]
        ent_types_raw = [f.pop("entity_token_type") for f in features]  # list of list[str|None]
        special_masks = [torch.tensor(f["special_tokens_mask"], dtype=torch.bool) for f in features]

        pad = DataCollatorWithPadding(self.tokenizer, return_tensors="pt")
        batch = pad(features)

        max_len = batch["input_ids"].size(1)

        def pad_to_len_bool(m: torch.Tensor) -> torch.Tensor:
            L = m.size(0)
            if L >= max_len: return m[:max_len]
            return torch.nn.functional.pad(m, (0, max_len - L), value=False)

        def pad_types(types: List[Optional[str]]) -> List[Optional[str]]:
            if len(types) >= max_len: return types[:max_len]
            return types + [None] * (max_len - len(types))

        ent_masks = torch.stack([pad_to_len_bool(m) for m in ent_masks], 0)
        special_masks = torch.stack([pad_to_len_bool(m) for m in special_masks], 0)
        ent_types = [pad_types(t) for t in ent_types_raw]

        input_ids = batch["input_ids"].clone()
        labels = batch["input_ids"].clone()

        # never compute loss on specials
        labels[special_masks] = -100

        # process each sequence: group contiguous entity tokens into spans
        B, L = input_ids.size()
        mask_id = self.tokenizer.mask_token_id
        vocab_size = self.tokenizer.vocab_size

        for b in range(B):
            mask_b = ent_masks[b] & (~special_masks[b])

            for (s, e) in contiguous_true_spans(mask_b):
                if random.random() > self.mask_prob:
                    continue
                r = random.random()
                if r < (1.0 - self.rep_prob - self.keep_prob):  # 80%: MASK
                    input_ids[b, s:e] = mask_id
                    # labels already original ids (good)
                elif r < (1.0 - self.keep_prob):  # 10%: REPLACE with same-type token pool
                    # Determine the type (majority vote inside span)
                    # We’ll pick the first non-None type in the span
                    etype = None
                    for i in range(s, e):
                        if ent_types[b][i] is not None:
                            etype = ent_types[b][i]
                            break
                    pool = self.type_token_pool.get(etype or "", [])
                    for i in range(s, e):
                        if pool:
                            rid = random.choice(pool)
                        else:
                            # fallback to random vocab id excluding specials if possible
                            rid = random.randrange(vocab_size)
                        input_ids[b, i] = rid
                    # labels: keep original ids so the model must predict originals
                else:  # 10%: KEEP (no change)
                    # input stays; labels already original ids
                    pass

            # For non-entity tokens: do not predict (optional)
            # If you prefer to also learn on 15% random non-entity WWM, extend here.
            non_entity = (~ent_masks[b]) & (~special_masks[b])
            labels[b, non_entity] = -100

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch

# ---------- Glue code ----------
class MappedDataset(Dataset):
    def __init__(self, base: Dataset, fn): self.base, self.fn = base, fn
    def __len__(self): return len(self.base)
    def __getitem__(self, i): return self.fn(self.base[i])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--eval_json", default=None)
    ap.add_argument("--model_name", default="cl-tohoku/bert-base-japanese-v3")
    ap.add_argument("--output_dir", default="./bert-entity-wwm-801010")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name, config=config)

    train_raw = EntityMaskDataset(args.train_json)
    eval_raw = EntityMaskDataset(args.eval_json) if args.eval_json else None

    # Build type->token-id pool from TRAIN ONLY (情報リーク防止)
    type_pool = build_type_token_pool(train_raw, tokenizer)

    preprocess = build_preprocess_fn(tokenizer, max_length=args.max_length)
    train_ds = MappedDataset(train_raw, preprocess)
    eval_ds = MappedDataset(eval_raw, preprocess) if eval_raw else None

    collator = DataCollatorForEntityWWM801010(
        tokenizer=tokenizer,
        type_token_pool=type_pool,
        mask_prob=1.0,   # 全エンティティを対象
        rep_prob=0.10,   # 10% 代替
        keep_prob=0.10   # 10% そのまま
    )

    args_tr = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=500 if eval_ds is not None else None,
        save_steps=1000,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
