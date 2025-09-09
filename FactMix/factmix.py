#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import json
import random
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from transformers.pipelines import TokenClassificationPipeline

# ===================== Data structures =====================

@dataclass
class Entity:
    name: str
    span: Tuple[int, int]  # [start, end)
    type: str

@dataclass
class Sample:
    curid: str
    text: str
    entities: List[Entity] = field(default_factory=list)

# ===================== I/O =====================

def load_dataset(path: str) -> List[Sample]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    ds: List[Sample] = []
    for ex in raw:
        ents = [Entity(e["name"], (e["span"][0], e["span"][1]), e["type"]) for e in ex.get("entities", [])]
        ds.append(Sample(ex.get("curid", ""), ex["text"], ents))
    return ds

def dump_dataset(path: str, data: List[Sample]) -> None:
    out = []
    for s in data:
        out.append({
            "curid": s.curid,
            "text": s.text,
            "entities": [{"name": e.name, "span": [e.span[0], e.span[1]], "type": e.type} for e in s.entities]
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

# ===================== BIO helpers =====================

def char_bio(text: str, entities: List[Entity]) -> List[str]:
    labels = ["O"] * len(text)
    for ent in sorted(entities, key=lambda e: e.span[0]):
        s, e = ent.span
        if s < 0 or e > len(text) or s >= e:
            continue
        labels[s] = f"B-{ent.type}"
        for i in range(s + 1, e):
            labels[i] = f"I-{ent.type}"
    return labels

def update_spans_after_replacement(
    text: str,
    entities: List[Entity],
    replaced_range: Tuple[int, int],
    replacement: str
) -> Tuple[str, List[Entity]]:
    rs, re = replaced_range
    new_text = text[:rs] + replacement + text[re:]
    diff = len(replacement) - (re - rs)
    new_entities: List[Entity] = []
    for ent in entities:
        s, e = ent.span
        if e <= rs:
            # before
            new_entities.append(Entity(ent.name, (s, e), ent.type))
        elif s >= re:
            # after -> shift
            new_entities.append(Entity(ent.name, (s + diff, e + diff), ent.type))
        else:
            # overlap (原則：_ctx_ではOのみを置換するので発生しない想定。防御的に最小限の補正)
            ns, ne = s, e
            if s >= rs:
                ns = s + diff
                ne = e + diff
            new_entities.append(Entity(ent.name, (ns, ne), ent.type))
    return new_text, new_entities

# ===================== Entity base =====================

def build_entity_base(dataset: List[Sample]) -> Dict[str, List[str]]:
    base: Dict[str, set] = {}
    for s in dataset:
        for e in s.entities:
            base.setdefault(e.type, set()).add(e.name)
    return {k: sorted(list(v)) for k, v in base.items()}

# ===================== Context-level (char/word) =====================

class ContextAugmentor:
    """
    Context-level semi-fact generation.
    - unit='char': 1 文字だけ [MASK]
    - unit='word': Tokenizer(Fast)のword_ids()とoffset_mappingで語スパンを1個の[MASK]に置換
    """
    def __init__(self, mlm_model: str, device: int = -1, unit: str = "word", mask_token: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(mlm_model, use_fast=True)
        self.unmasker = pipeline(
            "fill-mask",
            model=AutoModelForMaskedLM.from_pretrained(mlm_model),
            tokenizer=self.tokenizer,
            device=device
        )
        self.mask_token = mask_token or self.tokenizer.mask_token
        if self.mask_token is None:
            raise ValueError("This tokenizer has no mask token. Use a masked LM (e.g., BERT).")
        if unit not in ("char", "word"):
            raise ValueError("--mask_unit must be 'char' or 'word'")
        self.unit = unit

    # ---------- char-level ----------
    def _pick_one_O_char_index(self, text: str, entities: List[Entity], rng: random.Random) -> Optional[int]:
        bio = char_bio(text, entities)
        cand = [i for i, lab in enumerate(bio) if lab == "O" and not text[i].isspace()]
        return rng.choice(cand) if cand else None

    def _replace_char(self, sample: Sample, rng: random.Random, top_k: int = 5) -> Optional[Sample]:
        i = self._pick_one_O_char_index(sample.text, sample.entities, rng)
        if i is None:
            return None
        masked_text = sample.text[:i] + self.mask_token + sample.text[i+1:]
        try:
            preds = self.unmasker(masked_text, top_k=top_k)
        except Exception:
            return None
        if not preds:
            return None
        original = sample.text[i]
        candidate = None
        for p in preds:
            tok = p["token_str"]
            if tok and tok.strip() and tok != original:
                candidate = tok
                break
        candidate = candidate or preds[0]["token_str"]
        new_text, new_entities = update_spans_after_replacement(sample.text, sample.entities, (i, i+1), candidate)
        return Sample(sample.curid, new_text, new_entities)

    # ---------- word-level ----------
    def _get_word_spans(self, text: str) -> List[Tuple[int, int]]:
        """
        Fast tokenizerでword_ids()とoffset_mappingを使い、
        各 word_id に対する [start,end)（文字オフセット）を返す。
        """
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True
        )
        # transformers>=4.21 のFastトークナイザは .word_ids() を持つ
        # enc からエンコーディングを取り、word_ids を得る
        try:
            word_ids = enc.word_ids()
            offsets = enc["offset_mapping"]
        except Exception:
            # 一部の古いバージョン対応
            word_ids = enc.encodings[0].word_ids
            offsets = enc.encodings[0].offsets

        # 特殊トークン(None)や (0,0) を除外しつつ、同じword_idのトークン群の最小start・最大endを取る
        word_bounds: Dict[int, Tuple[int, int]] = {}
        for wid, (s, e) in zip(word_ids, offsets):
            if wid is None:
                continue
            if s == e:
                continue
            if wid not in word_bounds:
                word_bounds[wid] = (s, e)
            else:
                c_s, c_e = word_bounds[wid]
                word_bounds[wid] = (min(c_s, s), max(c_e, e))

        # word_id の昇順で整列
        spans = [word_bounds[k] for k in sorted(word_bounds.keys()) if word_bounds[k][0] < word_bounds[k][1]]
        # 空白や制御文字だけの語は除外（基本発生しない想定）
        spans = [(s, e) for (s, e) in spans if any(not text[i].isspace() for i in range(s, e))]
        return spans

    def _pick_one_O_word_span(self, text: str, entities: List[Entity], rng: random.Random) -> Optional[Tuple[int, int]]:
        bio = char_bio(text, entities)
        word_spans = self._get_word_spans(text)
        # 完全に O で覆われる語スパンのみ候補
        cand = []
        for s, e in word_spans:
            if s < 0 or e > len(text) or s >= e:
                continue
            if all(bio[i] == "O" for i in range(s, e)):
                cand.append((s, e))
        return rng.choice(cand) if cand else None

    def _replace_word(self, sample: Sample, rng: random.Random, top_k: int = 5) -> Optional[Sample]:
        span = self._pick_one_O_word_span(sample.text, sample.entities, rng)
        if span is None:
            return None
        s, e = span
        # 語全体を1個の[MASK]に置換（複数[MASK]よりも堅牢）
        masked_text = sample.text[:s] + self.mask_token + sample.text[e:]
        try:
            preds = self.unmasker(masked_text, top_k=top_k)
        except Exception:
            return None
        if not preds:
            return None
        original = sample.text[s:e]
        candidate = None
        for p in preds:
            tok = p["token_str"]
            if tok and tok.strip() and tok != original:
                candidate = tok
                break
        candidate = candidate or preds[0]["token_str"]
        new_text, new_entities = update_spans_after_replacement(sample.text, sample.entities, (s, e), candidate)
        return Sample(sample.curid, new_text, new_entities)

    # ---------- public ----------
    def replace_one(self, sample: Sample, rng: random.Random, top_k: int = 5) -> Optional[Sample]:
        if self.unit == "char":
            return self._replace_char(sample, rng, top_k=top_k)
        else:
            return self._replace_word(sample, rng, top_k=top_k)

# ===================== Entity-level =====================

def choose_alt_entity(entity_base: Dict[str, List[str]], ent_type: str, current: str, rng: random.Random) -> Optional[str]:
    cand = [w for w in entity_base.get(ent_type, []) if w != current]
    return rng.choice(cand) if cand else None

def entity_level_replace(sample: Sample, entity_base: Dict[str, List[str]], rng: random.Random) -> Optional[Sample]:
    if not sample.entities:
        return None
    ent = rng.choice(sample.entities)
    alt = choose_alt_entity(entity_base, ent.type, ent.name, rng)
    if not alt:
        return None
    s, e = ent.span
    new_text, shifted_entities = update_spans_after_replacement(sample.text, sample.entities, (s, e), alt)
    # 置換対象エンティティの name/span を更新
    final_ents: List[Entity] = []
    for en in shifted_entities:
        if en is ent:
            final_ents.append(Entity(alt, (s, s + len(alt)), ent.type))
        else:
            final_ents.append(en)
    return Sample(sample.curid, new_text, final_ents)

# ===================== Filtering =====================

def to_char_bio_from_ner_pipeline(text: str, ner: TokenClassificationPipeline) -> List[str]:
    out = ner(text)
    labels = ["O"] * len(text)
    for ent in out:
        start = int(ent.get("start", 0))
        end = int(ent.get("end", 0))
        t = ent.get("entity_group") or ent.get("entity") or "ENT"
        if start < 0 or end > len(text) or start >= end:
            continue
        labels[start] = f"B-{t}"
        for i in range(start + 1, end):
            labels[i] = f"I-{t}"
    return labels

def filtering_pass(
    sample: Sample,
    ner_pipeline: Optional[TokenClassificationPipeline],
    type_mapping: Optional[Dict[str, str]] = None
) -> bool:
    if ner_pipeline is None:
        return True
    gold = char_bio(sample.text, sample.entities)
    pred = to_char_bio_from_ner_pipeline(sample.text, ner_pipeline)
    if type_mapping:
        mapped = []
        for lab in gold:
            if lab == "O":
                mapped.append("O")
            else:
                prefix, typ = lab.split("-", 1)
                mapped.append(f"{prefix}-{type_mapping.get(typ, typ)}")
        gold = mapped
    return len(gold) == len(pred) and all(g == p for g, p in zip(gold, pred))

# ===================== Main FactMix =====================

def factmix_augment(
    dataset: List[Sample],
    ctx_aug: Optional[ContextAugmentor],
    entity_base: Dict[str, List[str]],
    ner_pipeline: Optional[TokenClassificationPipeline],
    ratio_context: int = 5,
    ratio_entity: int = 8,
    seed: int = 42,
    type_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, List[Sample]]:
    rng = random.Random(seed)
    orig = dataset
    c_semi: List[Sample] = []
    e_semi: List[Sample] = []

    # Context-level
    if ctx_aug is not None and ratio_context > 0:
        target = len(orig) * ratio_context
        trials = 0
        while len(c_semi) < target and trials < target * 5:
            trials += 1
            s = rng.choice(orig)
            aug = ctx_aug.replace_one(s, rng)
            if aug is None:
                continue
            if filtering_pass(aug, ner_pipeline, type_mapping):
                c_semi.append(aug)

    # Entity-level
    if ratio_entity > 0:
        target_e = len(orig) * ratio_entity
        trials_e = 0
        while len(e_semi) < target_e and trials_e < target_e * 5:
            trials_e += 1
            s = rng.choice(orig)
            aug = entity_level_replace(s, entity_base, rng)
            if aug is None:
                continue
            if filtering_pass(aug, ner_pipeline, type_mapping):
                e_semi.append(aug)

    mix = orig + c_semi + e_semi
    return {"orig": orig, "c_semi": c_semi, "e_semi": e_semi, "mix": mix}

# ===================== CLI =====================

def main():
    ap = argparse.ArgumentParser(description="FactMix augmentation (context-level + entity-level)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_orig", default="orig.json")
    ap.add_argument("--out_csemi", default="c_semi.json")
    ap.add_argument("--out_esemi", default="e_semi.json")
    ap.add_argument("--out_mix", default="mix.json")

    ap.add_argument("--mlm_model", default=None, help="Masked LM id/path (e.g., cl-tohoku/bert-base-japanese-v2)")
    ap.add_argument("--ner_model", default=None, help="NER model id/path for filtering (optional)")
    ap.add_argument("--device", type=int, default=-1)
    ap.add_argument("--mask_unit", choices=["char", "word"], default="word", help="Context masking unit (default: word)")

    ap.add_argument("--ratio_context", type=int, default=5)
    ap.add_argument("--ratio_entity", type=int, default=8)
    ap.add_argument("--type_mapping_json", default=None, help='e.g., {"その他の組織名":"ORG","人名":"PER"}')
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = load_dataset(args.input)
    ent_base = build_entity_base(data)

    ctx_aug = None
    if args.mlm_model:
        ctx_aug = ContextAugmentor(args.mlm_model, device=args.device, unit=args.mask_unit)

    ner_pl = None
    if args.ner_model:
        ner_pl = pipeline("ner", model=args.ner_model, tokenizer=args.ner_model, aggregation_strategy="simple", device=args.device)

    type_mapping = json.loads(args.type_mapping_json) if args.type_mapping_json else None

    out = factmix_augment(
        dataset=data,
        ctx_aug=ctx_aug,
        entity_base=ent_base,
        ner_pipeline=ner_pl,
        ratio_context=args.ratio_context,
        ratio_entity=args.ratio_entity,
        seed=args.seed,
        type_mapping=type_mapping
    )

    dump_dataset(args.out_orig, out["orig"])
    dump_dataset(args.out_csemi, out["c_semi"])
    dump_dataset(args.out_esemi, out["e_semi"])
    dump_dataset(args.out_mix, out["mix"])
    print(f"Saved: {args.out_orig}, {args.out_csemi}, {args.out_esemi}, {args.out_mix}")

if __name__ == "__main__":
    main()
