import json
import random
from copy import deepcopy
from typing import List, Dict, Union, Tuple, Optional

def _sanitize_span(s: int, e: int, n: int) -> Tuple[int, int]:
    s = max(0, min(s, n))
    e = max(0, min(e, n))
    if s > e:
        s, e = e, s
    return s, e

def _mask_per_entity_with_updates(
    text: str,
    entities: List[Dict],
    mask_token: str = "[MASK]"
) -> Tuple[str, List[Optional[Tuple[int, int]]]]:
    """
    エンティティごとに [MASK] を挿入していき、各エンティティの span を更新する。
    - 左→右で処理し、都度生じるオフセット差分(delta)を考慮して置換。
    - 交差/重複するエンティティは、先に処理した置換区間に飲み込まれるため、
      後続のエンティティ span は、その直近の [MASK] 区間に合わせる（=同一スパン）。
    戻り値: (置換後テキスト, 各エンティティの新 span（該当なしは None）)
    """
    n = len(text)
    # 元順を保ったまま、開始位置で安定ソート
    idx_and_spans = []
    for i, ent in enumerate(entities):
        span = ent.get("span")
        if not (isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span)):
            idx_and_spans.append((i, None))
            continue
        s, e = _sanitize_span(span[0], span[1], n)
        idx_and_spans.append((i, (s, e)))

    # 有効なものだけ取り出して並べ替え
    valid = [(i, se) for i, se in idx_and_spans if se is not None]
    valid.sort(key=lambda x: (x[1][0], x[1][1]))  # start, end

    # 置換を左→右で進める
    out = []
    cursor = 0
    delta = 0  # 累積の長さ差分
    # 「直近で作った [MASK] の新座標」を覚えておき、重複はそこへ吸収
    last_mask_new_span: Optional[Tuple[int, int]] = None
    # 各エンティティの新 span（元の順に戻して格納）
    new_spans: List[Optional[Tuple[int, int]]] = [None] * len(entities)

    for _, (s, e) in valid:
        # 現在テキスト上での有効位置（これまでの置換の影響を加味）
        s_cur = s + delta
        e_cur = e + delta

        # すでに直前の [MASK] 区間に重なるなら、置換はせず、同じスパンに吸収
        if last_mask_new_span is not None:
            lm_s, lm_e = last_mask_new_span
            if s_cur < lm_e:  # 重複・交差
                # このエンティティは最後の [MASK] 区間に乗せる
                # 実際のテキスト変更は行わない
                # どのエンティティかは後で設定するため、ここではスパンだけ覚える
                pass
            else:
                # 重ならない：新たに置換を追加
                out.append(text[cursor:s + delta])  # 未出力部分（元座標に delta を足さないよう注意）
                # ↑注意: out には「元テキスト上の未出力部分」を入れる必要がある。
                # ここで cursor は「元テキスト上の位置」を指し続けるように実装する。
                # したがって s+delta ではなく s を使いたいが、これまでの out 連結と整合性をとるため、
                # 方針を変える：out 構築は delta を使わず、逐次スライスで管理する。
                # ---- 方針調整 ----

    # ---- out 構築の方針をシンプルにやり直し ----

def _mask_per_entity_with_updates(
    text: str,
    entities: List[Dict],
    mask_token: str = "[MASK]"
) -> Tuple[str, List[Optional[Tuple[int, int]]]]:
    n = len(text)
    idx_and_spans = []
    for i, ent in enumerate(entities):
        span = ent.get("span")
        if not (isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span)):
            idx_and_spans.append((i, None))
            continue
        s, e = _sanitize_span(span[0], span[1], n)
        idx_and_spans.append((i, (s, e)))

    valid = [(i, se) for i, se in idx_and_spans if se is not None]
    valid.sort(key=lambda x: (x[1][0], x[1][1]))

    # 新テキストを文字列ビルドしつつ、逐次置換で delta を管理
    new_text = text
    delta = 0
    last_mask_span_global: Optional[Tuple[int, int]] = None  # new_text 上の直近マスク範囲
    new_spans: List[Optional[Tuple[int, int]]] = [None] * len(entities)

    for i, (s, e) in valid:
        s_adj = s + delta
        e_adj = e + delta
        # 直近マスクと重なる？
        if last_mask_span_global is not None and s_adj < last_mask_span_global[1]:
            # このエンティティは直近マスクに吸収
            new_spans[i] = last_mask_span_global
            continue

        # 実置換
        new_text = new_text[:s_adj] + mask_token + new_text[e_adj:]
        new_span = (s_adj, s_adj + len(mask_token))
        new_spans[i] = new_span
        # delta 更新（置換で長さが変化）
        delta += len(mask_token) - (e - s)
        last_mask_span_global = new_span

    return new_text, new_spans

def mask_entities_for_type_prediction(
    dataset: Union[str, List[Dict]],
    percentage: float,
    seed: int = 42,
    mask_token: str = "[MASK]",
    update_spans: bool = True,
    entity_name_policy: str = "drop",  # "drop" | "mask" | "keep"
    augment_data: bool = True,  # デフォルトをTrueに変更
) -> List[Dict]:
    """
    目的：エンティティの表層を [MASK] に置換しつつ、各エンティティの span/type を残して
          「マスク状態でタイプ予測」を学習できるデータにする。

    - percentage% のレコードをランダムにマスク
    - augment_data=True のとき、元データを保持してマスキング版を追加（データ拡張）
    - augment_data=False のとき、従来通り選択されたレコードを置換
    - update_spans=True のとき、置換後テキストに合わせて各エンティティ span を更新
      （重複/交差は先行置換に吸収。後続は同じ [MASK] span を参照）
    - entity_name_policy:
        - "drop": entities[].name を削除（デフォルト・リーク防止）
        - "mask": entities[].name を "[MASK]" に統一
        - "keep": そのまま保持（MLM 的な補助タスク用途）
    """
    if isinstance(dataset, str):
        data = json.loads(dataset)
    else:
        data = dataset
    
    original_data = deepcopy(data)  # 元データの完全なコピー
    
    n = len(original_data)
    k = round(n * (percentage / 100.0))
    rng = random.Random(seed)
    target_indices = set(rng.sample(range(n), k))

    # マスキング処理されたレコードを格納するリスト
    masked_records = []

    for idx, rec in enumerate(original_data):
        if idx not in target_indices:
            continue

        # レコードのコピーを作成してマスキング処理
        masked_rec = deepcopy(rec)
        text = masked_rec.get("text", "")
        ents = masked_rec.get("entities", []) or []
        if not text or not ents:
            continue

        if update_spans:
            new_text, new_spans = _mask_per_entity_with_updates(text, ents, mask_token=mask_token)
            masked_rec["text"] = new_text
            # 各エンティティへ新 span を反映
            for ent, ns in zip(ents, new_spans):
                # name の扱い
                if entity_name_policy == "drop":
                    ent.pop("name", None)
                elif entity_name_policy == "mask":
                    ent["name"] = mask_token
                # keep の場合は何もしない

                # span 更新
                if ns is not None:
                    ent["span"] = [ns[0], ns[1]]
                else:
                    last = next((s for s in reversed(new_spans) if s is not None), None)
                    if last is not None:
                        ent["span"] = [last[0], last[1]]

        else:
            # 右→左で置換し、元 span は不変
            spans = []
            for ent in ents:
                sp = ent.get("span")
                if isinstance(sp, list) and len(sp) == 2 and all(isinstance(x, int) for x in sp):
                    spans.append((sp[0], sp[1]))
            spans.sort(key=lambda x: x[0], reverse=True)
            new_text = text
            for s, e in spans:
                s, e = _sanitize_span(s, e, len(new_text))
                new_text = new_text[:s] + mask_token + new_text[e:]
            masked_rec["text"] = new_text

            # name の扱いのみ反映
            for ent in ents:
                if entity_name_policy == "drop":
                    ent.pop("name", None)
                elif entity_name_policy == "mask":
                    ent["name"] = mask_token

        masked_records.append(masked_rec)

    if augment_data:
        # データ拡張モード：元データ + マスキング版
        return original_data + masked_records
    else:
        # 従来モード：選択されたレコードを置換
        result_data = deepcopy(original_data)
        masked_idx = 0
        for idx in range(len(result_data)):
            if idx in target_indices:
                result_data[idx] = masked_records[masked_idx]
                masked_idx += 1
        return result_data

# 使い方例：
if __name__ == "__main__":
    import argparse, pathlib
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", type=pathlib.Path, required=True)
    p.add_argument("-o", "--output", type=pathlib.Path, required=True)
    p.add_argument("-p", "--percent", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask-token", type=str, default="[MASK]")
    p.add_argument("--entity-name-policy", type=str, choices=["drop", "mask", "keep"], default="mask")
    p.add_argument("--no-update-spans", action="store_true", help="指定すると spans を更新しない")
    p.add_argument("--no-augment", action="store_true", help="データ拡張を無効化：従来通り選択されたレコードを置換")
    args = p.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    out = mask_entities_for_type_prediction(
        raw,
        percentage=args.percent,
        seed=args.seed,
        mask_token=args.mask_token,
        update_spans=not args.no_update_spans,
        entity_name_policy=args.entity_name_policy,
        augment_data=not args.no_augment,  # --no-augmentの逆を渡す
    )
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)