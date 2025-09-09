import json
import os
from collections import Counter
from pathlib import Path
import glob

def count_entity_tags(json_dir):
    """
    指定されたディレクトリ内の全JSONファイルからエンティティタグをカウントする
    
    Args:
        json_dir (str): JSONファイルが格納されているディレクトリのパス
    
    Returns:
        Counter: エンティティタグとその出現回数のカウンター
    """
    tag_counter = Counter()
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    print(f"処理対象ファイル数: {len(json_files)}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 各文のエンティティからタグを抽出
            for sentence in data:
                if 'entities' in sentence:
                    for entity in sentence['entities']:
                        if 'type' in entity:
                            tag_counter[entity['type']] += 1
            
            print(f"処理完了: {os.path.basename(json_file)}")
            
        except Exception as e:
            print(f"エラー発生 - {json_file}: {e}")
    
    return tag_counter

def save_tag_counts(tag_counter, output_file):
    """
    タグカウント結果をファイルに保存する
    
    Args:
        tag_counter (Counter): タグカウンター
        output_file (str): 出力ファイルのパス
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("エンティティタグ,出現回数\n")
        for tag, count in tag_counter.most_common():
            f.write(f"{tag},{count}\n")

def save_category_counts(categories, output_file):
    """
    カテゴリカウント結果をファイルに保存する
    
    Args:
        categories (dict): カテゴリカウンター
        output_file (str): 出力ファイルのパス
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("エンティティカテゴリ,出現回数\n")
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{category},{count}\n")

def main():
    # JSONファイルのディレクトリを指定
    json_directory = "datasets/icorpus_20220531 2/data/json/"
    
    print("エンティティタグのカウントを開始します...")
    
    # タグをカウント
    tag_counts = count_entity_tags(json_directory)
    
    # すべてのタグでハイフンの最初の部分をカテゴリとして集計
    categories = {}
    for tag, count in tag_counts.items():
        # ハイフンがある場合は最初の部分、ない場合はそのまま使用
        category = tag.split('-')[0]
        if category not in categories:
            categories[category] = 0
        categories[category] += count
    
    # 結果を表示
    print(f"\n=== エンティティカテゴリカウント結果 ===")
    print(f"総カテゴリ種類数: {len(categories)}")
    print(f"総エンティティ数: {sum(categories.values())}")
    print("\n上位20位:")
    
    for i, (category, count) in enumerate(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:20], 1):
        print(f"{i:2d}. {category:<20} : {count:>6,}回")
    
    # カテゴリ別結果をCSVファイルに保存
    output_file = "entity_category_counts.csv"
    save_category_counts(categories, output_file)
    print(f"\nカテゴリ別集計結果を {output_file} に保存しました。")
    
    # 追加の統計情報
    sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    print(f"\n=== 追加統計 ===")
    print(f"最も多いカテゴリ: {sorted_categories[0][0]} ({sorted_categories[0][1]:,}回)")
    print(f"最も少ないカテゴリ: {sorted_categories[-1][0]} ({sorted_categories[-1][1]}回)")

if __name__ == "__main__":
    main()