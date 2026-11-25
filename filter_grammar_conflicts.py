import json
import os
import argparse


def is_grammatically_valid(sample):
    """
    Validates a counterfactual benchmark sample.
    Ensures that the sample has the required keys and that the
    syntactic slot (clf) and the target noun (noun_type) are not empty.
    """
    required_keys = ['clf', 'noun_type', 'context']
    for key in required_keys:
        if key not in sample or not sample[key]:
            return False

    # 可选：进一步检查 clf 和 noun_type 是否出现在 context 的特定槽位中
    # if sample['clf'] not in sample['context'] or sample['noun_type'] not in sample['context']:
    #     return False

    return True


def filter_grammar(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"[ERROR] File not found: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned = [s for s in data if is_grammatically_valid(s)]

    print(f"[INFO] Processed: {len(data)} -> {len(cleaned)} (Removed: {len(data) - len(cleaned)})")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"[SUCCESS] Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/double_blind_benchmark.json")
    parser.add_argument("--output", type=str, default="data/double_blind_benchmark.json")
    args = parser.parse_args()

    filter_grammar(args.input, args.output)