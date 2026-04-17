import json
import ast

# ====== 把这里改成你的原始 Amazon_Games json 文件路径 ======
file_path = "/root/autodl-tmp/Amazon_Games/meta_Games.json"
# 例如也可能是:
# file_path = "/autodl-pub/data/Amazon_Games/Amazon_Games.json"
# file_path = "/root/autodl-fs/Amazon_Games/Amazon_Games_review.json"

PRINT_LINES = 100


def parse_line(line: str):
    """
    优先按标准 JSON 解析；
    如果失败，再尝试按 Python 字典格式解析。
    """
    line = line.strip()
    if not line:
        return None

    try:
        return json.loads(line)
    except Exception:
        try:
            return ast.literal_eval(line)
        except Exception:
            return None


def main():
    first_100_with_title = 0
    first_100_valid = 0

    total_lines = 0
    total_valid = 0
    total_with_title = 0

    print("=" * 100)
    print(f"开始检查文件: {file_path}")
    print("=" * 100)

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            total_lines += 1

            obj = parse_line(line)
            if isinstance(obj, dict):
                total_valid += 1
                if "title" in obj and obj["title"] not in [None, ""]:
                    total_with_title += 1

            # 打印前100行
            if idx <= PRINT_LINES:
                print(f"\n----- 第 {idx} 行原始内容 -----")
                print(line.rstrip())

                if isinstance(obj, dict):
                    first_100_valid += 1
                    has_title = ("title" in obj and obj["title"] not in [None, ""])
                    if has_title:
                        first_100_with_title += 1

                    print(f"[解析成功] keys = {list(obj.keys())}")
                    print(f"[是否有 title] {has_title}")
                    if "title" in obj:
                        print(f"[title 值] {repr(obj.get('title'))}")
                else:
                    print("[解析失败] 这一行不是合法 JSON / Python dict 格式")

    print("\n" + "=" * 100)
    print("检查结果汇总")
    print("=" * 100)
    print(f"前 {PRINT_LINES} 行中，成功解析的行数: {first_100_valid}")
    print(f"前 {PRINT_LINES} 行中，包含 title 的行数: {first_100_with_title}")

    if first_100_with_title == 0:
        print(f"结论1：前 {PRINT_LINES} 行里没有发现 title 字段。")
    else:
        print(f"结论1：前 {PRINT_LINES} 行里发现了 title 字段，不是完全没有。")

    print("-" * 100)
    print(f"全文件总行数: {total_lines}")
    print(f"全文件成功解析的行数: {total_valid}")
    print(f"全文件包含 title 的行数: {total_with_title}")

    if total_with_title == 0:
        print("结论2：整个文件里都没有 title 字段。")
    else:
        print("结论2：整个文件里存在 title 字段，只是可能不是每一行都有。")


if __name__ == "__main__":
    main()