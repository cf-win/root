import json
import pandas as pd
import os
import time


def get_label_amazon_legacy(review_data):
    """
    针对 2014/2018 Amazon 数据集
    字段格式: "helpful": [helpful_votes, total_votes]
    """
    helpful_list = review_data.get("helpful", None)

    # 防御：字段缺失或格式不对
    if not (isinstance(helpful_list, (list, tuple)) and len(helpful_list) >= 2):
        return -1

    try:
        helpful_votes = int(float(helpful_list[0]))
        total_votes = int(float(helpful_list[1]))
    except (ValueError, TypeError):
        return -1

    helpful_votes = max(0, helpful_votes)
    total_votes = max(0, total_votes)

    if total_votes == 0:
        return -1

    # 防御：helpful_votes 不应超过 total_votes
    helpful_votes = min(helpful_votes, total_votes)

    ratio = helpful_votes / total_votes

    if ratio > 0.7:
        return 1  # Genuine
    elif ratio < 0.3:
        return 0  # Malicious
    else:
        return -1  # 中间状态


def read_csv_with_retry(csv_path, max_retries=3, delay=2):
    for retry in range(max_retries):
        try:
            return pd.read_csv(csv_path, encoding="utf-8-sig")
        except PermissionError:
            if retry < max_retries - 1:
                time.sleep(delay)
            else:
                raise
        except Exception:
            raise


def add_label_to_csv(json_path, csv_path, output_csv_path):
    labels = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    review_data = json.loads(line.strip())
                    labels.append(get_label_amazon_legacy(review_data))
                except json.JSONDecodeError:
                    labels.append(-1)
                except Exception:
                    labels.append(-1)
    except FileNotFoundError:
        return
    except Exception:
        return

    try:
        df = read_csv_with_retry(csv_path)

        # 行数对齐
        if len(df) != len(labels):
            min_length = min(len(df), len(labels))
            df = df.iloc[:min_length]
            labels = labels[:min_length]

        df["label"] = labels

        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        except PermissionError:
            pass
        except Exception:
            pass

    except FileNotFoundError:
        return
    except PermissionError:
        pass
    except Exception:
        return


if __name__ == "__main__":
    json_file_path = "/root/autodl-tmp/Beauty_5/Beauty_5.json"
    csv_file_path = "/root/autodl-tmp/Beauty_5/Beauty_5_filled.csv"
    new_output_path = "/root/autodl-tmp/Beauty_5/Beauty_5_filled_label.csv"

    add_label_to_csv(json_file_path, csv_file_path, new_output_path)
