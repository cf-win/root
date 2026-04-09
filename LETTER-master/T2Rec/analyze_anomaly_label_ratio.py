import argparse
from types import SimpleNamespace

from data_t2rec import AnomalyRecDataset


def ratio_line(yes_count, no_count):
    total = yes_count + no_count
    if total == 0:
        return "Yes=0 (0.00%), No=0 (0.00%), total=0"
    yes_ratio = 100.0 * yes_count / total
    no_ratio = 100.0 * no_count / total
    return f"Yes={yes_count} ({yes_ratio:.2f}%), No={no_count} ({no_ratio:.2f}%), total={total}"


def count_user_labels_from_inter_data(inter_data):
    user_to_label = {}
    conflict_users = 0
    for d in inter_data:
        uid = str(d.get("user_id", ""))
        lbl = d.get("anomaly_label", "No")
        if uid in user_to_label and user_to_label[uid] != lbl:
            conflict_users += 1
        else:
            user_to_label[uid] = lbl

    yes_count = sum(1 for v in user_to_label.values() if v == "Yes")
    no_count = sum(1 for v in user_to_label.values() if v == "No")
    return yes_count, no_count, len(user_to_label), conflict_users


def count_sample_labels(inter_data):
    yes_count = sum(1 for d in inter_data if d.get("anomaly_label", "No") == "Yes")
    no_count = len(inter_data) - yes_count
    return yes_count, no_count, len(inter_data)


def build_dataset_args(cli_args):
    return SimpleNamespace(
        dataset=cli_args.dataset,
        data_path=cli_args.data_path,
        max_his_len=cli_args.max_his_len,
        his_sep=", ",
        index_file=cli_args.index_file,
        add_prefix=False,
        task="simple_rec",
        top_k=cli_args.top_k,
        use_title=False,
        graph_token_path=cli_args.graph_token_path,
        behavior_token_path=cli_args.behavior_token_path,
    )


def print_split_report(name, ds):
    uy, un, utotal, conflicts = count_user_labels_from_inter_data(ds.inter_data)
    sy, sn, stotal = count_sample_labels(ds.inter_data)

    print(f"[{name}] user-level   : {ratio_line(uy, un)}")
    print(f"[{name}] sample-level : {ratio_line(sy, sn)}")
    if conflicts > 0:
        print(f"[{name}] warning      : found {conflicts} user label conflicts in inter_data")
    print(f"[{name}] user_count={utotal}, sample_count={stotal}")


def main():
    parser = argparse.ArgumentParser(description="Analyze anomaly_label Yes/No ratio by users and samples")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_his_len", type=int, default=20)
    parser.add_argument("--index_file", type=str, default=".index.json")
    parser.add_argument("--graph_token_path", type=str, default="")
    parser.add_argument("--behavior_token_path", type=str, default="")
    args = parser.parse_args()

    ds_args = build_dataset_args(args)

    train_ds = AnomalyRecDataset(ds_args, mode="train", prompt_sample_num=1, sample_num=-1)
    valid_ds = AnomalyRecDataset(ds_args, mode="valid", prompt_sample_num=1, sample_num=-1)
    test_ds = AnomalyRecDataset(ds_args, mode="test", prompt_sample_num=1, sample_num=-1)

    raw_user_yes = sum(1 for v in train_ds.anomaly_labels.values() if v == "Yes")
    raw_user_no = sum(1 for v in train_ds.anomaly_labels.values() if v == "No")
    raw_total = len(train_ds.anomaly_labels)

    print("=== Raw User Label Distribution (before split views) ===")
    print(f"[all-users] {ratio_line(raw_user_yes, raw_user_no)}")
    print(f"[all-users] user_count={raw_total}")
    print()

    print("=== Split Distribution ===")
    print_split_report("train", train_ds)
    print_split_report("valid", valid_ds)
    print_split_report("test", test_ds)


if __name__ == "__main__":
    main()
