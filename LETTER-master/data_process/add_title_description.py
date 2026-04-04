import csv
import gzip
import ast
from typing import Dict, Tuple, Iterable, Any

INPUT_CSV = "/root/autodl-tmp/Amazon_Toys/Amazon_Toys.csv"
METADATA_PATH = "/root/autodl-tmp/Amazon_Toys/Amazon_Toys.json"
OUTPUT_CSV = "/root/autodl-tmp/Amazon_Toys/Amazon_Toys_filled.csv"

CSV_ITEM_ID_FIELD = "item_id"
META_ID_FIELD = "asin"
META_TITLE_FIELD = "title"
META_CATEGORIES_FIELD = "categories"  
META_SALES_RANK_FIELD = "salesRank"    
META_BRAND_FIELD = "brand"           

DEFAULT_TITLE = "No matched title"
DEFAULT_DESC = "No matched description"



def open_text_auto(path: str):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def flatten_categories(categories: Any) -> str:
    if not categories or not isinstance(categories, list):
        return ""

    paths = []
    for path in categories:
        if isinstance(path, (list, tuple)):
            segs = [normalize_text(s) for s in path if normalize_text(s)]
            if segs:
                paths.append(" > ".join(segs))
        else:
            s = normalize_text(path)
            if s:
                paths.append(s)

    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    return "; ".join(uniq)


def format_sales_rank(sales_rank: Any) -> str:
    if not sales_rank:
        return ""
    if isinstance(sales_rank, dict):
        parts = []
        for k, v in sales_rank.items():
            k2 = normalize_text(k)
            v2 = normalize_text(v)
            if k2 and v2:
                parts.append(f"{k2}: {v2}")
        return "; ".join(parts)
    return normalize_text(sales_rank)


def build_description(cat_str: str, brand: str, sales_rank_str: str) -> str:
    category_text = cat_str if cat_str else "Unknown category"
    brand_text = brand if brand else "Unknown brand"
    rank_text = sales_rank_str if sales_rank_str else "Unknown rank"

    sections = [
        f"[Category] {category_text}",
        f"[Brand] {brand_text}",
        f"[SalesRank] {rank_text}",
    ]
    return " | ".join(sections)


def iter_meta_objects(path: str) -> Iterable[dict]:
    with open_text_auto(path) as f:
        for _, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = ast.literal_eval(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def read_csv_rows_and_needed_ids(csv_path: str) -> Tuple[list, set, list]:
    rows = []
    needed = set()

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames_in = reader.fieldnames or []

        for row in reader:
            rows.append(row)
            item_id = normalize_text(row.get(CSV_ITEM_ID_FIELD))
            if item_id:
                needed.add(item_id)

    return rows, needed, fieldnames_in


def build_meta_map(meta_path: str, needed_ids: set) -> Tuple[Dict[str, Tuple[str, str]], dict]:
    meta_map: Dict[str, Tuple[str, str]] = {}
    remaining = set(needed_ids)

    stats = {
        "meta_total_dict_objects": 0,
        "meta_hit_needed": 0,
        "meta_missing_asin": 0,
        "meta_missing_title": 0,
        "needed_total": len(needed_ids),
        "needed_unmatched_after_scan": 0,
    }

    for obj in iter_meta_objects(meta_path):
        stats["meta_total_dict_objects"] += 1

        asin = normalize_text(obj.get(META_ID_FIELD))
        if not asin:
            stats["meta_missing_asin"] += 1
            continue
        if asin not in remaining:
            continue

        title = normalize_text(obj.get(META_TITLE_FIELD)) or "No product title"
        cat_str = flatten_categories(obj.get(META_CATEGORIES_FIELD))
        brand = normalize_text(obj.get(META_BRAND_FIELD))
        sales_rank_str = format_sales_rank(obj.get(META_SALES_RANK_FIELD))

        desc = build_description(cat_str=cat_str, brand=brand, sales_rank_str=sales_rank_str)

        meta_map[asin] = (title, desc)
        remaining.remove(asin)
        stats["meta_hit_needed"] += 1
        if not remaining:
            break

    stats["needed_unmatched_after_scan"] = len(remaining)
    return meta_map, stats


def fill_rows(rows: list, meta_map: Dict[str, Tuple[str, str]]) -> Tuple[list, int]:
    matched = 0
    for row in rows:
        item_id = normalize_text(row.get(CSV_ITEM_ID_FIELD))
        title, desc = meta_map.get(item_id, (DEFAULT_TITLE, DEFAULT_DESC))
        row["title"] = title
        row["description"] = desc
        if title != DEFAULT_TITLE:
            matched += 1
    return rows, matched


def write_csv(out_path: str, rows: list, fieldnames_in: list):
    out_fields = list(fieldnames_in)
    if "title" not in out_fields:
        out_fields.append("title")
    if "description" not in out_fields:
        out_fields.append("description")

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows, needed_ids, fieldnames_in = read_csv_rows_and_needed_ids(INPUT_CSV)
    print(f"[CSV] rows={len(rows)}, unique_item_ids={len(needed_ids)}")

    if not needed_ids:
        print("⚠️ No item_id found in CSV. Check column name or file format.")
        return

    meta_map, stats = build_meta_map(METADATA_PATH, needed_ids)
    print(f"[META] matched={len(meta_map)}/{stats['needed_total']}, missing_asin={stats['meta_missing_asin']}, missing_title={stats['meta_missing_title']}, unmatched={stats['needed_unmatched_after_scan']}")

    rows, matched_count = fill_rows(rows, meta_map)
    write_csv(OUTPUT_CSV, rows, fieldnames_in)

    rate = (matched_count / len(rows)) if rows else 0.0
    print(f"✅ Done: {OUTPUT_CSV}")
    print(f"✅ Match rate: {matched_count}/{len(rows)} = {rate:.2%}")


if __name__ == "__main__":
    main()
