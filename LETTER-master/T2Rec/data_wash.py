import pandas as pd
import json

def parse_review(input_path, output_path):
    reviews = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line)
            reviews.append({
                'user_id': review['reviewerID'],
                'item_id': review['asin'],
                'date': review['reviewTime']
            })
    df = pd.DataFrame(reviews)
    df['time'] = pd.to_datetime(df['date']).astype('int64') // 10**9
    df = df[['user_id', 'item_id', 'time']]
    df.to_csv(output_path, index=False)

parse_review(
    input_path="/root/autodl-tmp/Beauty_5/Beauty_5.json",
    output_path="/root/autodl-tmp/Beauty_5/Beauty_5.csv"
)