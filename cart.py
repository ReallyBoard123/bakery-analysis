#!/usr/bin/env python3

import os
import pandas as pd
import ast
from pathlib import Path

CART_IDS = ['7004', '7005', '7006', '7007', '7008', '7009', '7010']

def parse_while_using(value):
    if pd.isna(value) or value == '' or value == '[]':
        return []
    try:
        return [str(item) for item in ast.literal_eval(value)]
    except:
        return []

def main():
    data = pd.read_csv('data/raw/processed_sensor_data.csv')
    data['whileUsing_parsed'] = data['whileUsing'].apply(parse_while_using)
    data_sorted = data.sort_values(['date', 'startTime'])
    
    # Use the environment variable if set, otherwise default to output/cart_data
    output_dir = Path(os.environ.get('CART_DATA_DIR', 'output/cart_data'))
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"📂 Saving cart data to: {output_dir.absolute()}")
    
    cart_last_info = {cart_id: {'last_person': None, 'last_region': None} for cart_id in CART_IDS}
    
    for cart_id in CART_IDS:
        cart_records = []
        
        for _, row in data_sorted.iterrows():
            if cart_id in row['whileUsing_parsed']:
                last_person = cart_last_info[cart_id]['last_person']
                last_region = cart_last_info[cart_id]['last_region']
                
                cart_records.append({
                    'cartId': cart_id,
                    'current_person_id': row['id'],
                    'last_person_id': last_person if last_person else 'None',
                    'shift': row['shift'],
                    'date': row['date'],
                    'startTime': round(float(row['startTime']), 3),
                    'endTime': round(float(row['endTime']), 3),
                    'duration': round(float(row['duration']), 3),
                    'region': row['region'],
                    'isPreviousRegionSameAsCurrentRegion': last_region == row['region'] if last_region else True,
                    'activity': row['activity']
                })
                
                cart_last_info[cart_id]['last_person'] = row['id']
                cart_last_info[cart_id]['last_region'] = row['region']
        
        if cart_records:
            pd.DataFrame(cart_records).to_csv(output_dir / f'{cart_id}_data.csv', index=False)

if __name__ == "__main__":
    main()