#!/usr/bin/env python
# coding: utf-8
import argparse
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year, month):
    # Assuming the path to data and model are correctly set
    data_path = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    model_path = 'model.bin'

    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)

    df = read_data(data_path)
    dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    mean_duration = y_pred.mean()
    print(f"Mean predicted duration: {mean_duration:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, help='Year of the data')
    parser.add_argument('--month', type=int, help='Month of the data')
    args = parser.parse_args()

    main(args.year, args.month)

