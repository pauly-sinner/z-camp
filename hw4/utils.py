
import pickle
import pandas as pd
from constants import MODEL_PATH, CATEGORICAL, DATA_PATH, OUTPUT_PATH

def load_model(model_path):
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype('int').astype('str')

    return df

def upload_predictions(df, y_pred, output_file):

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predictions'] = y_pred

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
