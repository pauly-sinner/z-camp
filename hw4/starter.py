import click
from constants import MODEL_PATH, CATEGORICAL, DATA_PATH, OUTPUT_PATH
from utils import load_model, read_data, upload_predictions

@click.command()
@click.argument("year", type=click.STRING)
@click.argument("month", type=click.STRING)
def run(year: str, month: str):

    DATA_PATH = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet'
    OUTPUT_PATH = f'results_{year}-{month}.parquet'

    dv, model = load_model(MODEL_PATH)

    df = read_data(DATA_PATH)
    df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')

    dicts = df[CATEGORICAL].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    upload_predictions(df, y_pred, OUTPUT_PATH)

    print(y_pred.mean(), flush=True)
    print('DONE', flush=True)

if __name__ == "__main__":
    run()



