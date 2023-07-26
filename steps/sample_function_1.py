import click
import mlflow
import inspect
import os

from model_factory.core.data_access_object.data import read_csv, write_csv


@click.command(help="sample function 1")
@click.option(
    "--data_path",
    type=click.STRING,
    help="Dataset folder path",
)
@click.option(
    "--storage_path",
    type=click.STRING,
    default="file://./storage/",
    help="local tmp storage path",
)
def sample_function_1(data_path, storage_path):
    mlflow.set_tag("recipe", "sample-recipe")
    mlflow.set_tag("step", inspect.stack()[0][3])

    df = read_csv(data_path)
    df = df.dropna()

    processed_data_path = os.path.join(storage_path, 'processed_data.csv')
    write_csv(processed_data_path, df)

    output = dict()
    output["processed_data_path"] = processed_data_path
    mlflow.log_dict(output, "output_config.json")


if __name__ == "__main__":
    sample_function_1()
