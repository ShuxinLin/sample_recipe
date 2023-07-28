import click
import mlflow
import mlflow.sklearn
import inspect
import os
from sklearn.linear_model import LogisticRegression

from model_factory.core.data_access_object.data import read_csv, write_csv


@click.command(help="sample function 2")
@click.option(
    "--processed_data_path",
    type=click.STRING,
    help="Processed Dataset path",
)
@click.option(
    "--feature_columns",
    type=click.STRING,
    help="Feature columns separated by ,",
)
@click.option(
    "--target_columns",
    type=click.STRING,
    help="Target columns separated by ,",
)
@click.option(
    "--log_model",
    type=click.STRING,
    help="Whether to log model",
)
def sample_function_2(processed_data_path, feature_columns, target_columns, log_model):
    mlflow.set_tag("recipe", "sample-recipe")
    mlflow.set_tag("step", inspect.stack()[0][3])
    mlflow.set_tag("summary", "true")

    df = read_csv(processed_data_path)
    features = feature_columns.split(",")
    target = target_columns
    X = df[features]
    y = df[target]
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print("Score: %s" % score)
    mlflow.log_metric("score", score)

    if log_model == 'True':
        print("log the lr model...")
        mlflow.sklearn.log_model(lr, "lr_model")

    summary_dict = dict()
    summary_dict["score"] = score
    mlflow.log_dict(summary_dict, "summary.json")


if __name__ == "__main__":
    sample_function_2()
