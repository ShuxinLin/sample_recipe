from pathlib import Path
import click
from model_factory.core.recipes import Recipe


def run_recipe(recipe="Recipe"):
    recipe_obj = Recipe.from_yaml(recipe)

    if recipe_obj.backend == "ray":
        import ray

        ray.init(address='auto')

    recipe_obj.run_recipe(
        project_uri=str(Path(__file__).parent),
        recipe_name="sample-recipe"
    )


@click.command(help="start workflow")
@click.option(
    "--recipe",
    type=click.STRING,
    default="Recipe",
    help="Recipe name file",
)
def start(recipe):
    run_recipe(recipe)


if __name__ == "__main__":
    start()
