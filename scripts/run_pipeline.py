import argparse

from flr.pipeline import Pipeline

parser = argparse.ArgumentParser(description="Execute pipeline.")
parser.add_argument("--config_path", type=str, default="config/pipelines/example_flr_dataset.yml")
args = parser.parse_args()

pipeline = Pipeline(config_path = args.config_path)
pipeline.fit()
score = pipeline.evaluate()
print("Score:", score)
