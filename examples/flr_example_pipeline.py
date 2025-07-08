from flr.pipeline import Pipeline

pipeline = Pipeline(config_path = "config/pipelines/example_flr_dataset.yml")
pipeline.fit()
score = pipeline.evaluate()
print("Score:", score)
print("Done.")