from flr.pipeline import Pipeline

pipeline = Pipeline(config_path = "config/pipelines/example_load_from_disk.yml")
pipeline.fit()
score = pipeline.evaluate()
print("Score:", score)
print("Done.")