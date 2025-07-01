from flr.pipeline import Pipeline

pipeline = Pipeline(config_path = "config/example_pipeline.yml")
pipeline.fit()

print("Done.")