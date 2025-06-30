from flr.hub.model_hub import ModelHub

if __name__ == "__main__":
    model_hub = ModelHub("llm_instruct")
    print("Available models:")
    for model in model_hub.get_models():
        print(model)
    print("\nModel attributes:")