import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a pairwise ranking model.")
    parser.add_argument("--workspace_path", type=str, default= "workspaces/test", help="Path to save the workspace.")
    parser.add_argument("--dataset_name", type=str,default= "llm-blender/mix-instruct", help="Name of the dataset.")
    parser.add_argument("--test_size", type=float, default=0.001, help="Size of the test set.")
    parser.add_argument("--prompt_embedder_name", type=str, default="albert-base-v2", help="Name of the prompt embedder.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--hidden_size", type=int, default=32, help="Hidden size of the model.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--loss_fun_name", type=str, default="list_mle", help="Loss function to use.")
    parser.add_argument("--list_size", type=int, default=3, help="Size of the list for listwise ranking.")
    parser.add_argument("--model_hub_name", type=str, default="llm_instruct", help="Name of the model hub.")
    return parser.parse_args()