import os
from typing import List, Dict

from collections import defaultdict
from pathlib import Path
import yaml

from .trainer import Trainer
from .trainer_args import TrainerArgs
from .router_evaluator import RouterEvaluator
from .router_evaluator_args import RouterEvaluatorArgs
from .hub.model_hub import ModelHub

from .scorers import *
from .callbacks import *
from .datasets import *
from .routers import *

class Pipeline:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.pipeline_config = self.load_pipeline_config()
        self.router = None

    def load_pipeline_config(self):
        """Load the pipeline configuration from a YAML file.
        Returns:
            dict: The loaded configuration.
        """
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.workspace_path = config.get("workspace_path", None)
        self.config_path = config.get("config_path", None)

        if self.workspace_path is None:
            self.workspace_path = Path(os.environ.get("LLUMUX_HOME", "")) / "workspace"
        else:
            self.workspace_path = Path(self.workspace_path)
        
        if self.config_path is None:
            self.config_path = Path(os.environ.get("LLUMUX_HOME", "")) / "config"
        else:
            self.config_path = Path(self.config_path)

        return config

    def get_callbacks(self, callbacks_args : dict):
        callbacks_dict = {}
        for args in callbacks_args:
            callbacks_dict.update(self.get_callback(**args))
        return callbacks_dict
    
    def get_callback(self, callback_class, callback_name, **kwargs):
        callback_class = eval(callback_class)
        return {callback_name: callback_class(**kwargs)}

    def get_scorer(self, scorer_class,
                   scorer_name,
                   model_list : List,
                   load_from_disk: bool = False,
                   **scorer_args):
        scorer_class = eval(scorer_class)
        scorer_path = self.workspace_path / "scorers" / scorer_name / "best_checkpoint.pt"
        
        if load_from_disk:
            scorer = scorer_class.from_checkpoint(scorer_path)
            scorer.trained = True
        else:
            scorer = scorer_class(model_list, **scorer_args)
            scorer.trained = False
        return scorer
    
    def get_dataset(self, dataset_name : str, 
                     dataset_class : BaseDataset,
                     split : str,
                     **dataset_args):
        """ Get the train, validation, and test datasets based on the dataset name.
        Args:
            dataset_name (str): Name of the dataset to load.
            dataset_class (class): The class of the dataset to instantiate.
            test_size (float): Proportion of the dataset to include in the test split.
            list_size (int): Size of the list for listwise datasets.
        Returns:
            tuple: A tuple containing the train, validation, and test datasets.
        """
        dataset_class = eval(dataset_class)
        dataset = dataset_class(dataset_name, split=split, **dataset_args)
        return dataset
    
    def get_datasets(self, model_hub_name, dataset_args : Dict):
        datasets = defaultdict(lambda:{})

        for args in dataset_args:
            splits = args.pop("splits", ["train", "validation"])
            propietary = args.pop("propietary")
            for split in splits:
                
                args.update({"model_hub_name":model_hub_name})
                dataset = self.get_dataset(split=split, **args)
                datasets[propietary][split] = dataset

        return datasets

    def get_scorers(self, model_hub : ModelHub,
                    scorers_args: List[dict]):
        """
        Get a list of scorers based on the provided arguments and datasets.
        Args:
            scorers_args (list): List of dictionaries containing arguments for each scorer.
            callbacks_args (list): List of dictionaries containing callbacks args.
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset): The validation dataset.
        Returns:
            list: A list of initialized Scorer instances.
        """
        scorers = {}
        for args in scorers_args:
            scorer_name = args["scorer_name"]
            scorer = self.get_scorer(model_list=model_hub.get_models(),
                                     **args)
            scorers[scorer_name] = scorer
        
        return scorers
        
    def get_trainer(self, scorer : BaseScorer, 
                    train_dataset : BaseDataset, 
                    val_dataset : BaseDataset, 
                    callbacks : List[BaseCallback],
                    **trainer_args):
        
        """Get a trainer instance for the given scorer and datasets.
        Args:
            scorer (Scorer): The scorer to train.
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset): The valuation dataset.
            callbacks (list): List of callbacks to use during training.
        Returns:
            Trainer: The configured trainer instance.
        """
        trainer_args = TrainerArgs(**trainer_args)
        return Trainer(scorer, trainer_args, train_dataset, val_dataset, callbacks)

    def get_router(self, scorers : Dict[str, BaseScorer], 
                   model_hub : ModelHub, 
                   router_class : str,
                   **router_args):
        """Get a router instance based on the provided arguments.
        Args:
            router_args (dict): Arguments for initializing the router.
        Returns:
            Router: The initialized router instance.
        """
        router_class = eval(router_class)
        router = router_class(scorers=scorers, model_hub=model_hub, **router_args)
        return router

    def train_scorer(self, scorer : BaseScorer, 
                     callbacks : Dict[str, BaseCallback], 
                     train_dataset : BaseDataset, 
                     val_dataset : BaseDataset,
                     **trainer_args):
        """ Train a scorer based on the provided arguments and datasets.

        Args:
            scorer_args (dict): Arguments for initializing the scorer.
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset): The validation dataset.
        Returns:
            Scorer: The trained scorer instance.
        """
        trainer = self.get_trainer(scorer, train_dataset, val_dataset, 
                                   callbacks.values(), **trainer_args)
        trainer.train()

    def fit(self):
        """
        Fit the pipeline by loading datasets, initializing scorers, router, and router evaluator.
        Args:
            None
        """
        if not self.pipeline_config:
            raise ValueError("Pipeline configuration is not loaded. Please check the config path.")
        
        model_hub_name = self.pipeline_config["model_hub_name"]
        model_hub = ModelHub(model_hub_name)
        callbacks = self.get_callbacks(self.pipeline_config["callbacks"])

        self.datasets = self.get_datasets(model_hub_name, self.pipeline_config['datasets'])

        scorers = self.get_scorers(model_hub, self.pipeline_config["scorers"])
        for scorer_name, scorer in scorers.items():
            if not scorer.trained:
                [callback.start(scorer_name=scorer_name) for callback in callbacks.values()]
                self.train_scorer(scorer, callbacks, 
                                self.datasets[scorer_name]["train"], 
                                self.datasets[scorer_name]["validation"], 
                                **self.pipeline_config["trainer"])
            scorer.eval()
        if "router" in self.pipeline_config.keys():
            router_evaluator_args = RouterEvaluatorArgs(**self.pipeline_config['router_evaluator'])
            self.router = self.get_router(scorers, model_hub, **self.pipeline_config["router"])
            self.router_evaluator = RouterEvaluator(evaluator_args=router_evaluator_args,
                                                        router=self.router)

    def evaluate(self, eval_dataset : BaseDataset = None):
        """Evaluate the router on the provided dataset.
        Args:
            eval_dataset (Dataset): The dataset to use for evaluation.
        Returns:
            float: The evaluation score.
        """
        if eval_dataset is None:
            eval_dataset = self.datasets["router"]["test"]
        
        if not hasattr(self, 'router_evaluator'):
            raise ValueError("Router evaluator has not been initialized. Please call fit() first.")
        
        return self.router_evaluator.evaluate(eval_dataset)