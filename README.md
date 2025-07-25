<p align="center">
<img src="images/llumux.svg" alt="Logo" width="500"/>
</p>


<h4 align="center"><strong> Compose, train and test fast LLM routers</strong></h4>

**Llumux** (from LLM Multiplexors) is a lightweight library for training and testing **routers** a.k.a. multiplexors that select the most appropriate Large Language Model (LLM) for each prompt. The routers can be built in a composable way by combining **scorers** or reward models that predict the answer complexity, performance, or any other user-defined criteria. It enables efficient use of multiple LLMs—balancing cost, speed, and accuracy.



## 🌟 Features

- 🧠 Train and test score or reward models to predict LLM attributes such as answer performance or complexity using custom data
- 🔀 Build routers by composing scorers, to select specific models given a prompt
- 🚀 Adaptable to different model hubs, scoring schemes, loss functions and datasets
- 🪄 Simple yet flexible usage, by merely specifying configurations in yaml files
- 📊 Logging and tracking of experiments

---

## 📦 Installation

```bash
git clone https://github.com/sebastianpinedaar/llumux.git
cd llumux
pip install -e .
```

---

## 🚀 Quick Start

To train and test routers, we can define a pipeline structure, using a yaml file. Check examples in `config/pipelines/example_pipeline.yml`.

```python
from llumux.pipeline import Pipeline

pipeline = Pipeline(config_path = "config/pipelines/example_llumux_dataset.yml")
pipeline.fit()
score = pipeline.evaluate()
print("Score:", score)
```

---

## 📘 Use Cases

- 🧠 Route factual questions to fast and cheap LLMs, and creative writing to more capable ones
- 💰 Optimize cost vs. fairness vs. performance when deploying multi-LLM architectures
- 🧪 Evaluate different routing strategies with built-in metrics
- 🧰 Build reward models

---

## 🛠️ Training a General Scorer

You can easily train a general scorer, by defining important parameters such as the dataset and the model hub.

```python
from llumux.datasets import ListwiseDataset
from llumux.scorers import GeneralScorer
from llums.hub import ModelHub
from llumux import Trainer, TrainerArgs

train_dataset = ListwiseDataset(dataset_name="llm-blender/mix-instruct", split="train",  list_size=3)

model_hub = ModelHub(args.model_hub_name)
model_list = model_hub.get_models()

scorer = GeneralScorer(model_list, prompt_embedder_name="albert-base-v2")

trainer_args = TrainerArgs(batch_size=4 epochs=1),
trainer = Trainer(scorer, trainer_args, train_dataset=train_dataset)
trainer.train()
```

You can adapt it to your own custome dataset, following a format as in the example located in `examples/example_llumux_dataset.yml`.

---

## 🧪 Create router composed of scorers

Sometimes it is desiderable to select a model based on several criterias, i.e. based on the ouput of many scorers, where every scorer predicts a specific criteria. We can combine this scorers by defining our own router. We provide several examples of router classes, but these are easily extendible.


```python
from llumux.datasets import RouterDataset
from llumux.routers import RatioRouter
from llumux import RouterEvaluater, RouterEvaluatorArgs

perf_scorer = ...
cost_scorer = ...
eval_dataset = RouterDataset(dataset_name = ..., 
                            model_hub_name= ...)
scorers = {
    "perf_scorer": perf_scorer,
    "cost_scorer": cost_scorer
}
router = RatioRouter(scorers = scorers)
evaluator_args = RouterEvaluatorArgs(batch_size = batch_size)
evaluator = RouterEvaluator(router=router, 
                                evaluator_args=evaluator_args, 
                                eval_dataset=eval_dataset)

eval_score = evaluator.evaluate()
```

---



## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT License.

---

## 🔗 Related Projects

- [OpenRouter](https://openrouter.ai/)
- [AutoGPT](https://github.com/Torantulino/Auto-GPT)