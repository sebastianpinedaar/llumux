# 🚀 Fast LLM Routers

**Fast LLM Routers** is a lightweight library for training and testing **routers** that select the most appropriate Large Language Model (LLM) for each propmpt. The routers can be built in a composable way by combining **scorers** that predict the answer complexity, performance, or any other user-defined criteria. It enables efficient use of multiple LLMs—balancing cost, speed, and accuracy.

<img src="./images/logo2.png" alt="Logo" width="200"/>


## 🌟 Features

- 🔀 Train routing models using custom data or auto-labeled decisions
- 🧠 Built-in support for popular LLMs (OpenAI, Anthropic, HuggingFace, etc.)
- 🪄 Simple API for integrating routing into your existing pipelines
- 📊 Logging, evaluation, and routing performance metrics
- 💾 Easy deployment via REST API or CLI

---

## 📦 Installation

```bash
pip install fast-llm-router
```

---

## 🚀 Quick Start

```python
from fast_llm_router import Router, LLMClient

# Define the available LLMs
llms = {
    "gpt-3.5": LLMClient(name="gpt-3.5-turbo", cost=0.001, speed="fast"),
    "gpt-4":   LLMClient(name="gpt-4", cost=0.03, speed="slow"),
}

# Load or train a routing model
router = Router.load("my_router.pkl")  # or Router.train(data, labels)

# Use the router to select the best LLM for a given prompt
prompt = "Write a one-paragraph summary of the Theory of Relativity."
chosen_llm = router.route(prompt, llms)
response = chosen_llm.generate(prompt)

print(response)
```

---

## 📘 Use Cases

- 🧠 Route factual questions to fast and cheap LLMs, and creative writing to more capable ones
- 💰 Optimize cost vs. performance when deploying multi-LLM architectures
- 🧪 Evaluate different routing strategies with built-in metrics

---

## 🛠️ Training a Custom Router

```python
from fast_llm_router import Router
from your_dataset import load_dataset

X, y = load_dataset("routing_data.csv")  # e.g., prompt → best LLM
router = Router.train(X, y)
router.save("my_router.pkl")
```

You can also plug in transformer-based models (e.g., BERT, DistilBERT) as the routing model.

---

## 🧪 Evaluation

```python
router.evaluate(X_test, y_test)
```

---

## 🧰 CLI Usage

```bash
fast-llm-router route --prompt "What's the capital of France?"
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