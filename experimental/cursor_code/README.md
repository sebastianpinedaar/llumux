# Fast LLM Routing API

A sophisticated API service that provides intelligent routing recommendations for Large Language Models (LLMs) based on user prompts, performance metrics, and cost considerations.

## Features

- **Smart LLM Selection**: Analyzes input prompts and recommends the most suitable LLM(s)
- **Cost Estimation**: Provides estimated costs for running prompts on different LLMs
- **Performance Metrics**: Includes generation time and quality metrics for each LLM
- **Human Preference Integration**: Incorporates human feedback and preferences
- **Hyperparameter Optimization**: (Future Feature) Suggests optimal inference parameters like temperature

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from fast_llm_router import LLMRouter

# Initialize the router
router = LLMRouter()

# Get LLM recommendations
recommendations = router.get_recommendations(
    prompt="Your prompt here",
    max_budget=10.0,  # Optional: maximum budget in USD
    max_time=60.0     # Optional: maximum generation time in seconds
)

# Print recommendations
for rec in recommendations:
    print(f"Model: {rec.model_name}")
    print(f"Estimated Cost: ${rec.estimated_cost}")
    print(f"Estimated Time: {rec.estimated_time}s")
    print(f"Performance Score: {rec.performance_score}")
```

## Project Structure

```
fast-llm-routing/
├── src/
│   ├── models/          # Model definitions and database schemas
│   ├── router/          # Core routing logic
│   ├── metrics/         # Performance and cost metrics
│   └── api/            # FastAPI endpoints
├── tests/              # Test cases
├── examples/           # Example usage scripts
├── data/              # Performance data and human preferences
└── docs/              # Documentation
```

## Configuration

Create a `.env` file in the root directory with your API keys:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
COHERE_API_KEY=your_cohere_key
```

## Development

To run tests:
```bash
pytest tests/
```

To start the API server:
```bash
uvicorn src.api.main:app --reload
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License 