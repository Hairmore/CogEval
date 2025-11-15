## CogProbe: Cognitive Operations Evaluation Benchmark

### Overview
This repository contains the implementation of CogProbe, a diagnostic benchmark for evaluating latent cognitive capabilities of Large Language Models based on ACT-R cognitive architecture.

### Repository Structure
```
├── README.md
├── evaluation_scripts/      # Individual evaluation scripts for each cognitive operation       
├── sample_data/            # Representative data samples
├── configs/                # Model and experiment configurations  
└── requirements.txt        # Dependencies
```

### Quick Start

#### Installation
```bash
pip install -r requirements.txt
```

#### Usage Notes
```bash
Modular Design: Each cognitive operation has independent evaluation scripts
Flexible Configuration: Modify config files for different models/settings
Sample Data Provided: Representative examples for each of 16 cognitive operations
Analysis Tools: Scripts for reproducing paper results and visualizations
```

### Cognitive Operations Framework

Our benchmark evaluates 16 cognitive operations across 5 macro-capabilities:

1. **Goal Setting**
   - Task Identification
   - Goal Decomposition

2. **Declarative Knowledge Retrieval**
   - Factual Recall
   - Episodic Recall

3. **Procedural Rule Execution**
   - Classification
   - Rule Execution
   - Logical Deduction
   - Causal Reasoning
   - Mathematical Computation

4. **Association and Analogy**
   - Association
   - Analogy
   - Inductive Reasoning

5. **Metacognitive Monitoring**
   - Self-checking
   - Rationality Evaluation
   - Error Correction
   - Working Memory Update

### Data Format

Each cognitive operation follows a standardized format:
```json
{
    "cognitive_operation": "operation_name",
    "question":"",
    "gold_answer":"",
    "system_prompt": "task_prompt",
    "confidence_prompt": "for meta-cognitive",
    "rationality_prompt": "for meta-cognitive",
    "exact_answer": "1 or 0",
    "metrics": "evaluation_method",
    "language": "English/Spanish/Chinese",
    "id": "unique_identifier",
}
```
For Metacognitive operations:
```json
{
    "context": "orginal question", 
    "gold_answer": "", 
    "model_answer": "model's first answer", 
    "rationality_label": "Not reasonable", 
    "rationality_judgement": "Model‘s judgement", 
    "confidence_score": 75, 
    "model_reflection": ""
    "new_response":"model's second answer"

}
```


### Evaluation Metrics

Different cognitive operations use specialized metrics:
- **Accuracy**: Factual Recall, Episodic Memory, etc.
- **Graph Algorithms**: Associative Thinking (efficiency, clustering, modularity)
- **Calibration**: Metacognitive operations
- **Model Scoring**: Complex reasoning tasks requiring qualitative assessment

### Sample Data

The `data/sample_data/` directory contains representative examples for each cognitive operation. **Complete dataset will be released upon paper publication.**

### Supported Models

Currently supports:
- OpenAI GPT models (via API)
- Qwen series models
- LLaMA models
- Custom model integration (see `src/models/base_model.py`)

### Reproducibility

#### Random Seeds
All experiments use fixed random seeds for reproducibility.

### Results

Key findings from our evaluation:
- Models excel at memory-based tasks but struggle with higher-order reasoning
- Metacognitive capabilities show significant deficits across all models
- "Think mode" improves accuracy but introduces instability
- Cross-linguistic performance primarily reflects training data exposure



---

**Note**: This repository currently contains sample data and complete evaluation framework. Full dataset will be made available following paper acceptance to balance reproducibility with ongoing research protection.
