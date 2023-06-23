# QABert

This repository contains code for a Question Answering model using the QABert architecture. The model is trained on the SQuAD dataset and is capable of answering questions based on a given context.

## Installation

To run the code in this repository, please make sure you have the following dependencies installed:

- `transformers`
- `datasets`
- `evaluate`
- `torch`
- `accelerate`

You can install these dependencies by running the following command:

```shell
pip install transformers datasets evaluate transformers[torch] accelerate
```

## Usage

1. Load the SQuAD dataset:

```python
from datasets import load_dataset

squad = load_dataset("squad", split="train[:5000]")
```

2. Split the dataset into training and testing sets:

```python
squad = squad.train_test_split(test_size=0.3)
```

3. Tokenize and preprocess the dataset:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    # Preprocess the examples
    # ...
    return inputs

tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
```

4. Create a data collator:

```python
from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()
```

5. Define the training arguments:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="my_awesome_qa_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)
```

6. Initialize and train the model:

```python
from transformers import AutoModelForQuestionAnswering, Trainer

model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

7. Use the trained model for question answering:

```python
from transformers import pipeline

question = "Those who attended a Jesuit college may have been forbidden from joining which Law School due to the curricula at the Jesuit institution?"
context = "In 1919 Father James Burns became president of Notre Dame, and in three years he produced an academic revolution that brought the school up to national standards by adopting the elective system and moving away from the university's traditional scholastic and classical emphasis. By contrast, the Jesuit colleges, bastions of academic conservatism, were reluctant to move to a system of electives. Their graduates were shut out of Harvard Law School for that reason. Notre Dame continued to grow over the years, adding more colleges, programs, and sports teams. By 1921, with the addition of the College of Commerce, Notre Dame had grown from a small college to a university with five colleges and a professional law school. The university continued to expand and add new residence halls and buildings with each subsequent president."

question_answerer = pipeline("question-answering", model="/content/my_awesome_qa_model")
question_answerer(question=question, context=context)
```

## Additional Information

- The `squad` dataset is loaded and split into training and testing sets using the `train_test_split` function with a test size of 30%.

- The dataset is preprocessed and tokenized using the `

AutoTokenizer` from the "distilbert-base-uncased" model. The `preprocess_function` is used to preprocess each example in the dataset by tokenizing the questions and contexts, setting maximum length, handling truncation, and adding start and end positions for the answers.

- The model is trained using the `Trainer` class from the `transformers` library. The training arguments specify the output directory, evaluation strategy, learning rate, batch sizes, number of training epochs, and weight decay.

- After training, the trained model can be used for question answering by creating a pipeline with the `"question-answering"` task and providing the trained model's path. The pipeline can be used to answer questions based on a given context.


Please refer to the notebook for more details and information about the original code.
