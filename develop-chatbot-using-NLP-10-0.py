# Install spaCy:
# pip install spacy
# python -m spacy download en_core_web_sm
# Build a rule-based or machine learning-based chatbot that can answer user queries. Use NLP libraries like spaCy or NLTK to process and understand text.
# Machine Learning-Based Chatbot:

import spacy
from spacy.util import minibatch
from spacy.training import Example

# Training data format: (text, label)
TRAIN_DATA = [
    ("hello", {"cats": {"greeting": 1.0, "farewell": 0.0}}),
    ("hi", {"cats": {"greeting": 1.0, "farewell": 0.0}}),
    ("bye", {"cats": {"greeting": 0.0, "farewell": 1.0}}),
    ("goodbye", {"cats": {"greeting": 0.0, "farewell": 1.0}}),
]

# Load or create a blank spaCy model
nlp = spacy.blank("en")

# Add the text classifier to the pipeline
if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat")

# Add labels to the text classifier
textcat.add_label("greeting")
textcat.add_label("farewell")

# Training the model


def train_model(nlp, TRAIN_DATA, iterations=10):
    optimizer = nlp.begin_training()

    for i in range(iterations):
        losses = {}
        # Shuffle the training data
        examples = [Example.from_dict(nlp.make_doc(
            text), annotations) for text, annotations in TRAIN_DATA]
        batches = minibatch(examples, size=2)

        for batch in batches:
            nlp.update(batch, drop=0.5, sgd=optimizer, losses=losses)
        print(f"Iteration {i}: Losses - {losses}")


# Train the model
train_model(nlp, TRAIN_DATA)

# Test the trained model
doc = nlp("hello")
print(doc.cats)  # {'greeting': 0.9, 'farewell': 0.1}
