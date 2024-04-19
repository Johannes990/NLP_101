"""Only learning here."""

from transformers import pipeline

prompt = "This is a new prompt about nothing at all. void of content."


classifier = pipeline("zero-shot-classification", model="FacebookAI/roberta-large-mnli")
answer = classifier(prompt,
                    candidate_labels=['barbie', 'school', 'emptiness', 'lucidity', 'nirvana'])

for i in range(len(answer['labels'])):
    print(f"label {answer['labels'][i]}: probability {100 * answer['scores'][i]:.3f}%")
