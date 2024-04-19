"""Only learning here."""

from transformers import pipeline

prompt = "In this course, we will teach you how to"
candidate_labels = ['barbie', 'school', 'emptiness', 'lucidity', 'nirvana']

classifier = pipeline("text-generation", model="distilgpt2")
answer = classifier(prompt,
                    num_return_sequences=2,
                    truncation=True,
                    max_length=15)

# for i in range(len(answer['labels'])):
#     print(f"label {answer['labels'][i]}: probability {100 * answer['scores'][i]:.3f}%")
print(answer)

