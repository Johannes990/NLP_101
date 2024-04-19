"""Only learning here."""

from transformers import pipeline


# mask filling
mask_prompt = "This course will teach you all about [MASK] models."
unmasker = pipeline("fill-mask", model="bert-base-cased")
answers = unmasker(mask_prompt, top_k=3)

for ans in answers:
    print(ans['sequence'])

# named entity recognition
ner = pipeline("ner", grouped_entities=True)
entities = ner("Mary is a writer.")

print(entities)

# question answerer
q_a = pipeline("question-answering")
context_answer = q_a(
    question="What is my name",
    context="They said that I, Mister Deathly hallow, must go on vacation."
)

print(context_answer)
