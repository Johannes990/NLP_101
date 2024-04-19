"""Only learning here."""

from transformers import pipeline

# text generation
textgen_prompt = "Could you please explain to me what is the meaning of"

classifier = pipeline("text-generation", model="distilgpt2")
answer = classifier(textgen_prompt,
                    num_return_sequences=5,
                    truncation=True,
                    max_length=50)

for dicty in answer:
    for label, gen in dicty.items():
        print(f"{label}: {gen}")

# mask filling
mask_prompt = "Who the fuck killed my <mask>."
unmasker = pipeline("fill-mask")

unmasked = unmasker(mask_prompt, top_k=2)
for el in unmasked:
    print(el['sequence'])
