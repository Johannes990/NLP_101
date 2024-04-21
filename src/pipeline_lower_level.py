"""Taking a closer look at the pipeline function from the transformers library."""

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

input_sentences = [
    "I think this ought to bee good enough.",
    "This is positively great!!!"
]

sentences = tokenizer(input_sentences, padding=True, truncation=True, return_tensors="tf")
print(sentences)
