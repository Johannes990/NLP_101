"""Taking a closer look at the pipeline function from the transformers library."""

from transformers import AutoTokenizer, TFAutoModel, TFAutoModelForSequenceClassification
import tensorflow as tf

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModel.from_pretrained(checkpoint)
model_with_head = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

# what is the model bias??? -> from logits and probablilities it looks like the model
# assigns higher probabilities to negative sentences, even when the sentence is not
# actually negative. Bias towards negative predictions.
input_sentences = [
    "this is a bad sentence",
    "this is a good sentence"
]

# tensor representation of the two sentences
sentences = tokenizer(input_sentences, padding=True, truncation=True, return_tensors="tf")
print(sentences)

# from token tensor to model tensor
outputs = model(sentences)

print(outputs.last_hidden_state)
print(outputs['last_hidden_state'])
print(outputs[0])

# from sentences to logits
outputs = model_with_head(sentences)
print(outputs.logits.shape)  # here we have a 2, 2 tuple: 2 values for each sentence
print(outputs.logits)

# logits need to be converted to actual probabilities using 'softmax' activation
predictions = tf.math.softmax(outputs.logits, axis=-1)
print(predictions)

# what are the model labels?
print(model_with_head.config.id2label)
