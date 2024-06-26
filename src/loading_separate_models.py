"""
Checking out how to load different models,
custom configure them and save them to a local folder.
"""

from transformers import BertConfig, TFBertModel, TFAutoModel
import tensorflow as tf


# create config and build the model from config
bert_config = BertConfig()
bert_model = TFBertModel(bert_config)

print(bert_config)

gpt_model = TFAutoModel.from_pretrained("gpt2")
print(type(gpt_model))
print(gpt_model.config)

bart_model = TFAutoModel.from_pretrained("facebook/bart-base")
print(type(bart_model))
print(bart_model.config)

# save models to disk
# path = "C:\\Users\\johan\\Git\\NLP_101\\src\\gpt_model"
# gpt_model.save_pretrained(path)

encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

model_inputs = tf.constant(encoded_sequences)
outputs = bart_model(model_inputs)

print(outputs)
