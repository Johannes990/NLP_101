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
    context="They said that I, Peter jokes a lot, must go on vacation."
)

print(context_answer)

# Summarization
summarizer = pipeline("summarization")
summary = summarizer(
    """America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.""",
    max_length=50
)

print(summary)


# translator
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translation = translator("Ce cours est produit par Hugging Face.")

print(translation)
