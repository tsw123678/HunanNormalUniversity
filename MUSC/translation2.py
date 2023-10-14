from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


def translate(text, tokenizer, model):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    translated_ids = model.generate(input_ids)
    print(translated_ids.shape)

    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

    return translated_text


# Sample input sentence
input_text = "today,dog,faec"

# Perform the translation
output_text = translate(input_text, tokenizer, model)
print(f"Input: {input_text}")
print(f"Translated: {output_text}")
