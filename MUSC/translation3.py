from transformers import MarianMTModel, MarianTokenizer
import torch

model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text, tokenizer, model):
    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors="pt")
    print(f"Encoder input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

    # Encode the input text
    encoder_outputs = model.get_encoder()(input_ids)
    print(f"Encoder output shape: {encoder_outputs[0].shape}")

    # Prepare the decoder input using the encoded states and BOS token
    decoder_input_ids = torch.tensor([[tokenizer.eos_token_id]])  # Use EOS token as initial decoder input
    print(f"Decoder input tokens: {tokenizer.convert_ids_to_tokens(decoder_input_ids[0])}")

    # Decode the input using the encoder outputs and decoder input ids
    model_inputs = {
        "input_ids": None,
        "encoder_outputs": encoder_outputs,
        "past_key_values": None,
    }
    decoded_ids = model.generate(**model_inputs)

    # Decode the generated ids to text
    translated_text = tokenizer.decode(decoded_ids[0], skip_special_tokens=True)
    print(f"Decoder output tokens: {tokenizer.convert_ids_to_tokens(decoded_ids[0])}")

    return translated_text

# Sample input sentence
input_text = "Hello, how are you? I love your nation."

# Perform the translation
output_text = translate(input_text, tokenizer, model)
print(f"\nInput: {input_text}")
print(f"Translated: {output_text}")