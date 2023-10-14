import copy
import json
from transformers import MarianMTModel, MarianTokenizer
import torch
from torch import nn
import os
import warnings
import torchvision.datasets as dset
from PIL import Image
warnings.filterwarnings("ignore")
from base_nets import base_net
from channel_nets import channel_net
import time
import numpy as np
import torchvision
import random

torch.cuda.set_device(1)
class params():
    checkpoint_path = "checkpoints"
    device = "cuda"
    dataset = r"E:\pengyubo\datasets\UCF100"
    log_path = "logs"
    epoch = 5
    lr = 1e-3
    batchsize = 8
    snr = 25
    weight_delay = 1e-5
    sim_th = 0.7

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def test(tr_model, channel_net, tokenizer, training_texts, arg):
    tr_model = tr_model.to(arg.device)
    tr_model.eval()
    channel_model = channel_net.to(arg.device)
    weights_path = f"{arg.checkpoint_path}/t2t_ch_snr{arg.snr}.pth"
    weight = torch.load(weights_path, map_location="cpu")
    channel_model.load_state_dict(weight)
    channel_model.eval()
    raw_text = []
    rec_text = []
    random.shuffle(training_texts)
    for i in range(0, len(training_texts), arg.batchsize):
        if i + arg.batchsize < len(training_texts):
            b_text = training_texts[i:i + arg.batchsize]
        else:
            b_text = training_texts[i:]
        # Tokenize the input text
        input_ids = tokenizer.batch_encode_plus(b_text, return_tensors="pt", padding=True, max_length=512)[
            "input_ids"].to(arg.device)
        # input_ids = tokenizer.encode(b_text, return_tensors="pt").to(arg.device)
        # Encode the input text
        encoder_outputs = tr_model.get_encoder()(input_ids)
        model_inputs = {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": None,
        }
        decoded_ids = tr_model.generate(**model_inputs)
        # Decode the generated ids to text
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in decoded_ids]
        raw_text += translated_texts
        # print(translated_texts)

        shape = encoder_outputs[0].shape
        encoder_outputs_temp = encoder_outputs[0].view(-1, 512)
        encoder_outputs_with_noise = channel_model(encoder_outputs_temp)
        encoder_outputs[0].data = encoder_outputs_with_noise.view(shape).data
        model_inputs = {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": None,
        }
        decoded_ids = tr_model.generate(**model_inputs)
        # Decode the generated ids to text
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in decoded_ids]
        rec_text += translated_texts
        # print(translated_texts)

    with open(os.path.join(arg.log_path, f"t2t_snr{arg.snr}_eval_res.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"raw_text": raw_text, "rec_text": rec_text}, indent=4, ensure_ascii=False))
    evaluate(raw_text, rec_text,tokenizer, tr_model, arg)

def evaluate(src_txts, tar_txts, tokenizer, tr_model, arg):
    from sklearn.metrics.pairwise import cosine_similarity
    acc = 0
    for src_txt, tar_txt in zip(src_txts, tar_txts):
        # Tokenize the input
        # Calculate the similarity between the translated sentences using Cosine Similarity metric
        tokenized_inputs = tokenizer([src_txt, tar_txt], return_tensors="pt", padding=True).to(arg.device)
        with torch.no_grad():
            embeddings = tr_model.get_encoder()(**tokenized_inputs).last_hidden_state
            embeddings = embeddings.cpu().numpy()
        similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
        # Print the similarity score
        print(f"Cosine similarity score: {similarity}")
        if similarity > arg.sim_th:
            acc += 1
    print("accuracy:", acc / len(src_txts))

if __name__ == '__main__':
    same_seeds(1024)
    arg = params()
    training_texts = []
    for text in os.listdir(arg.dataset):
        if text.endswith(".json"):
            text_path = os.path.join(arg.dataset,text)
            with open(text_path,"r",encoding="utf-8")as f:
                content = json.load(f)
                content = [val.replace("<unk>","") for val in content]
            training_texts+=content
    print(len(training_texts))
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    tr_model = MarianMTModel.from_pretrained(model_name)
    # evaluate
    for snr in [0, 5,10,15,20,25]:
        arg.snr = snr
        channel_model = channel_net(M=512, snr=arg.snr)
        test(tr_model,channel_model,tokenizer,training_texts[:200], arg)

    # img:[0.08,0.04,0.055,0.075,0.08,0.07]
