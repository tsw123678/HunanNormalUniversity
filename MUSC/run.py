import json
import os
import warnings

import torch
from torch import nn
from transformers import MarianMTModel, MarianTokenizer

warnings.filterwarnings("ignore")
from channel_nets import channel_net
import numpy as np
import random

torch.cuda.set_device(0)


class params():
    checkpoint_path = "checkpoints"
    device = "cuda"
    dataset = r"E:\pengyubo\datasets\trainingtext"
    log_path = "logs"
    epoch = 5
    lr = 1e-3
    batchsize = 48
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


def train(tr_model, channel_net, tokenizer, training_texts, arg):
    tr_model = tr_model.to(arg.device)
    channel_model = channel_net.to(arg.device)

    # define optimizer
    optimizer = torch.optim.Adam(channel_model.parameters(), lr=arg.lr,
                                 weight_decay=arg.weight_delay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.1, patience=300,
                                                           verbose=True, threshold=0.0001,
                                                           threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)

    weights_path = f"{arg.checkpoint_path}/t2t_ch_snr{arg.snr}.pth"

    mse = nn.MSELoss()

    raw_text = []
    rec_text = []
    for epoch in range(arg.epoch):
        random.shuffle(training_texts)
        # range(0, len) step=48 : 0 47 94 ........
        for i in range(0, len(training_texts), arg.batchsize):
            # 当前批次结束索引+batchsize是否结束，
            # 小于直接把剩下的加入b_text，否则按batchsize取
            if i + arg.batchsize < len(training_texts):
                b_text = training_texts[i:i + arg.batchsize]
            else:
                b_text = training_texts[i:]
            raw_text += b_text
            print(f"input text:{b_text}")

            # Tokenize the input text
            input_ids = tokenizer.batch_encode_plus(b_text, return_tensors="pt", padding=True, max_length=512)[
                "input_ids"].to(arg.device)
            # input_ids = tokenizer.encode(b_text, return_tensors="pt").to(arg.device)

            # Encode the input text
            with torch.no_grad():
                encoder_outputs = tr_model.get_encoder()(input_ids)

            shape = encoder_outputs[0].shape
            encoder_outputs_temp = encoder_outputs[0].view(-1, 512)
            encoder_outputs_with_noise = channel_model(encoder_outputs_temp)

            loss = mse(encoder_outputs_temp, encoder_outputs_with_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            encoder_outputs[0].data = encoder_outputs_with_noise.view(shape).data

            # Decode the input using the encoder outputs and decoder input ids
            model_inputs = {
                "input_ids": None,
                "encoder_outputs": encoder_outputs,
                "past_key_values": None,
            }
            with torch.no_grad():
                decoded_ids = tr_model.generate(**model_inputs)

            # Decode the generated ids to text
            translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in decoded_ids]
            rec_text += translated_texts
            print(f"output text:{translated_texts}")
            print(f"epoch:{epoch}, loss:{loss.item()}")

            with open(os.path.join(arg.log_path, f"t2t_snr{arg.snr}_res.json"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"raw_text": raw_text, "rec_text": rec_text}, indent=4, ensure_ascii=False))
        torch.save(channel_model.state_dict(), weights_path)


@torch.no_grad()
def test(tr_model, channel_net, tokenizer, training_texts, arg):
    tr_model = tr_model.to(arg.device)
    channel_model = channel_net.to(arg.device)
    weights_path = f"{arg.checkpoint_path}/t2t_ch_snr{arg.snr}.pth"
    weight = torch.load(weights_path, map_location="cpu")
    channel_model.load_state_dict(weight)
    raw_text = []
    rec_text = []
    random.shuffle(training_texts)
    for i in range(0, len(training_texts), arg.batchsize):
        if i + arg.batchsize < len(training_texts):
            b_text = training_texts[i:i + arg.batchsize]
        else:
            b_text = training_texts[i:]
        raw_text += b_text
        print(f"input text:{b_text}")
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

    evaluate(raw_text, rec_text, tr_model)


@torch.no_grad()
def evaluate(src_txts, tar_txts, tr_model):
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
            text_path = os.path.join(arg.dataset, text)
            with open(text_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                content = [val.replace("<unk>", "") for val in content]
            training_texts += content
    print(len(training_texts))

    # 加载预训练的模型和分词器
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    tr_model = MarianMTModel.from_pretrained(model_name)

    # evaluate
    for snr in [5, 10, 15, 20, 25]:
        arg.snr = snr
        channel_model = channel_net(M=512, snr=arg.snr)
        train(tr_model, channel_model, tokenizer, training_texts, arg)
