import json
import os
from core.models.model_module_infer import model_module
from matplotlib import pyplot as plt
from PIL import Image
import torchaudio
import cv2


def text2img(prompt):
    # prompt = "A beautiful oil painting of a birch tree standing in a spring meadow with pink flowers, a distant mountain towers over the field in the distance. Artwork by Alena Aenami"
    images = inference_tester.inference(xtype=['image'],
                                        condition=[prompt],
                                        condition_types=['text'],
                                        n_samples=1,
                                        image_size=256,
                                        ddim_steps=50)
    plt.imshow(images[0][0])
    plt.axis = False
    plt.savefig("a.jpg")
    plt.show()


def img2text(dataset):
    for img in os.listdir(dataset):
        if not img.endswith(".jpg"):
            continue
        img_path = os.path.join(dataset, img)
        dst_txt_path = img_path.replace(".jpg", ".json")
        if os.path.exists(dst_txt_path):
            continue
        im = Image.open(img_path).resize((224, 224))
        text = inference_tester.inference(
            xtype=['text'],
            condition=[im],
            condition_types=['image'],
            n_samples=5,
            ddim_steps=10,
            scale=7.5, )
        text = text[0]
        print(text)
        with open(dst_txt_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(text, indent=4, ensure_ascii=False))


def audio2text(dataset):
    for audio in os.listdir(dataset):
        if not audio.endswith(".flac"):
            continue
        audio_path = os.path.join(dataset, audio)
        dst_txt_path = audio_path.replace(".flac", ".json")
        if os.path.exists(dst_txt_path):
            continue

        audio_wavs, sr = torchaudio.load(audio_path)
        audio_wavs = torchaudio.functional.resample(waveform=audio_wavs, orig_freq=sr, new_freq=16000).mean(0)[
                     :int(16000 * 10.23)]
        # Audio(audio_wavs.squeeze(), rate=16000)
        text = inference_tester.inference(
            xtype=['text'],
            condition=[audio_wavs],
            condition_types=['audio'],
            n_samples=2,
            ddim_steps=10,
            scale=7.5)
        text = text[0]
        print(text)
        with open(dst_txt_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(text, indent=4, ensure_ascii=False))


def video2text(dataset):
    for video in os.listdir(dataset):
        if not video.endswith(".avi"):
            continue
        video_path = os.path.join(dataset, video)
        dst_txt_path = video_path.replace(".avi", ".json")
        if os.path.exists(dst_txt_path):
            continue
        video_text = []
        cv = cv2.VideoCapture(video_path)  # 读入视频文件，命名cv
        if cv.isOpened():  # 判断是否正常打开
            rval, frame = cv.read()
            i = 0
        else:
            rval = False
            print('open video error!!')
        while rval:  # 正常打开 开始处理
            rval, frame = cv.read()
            if (i % 10 == 0):  # 每隔timeF帧进行存储操作
                try:
                    im = Image.fromarray(frame)
                except:
                    continue
            i += 1
            if i > 3:
                break
            # Audio(audio_wavs.squeeze(), rate=16000)
            text = inference_tester.inference(
                xtype=['text'],
                condition=[im],
                condition_types=['image'],
                n_samples=1,
                ddim_steps=10,
                scale=7.5)
            video_text += text[0]
        cv2.waitKey(1)
        cv.release()
        with open(dst_txt_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(video_text, indent=4, ensure_ascii=False))


def text2video():
    # Give A Prompt
    prompt = "The boy has a birthday cake decorated with a large candle while celebrating."

    n_samples = 1
    outputs = inference_tester.inference(
        ['video'],
        condition=[prompt],
        condition_types=['text'],
        n_samples=1,
        image_size=256,
        ddim_steps=50,
        num_frames=8,
        scale=7.5)

    video = outputs[0][0]
    # Visual video as gif
    from PIL import Image
    frame_one = video[0]
    path = "./generated_text2video.gif"
    frame_one.save(path, format="GIF", append_images=video[1:],
                   save_all=True, duration=2000 / len(video), loop=0)

    # from IPython import display
    # from IPython.display import Image
    # Image(data=open(path, 'rb').read(), format='png')


if __name__ == '__main__':
    model_load_paths = ['CoDi_encoders.pth', 'CoDi_text_diffuser.pth', 'CoDi_audio_diffuser_m.pth',
                        'CoDi_video_diffuser_8frames.pth']
    inference_tester = model_module(data_dir='../', pth=model_load_paths)
    inference_tester = inference_tester
    inference_tester = inference_tester.eval()
    # img_dataset = r"E:\pengyubo\datasets\VOC2012_img2text\VOC2012"
    # img2text(img_dataset)
    # audio_dataset = r"E:\pengyubo\datasets\LibriSpeech_audio2text"
    # audio2text(audio_dataset)
    # video_dataset = r"E:\pengyubo\datasets\UCF100"
    # video2text(video_dataset)
    text2video()
