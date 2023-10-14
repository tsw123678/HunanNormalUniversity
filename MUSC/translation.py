from googletrans import Translator

# initialize translator
translator = Translator()

# English to Chinese translation
en_text = "fuck your mother"
zh_text = translator.translate(en_text, dest='zh-CN').text
print(zh_text)

# Chinese to English translation
zh_text = "操你妈"
en_text = translator.translate(zh_text, dest='en').text
print(en_text)
