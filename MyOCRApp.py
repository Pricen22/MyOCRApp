# Importing libraries
import re
from difflib import SequenceMatcher

import cv2
import nltk
import numpy as np
import pytesseract
import pyttsx3
import torch
from enchant.checker import SpellChecker
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty
from kivy.uix.screenmanager import Screen
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

# location of tesseract installed on PC
pytesseract.pytesseract.tesseract_cmd = r'D:/pytesseract/tesseract.exe'


class CameraScreen(Screen):
    def capture(self):

        camera = self.ids['camera']
        camera.export_to_png("./picforocr.png")
        image = cv2.imread("./picforocr.png")
        # processing to improve accuracy
        # Rescaling of image to recognise smaller characters
        image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        # Grey scale image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Thinning and Skeletonisation, handles font with different stroke width
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.erode(image, kernel, iterations=1)
        # MedianBlue to reduce image noise
        image = cv2.medianBlur(image, 3)
        # Gaussian threshold used to handle differing lighting conditions
        cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        text = pytesseract.image_to_string(image)

        # Error catch for if no text is detected in image
        text_original = str(text)
        if len(text_original) < 4:
            text_original = 'No text detected'
            text = 'No text detected'

        rep = {'\n': ' ', '\\': ' ', '\"': '"', '-': ' ', '"': ' " ',
               ',': ' , ', '.': ' . ', '!': ' ! ',
               '?': ' ? ', "n't": " not", "'ll": " will", '*': ' * ',
               '(': ' ( ', ')': ' ) ', "s'": "s '"}
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

        # Used to keep Peoples names within text to could've potentially been removed
        def get_personslist(text):
            personlist = []
            for sent in nltk.sent_tokenize(text):
                for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                    if isinstance(chunk, nltk.tree.Tree) and chunk.label() == 'PERSON':
                        personlist.insert(0, (chunk.leaves()[0][0]))
            return list(set(personlist))

        personslist = get_personslist(text)
        # For Ignoring words with Alphabets+Digits structure
        alpha_digit = [idx for idx in text.split() if
                       any(chr.isalpha() for chr in idx) and any(chr.isdigit() for chr in idx)]

        # All ignorable components
        ignorewords = personslist + ["!", ",", ".", "\"", "?", '(', ')', '*', ':', '', '/'] + alpha_digit
        # using enchant.checker.SpellChecker, identify incorrect words
        d = SpellChecker("en_US")
        words = text.split()
        incorrectwords = [w for w in words if not d.check(w) and w not in ignorewords]
        # using enchant.checker.SpellChecker, get suggested replacements
        suggestedwords = [d.suggest(w) for w in incorrectwords]
        # replace incorrect words with [MASK]
        for w in incorrectwords:
            text = text.replace(w, '[MASK]')
            text_original = text_original.replace(w, '[MASK]')

        # Load, train and predict using pre-trained model
        tokenizer = BertTokenizer.from_pretrained("D:/BERT")
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        MASKIDS = [i for i, e in enumerate(tokenized_text) if e == '[MASK]']  # Indexes of [Mask]
        # prepare Torch inputs
        tokens_tensor = torch.tensor([indexed_tokens])
        # Load pre-trained model
        model = BertForMaskedLM.from_pretrained("D:/BERT")
        # Predict all tokens
        with torch.no_grad():
            predictions = model(tokens_tensor)

        # Predict words for mask using BERT;
        def predict_word(text_original, predictions, MASKIDS):
            pred_words = []
            for i in range(len(MASKIDS)):
                preds = torch.topk(predictions[0, MASKIDS[i]], k=50)
                indices = preds.indices.tolist()
                list1 = tokenizer.convert_ids_to_tokens(indices)
                list2 = suggestedwords[i]
                # simmax of 0 means always use BERT suggestion, can be between 0-1.
                simmax = 0
                predicted_token = ''
                for word1 in list1:
                    for word2 in list2:
                        s = SequenceMatcher(None, word1, word2).ratio()
                        if s is not None and s > simmax:
                            simmax = s
                            predicted_token = word1
                            text_original = text_original.replace('[MASK]', predicted_token, 1)
            return text_original

        text = predict_word(text_original, predictions, MASKIDS)
        self.manager.get_screen('text_screen').ocrtext = text


class TextScreen(Screen):
    ocrtext = StringProperty('')
    my_font = NumericProperty(20)

    def speak(self):

        voice_text = ""
        for i in self.manager.get_screen('text_screen').ocrtext.split():
            voice_text += i + ' '

        voice_text = voice_text[:-1]
        engine = pyttsx3.init()
        # speed of speech
        engine.setProperty("rate", 145)
        voices = engine.getProperty('voices')
        # type of voice, change int for male/female
        engine.setProperty('voice', voices[0].id)
        engine.say(voice_text)
        engine.runAndWait()

    # if the size is under 40, increase font by 4, if > 40 then reset to 20/default size
    def increase_font_size(self):
        if self.manager.get_screen('text_screen').my_font <= 40.0:
            self.manager.get_screen('text_screen').my_font = self.manager.get_screen('text_screen').my_font + 4
        else:
            self.manager.get_screen('text_screen').my_font = 20

    def reset_font_size(self):
        self.manager.get_screen('text_screen').my_font = 20

# Kivy code to construct GUI


GUI = Builder.load_string("""

GridLayout:
    cols: 1
    ScreenManager:
        id: screen_manager
        CameraScreen:
            name: "camera_screen"
            id: camera_screen
        TextScreen:
            name: "text_screen"
            id: text_screen



<CameraScreen>:
    orientation: 'vertical'
    GridLayout:
        cols: 1
        Camera:
            id: camera
            resolution: (800, 800)
        Button:
            text: 'Take Picture!'
            size_hint_y: None
            height: '48dp'
            on_press:
                root.capture()
                # root refers to <CameraScreen>
                # app refers to TestCamera, app.root refers to the GridLayout: at the top
                app.root.ids['screen_manager'].transition.direction = 'left'
                app.root.ids['screen_manager'].current = 'text_screen'

<TextScreen>:
    orientation: 'vertical'
    ScrollView:
        Label:
            id: ocr_output
            text: root.ocrtext
            halign: 'center'
            valign: 'center'
            text_size: 450, None
            size_hint_y: None
            height: self.parent.size[1] if self.texture_size[1] < self.parent.size[1] else self.texture_size[1]
            font_size: root.my_font
            background_color: (255/255, 255/255, 255/255)
            canvas.before:
                Color:
                    rgba: self.background_color
                Rectangle:
                    size: self.size
                    pos: self.pos
            # Text Properties
            color: (1/255, 1/255, 1/255)
    Button:
        id: back_to_cam
        text: "Take Another Picture"
        size_hint_y: None
        height: '48dp'
        font_size: 50
        on_press:
            app.root.ids['screen_manager'].transition.direction = 'right'
            app.root.ids['screen_manager'].current = 'camera_screen'
            root.reset_font_size()
    Button:
        id: text_to_speech
        size_hint: 0.125, 0.150
        pos_hint: {"x":0.80, "y":0.80}
        background_normal: "./speaker.png"
        background_down: "./speaker.png"
        on_press: root.speak()
    Button:
        id: increase_font_size
        size_hint: 0.125, 0.125
        pos_hint: {"x":0.10, "y":0.80}
        background_normal: "./fontincrease.png"
        background_down: "./fontincrease.png"
        on_press: root.increase_font_size()



""")


class MyOCRApp(App):
    def build(self):
        return GUI


if __name__ == "__main__":
    MyOCRApp().run()
