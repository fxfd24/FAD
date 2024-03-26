from threading import Thread
import threading

from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.scrollview import ScrollView
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.button import MDFloatingActionButton
from kivymd.uix.label import MDLabel
from kivy.core.audio import SoundLoader
from kivy.metrics import dp
from kivy.clock import Clock
from kivy.uix.gridlayout import GridLayout

import speech_recognition as sr
import pyttsx3

# для типа нейросети
import pandas as pd
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
import joblib
import pickle

KV = '''
BoxLayout:
    orientation: 'vertical'

    ScrollView:
        size_hint_y: None
        height: root.height - dp(80)  # Высота ScrollView занимает все место до RelativeLayout

        GridLayout:
            id: user_messages_container
            cols: 1
            size_hint_y: None
            height: self.minimum_height
            padding: dp(10)

    RelativeLayout:
        size_hint_y: None
        height: dp(80)
        pos_hint: {'center_x': 0.5}

        canvas:
            Color:
                rgba: 0, 0, 255, 0.1  # Белый цвет
            Rectangle:
                pos: self.pos
                size: self.width - dp(0), self.height

        BoxLayout:
            spacing: '10dp'
            padding: '10dp'
            size_hint_y: None
            height: dp(48)
            pos_hint: {'center_x': 0.5}

            MDTextField:
                id: text_field
                hint_text: "Введите сообщение"
                size_hint_x: 0.8
                on_text: app.on_text_changed(self.text)

            MDFloatingActionButton:
                id: action_button
                icon: 'microphone'
                elevation_normal: 8
                on_press: app.action_button_pressed()
                on_release: app.action_button_released()

'''

class FirstHelpApp(MDApp):
    recording = False  # Флаг для отслеживания записи звука
    r = sr.Recognizer()
    svm_loaded = joblib.load('data/svm_model.pkl')
    vectorizer = pickle.load(open('data/vectorizer.pickle', 'rb'))
    ill_help = pd.read_csv('data/FA.csv')
    help = ill_help['help'].tolist()
    target = ill_help['target'].tolist()



    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub("[^а-яА-яЁё]", " ", text)
        stop_words = set(stopwords.words("russian"))
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        text = " ".join(filtered_words)
        return text

    def build(self):
        self.theme_cls.primary_palette = "Purple"
        return Builder.load_string(KV)

    def show_file_manager(self):
        self.file_manager = MDFileManager(exit_manager=self.exit_manager)
        self.file_manager.show('/')

    def exit_manager(self, *args):
        self.file_manager.close()

    def on_text_changed(self, text):
        action_button = self.root.ids.action_button
        if text.strip():  # Если текст не пустой
            action_button.icon = 'send'  # Меняем иконку на 'send'
        else:
            action_button.icon = 'microphone'
            action_button.md_bg_color = [1, 1, 1, 1]
            action_button.md_bg_color = [0, 0, 255, 1]
              # Возвращаем исходную иконку

    def action_button_pressed(self):
        text_input = self.root.ids.text_field
        message_text = text_input.text.strip()
        if message_text:
            self.send_text()
        else:
            self.start_recording()

    def send_text(self):
        text_input = self.root.ids.text_field
        message_text = text_input.text
        text_input.text = ""  # Очищаем поле ввода текста
        if message_text:
            user_messages_container = self.root.ids.user_messages_container
            user_messages_container.cols = 1  # Установка одного столбца в GridLayout

            # Создаем новый виджет для сообщения и добавляем его в GridLayout
            message_label = MDLabel(text=message_text, halign='left', size_hint_y=None)
            message_label.bind(texture_size=message_label.setter('size'))
            message_label.padding = dp(10)
            message_label.margin = dp(10)
            message_label.md_bg_color = [0, 0, 255, 0.2]
            user_messages_container.add_widget(message_label)

            # Логика ответа типа умного приложения
            input_text = str(message_text)
            preprocessed_text = self.preprocess_text(input_text)
            input_text_vec = self.vectorizer.transform([preprocessed_text])
            predicted_class = self.svm_loaded.predict(input_text_vec)
            print("Predicted class:", predicted_class)
            print(self.target,'\n')
            print(self.target[predicted_class[0]-1])
            print(self.help[predicted_class[0]-1])
            output_model = self.target[predicted_class[0]-1] + '\n' + self.help[predicted_class[0]-1]

            # Создаем новый виджет для ответа и добавляем его в GridLayout
            message_label = MDLabel(text=output_model, halign='left', size_hint_y=None)
            message_label.bind(texture_size=message_label.setter('size'))
            message_label.padding = dp(10)
            message_label.md_bg_color = [0, 0, 255, 0.1]
            user_messages_container.add_widget(message_label)

            # Обновляем высоту GridLayout, чтобы учесть добавленные виджеты
            user_messages_container.height = user_messages_container.minimum_height

            # Устанавливаем вертикальную прокрутку в ноль
            user_messages_container.scroll_y = 0
        try:
            sound = SoundLoader.load('music/send_audio.mp3')
            if sound:
                sound.play()
        except:
            pass

    # логика записи гс
    def send_text_from_audio(self, text):
        def update_text(dt):
            text_input = self.root.ids.text_field  # Получаем доступ к виджету поля ввода текста
            text_input.text = text  # Устанавливаем распознанный текст в поле ввода
        
        Clock.schedule_once(update_text) 

    def recognize_and_print_text(self,audio_source):
        # r = sr.Recognizer()
        try:
            # Преобразование аудио в текст
            text = self.r.recognize_google(audio_source, language='ru-RU')
            print("Вы сказали:", text)
            self.send_text_from_audio(text)
        except sr.UnknownValueError:
            print("Google Speech Recognition не смог распознать аудио.")
        except sr.RequestError as e:
            print("Ошибка запроса к сервису Google Speech Recognition: {0}".format(e))
    
    def recording_in_process(self,recording):
        if recording:
            # Использование микрофона для записи звука
            print('recording in process')
            with sr.Microphone() as source:
                print("Говорите что-нибудь...")
                # Установка уровня шума окружающей среды
                self.r.adjust_for_ambient_noise(source)
                # Начало записи аудио
                audio_data = self.r.listen(source)
                # Создание отдельного потока для распознавания речи, чтобы приложение не блокировалось
                recognition_thread = threading.Thread(target=self.recognize_and_print_text, args=(audio_data,))
                recognition_thread.start()
    def action_button_released(self):
        if self.recording:
            self.stop_recording()

    def start_recording(self):
        self.recording = True
        print('Начать запись звука')
        micro_icon = self.root.ids.action_button
        micro_icon.md_bg_color = [0, 1, 0, 1] 

    def stop_recording(self):
        self.recording = False
        print('Прекратить запись звука')
        micro_icon = self.root.ids.action_button
        micro_icon.icon = 'microphone'  # Возвращаем исходную иконку микрофона
        micro_icon.md_bg_color = [1, 1, 1, 1]
        micro_icon.md_bg_color = [0, 0, 255, 1]


if __name__ == '__main__':
    FirstHelpApp().run()
