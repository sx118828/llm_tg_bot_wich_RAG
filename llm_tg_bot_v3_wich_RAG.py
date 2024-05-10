import logging
import requests
import telebot
import json
import os
import base64
from bs4 import BeautifulSoup

from langchain.chat_models.gigachat import GigaChat
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from chromadb.config import Settings
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.gigachat import GigaChatEmbeddings

from langchain.chains import RetrievalQA

from auth_data import token
from auth_data import credent
from auth_data import path_book

# Авторизация в сервисе GigaChat
llm = GigaChat(
    credentials=credent,
    verify_ssl_certs=False)

# Подготовка документов
# loader = TextLoader("/function/storage/tgbuccet/eltex_utf8.txt", encoding='utf-8')
loader = WebBaseLoader(
    path_book,
    encoding='utf-8')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(documents)

# Создание базы данных эмбеддингов
embeddings = GigaChatEmbeddings(
    credentials=credent,
    verify_ssl_certs=False)

db = Chroma.from_documents(
    documents,
    embeddings,
    client_settings=Settings(anonymized_telemetry=False),
)

# Cоздаем цепочку QnA, которая специально предназначена для ответов на вопросы по документам.
# В качестве аргументов передается языковая модель и ретривер (база данных)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

# Создаем функцию вывода ответа
def question_answer(question):
  return qa_chain({"query": question}).get('result', 0)

# Данные для аутентификации
logger = telebot.logger
telebot.logger.setLevel(logging.INFO)

API_TOKEN = token
# bot = telebot.TeleBot(os.environ.get('BOT_TOKEN'), threaded=False)
bot = telebot.TeleBot(API_TOKEN, threaded=False)

# Обработчики команд и сообщений

HELP = '''
Я телеграмм бот обученный на правилах приема в федеральное государственное бюджетное
образовательное учреждение высшего образования "Воронежский государственный университет"
Вы можете спросить меня по вопросам поступления  и я постараюсь Вам ответить.
ПРИМЕРЫ ВОПРОСОВ:
  * При поступлении каким документом подтверждается  уровень образование?
  * Как проводится прием на обучение по программам бакалавриата?
  * Как осуществляется оценка результатов вступительных испытаний?
  * Когда подается апелляция поступающим?
и т.п.
'''

@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.send_message(message.chat.id, HELP)

@bot.message_handler(func=lambda message: True, content_types=['text'])
def add_question(message):
  bot.send_message(message.chat.id, question_answer(message.text))

bot.polling(none_stop=True)



