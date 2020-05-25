import os
import glob
import zipfile
import datetime
import pandas as pd
import numpy as np
import csv
import sys

import keras
keras.__version__
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras.models import load_model

def vectorize_sequences(sequences, dimension=100000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def to_one_hot(labels, dimension=128):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, int(label)] = 1.
    return results

def load_json_line(file_path):
    #pandasだと一部のjsonlファイルでエラーが出るためnumpyに変換する
    #np.arrayでは一部のファイルでエラーが出るのでnp.asarrayを使う
    return np.asarray(pd.read_json(file_path, orient="records", lines=True, encoding="utf8", dtype="object"))

def load_csv(file_path):
    csv_list = []
    #空白行を無くすためにnewline=""を引数として渡す
    with open(file_path, "r", encoding="utf8", newline="") as f:
        csv_data = csv.reader(f)
        for current_line in csv_data:
            csv_list.append(str(current_line[0]))
    return csv_list
    del csv_list
    del csv_data

def save_csv(file_path, list_data):
    with open(file_path, "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        for i in list_data:
            writer.writerow(i)

def check_command(data):
    data = str.lower(data)
    count = 0
    color_command_list = ["white2","red2","pink2","orange2","yellow2","green2","cyan2","blue2","purple2","black2","white","red","pink","orange","yellow","green","cyan","blue","purple","black"]
    for i, name in enumerate(color_command_list):
        if name in data:
            if count == 0:
                command_index = i
                count += 1
            break
    if count == 0:
        command_index = 10
    return command_index

def count_code_point(training_data):
    global skip_string
    #valueがnullの時に備えてstr型に変換する
    training_data = str(training_data)
    #Unicodeのコードポイントをインデックスとする配列を用意する
    words = []
    #一文字ずつUnicodeのコードポイントに変換する
    for word_index in range(len(training_data)):
        temp_str = ord(training_data[word_index])
        if temp_str >= 100000:
            skip_string.append(temp_str)
            continue
        words.append(temp_str)
    return words

color_command_list = ["white2","red2","pink2","orange2","yellow2","green2","cyan2","blue2","purple2","black2","white","red","pink","orange","yellow","green","cyan","blue","purple","black"]

try:
    print(str(datetime.datetime.now()),"my_model.h5をロードしています")
    #学習済みのデータと分類器を読み込む
    model = load_model('my_model.h5')
except:
    print(str(datetime.datetime.now()),"my_model.h5をロードできませんでした")

while True:
    test_list =  model.predict(vectorize_sequences([count_code_point(str(input("テスト>")))]))
    size = []
    color_index = []
    for i in test_list:
        for i2,value in enumerate(i):
            size.append(value)
        size_max = max(size)
        for i2,value in enumerate(i):
            if size_max == value:
                color_index.append(i2)
    for i in color_index:
        print(color_command_list[i])
