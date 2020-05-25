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

#展開前のzipファイルがあるディレクトリ
zip_directory = "D:\\comment"

#zipファイルの展開先のディレクトリ
unfolded_zip_directory = "."
#unfolded_zip_directory = "D:\\unfolded"

#学習データのインデックス(jsonlファイルの「コメント」)
training_data_index = 2

#正解ラベルのインデックス(jsonlファイルの「コマンド」)
answer_label_index = 3

"""
#質問の回答を記録する
question04 = input("初期化しますか？ y(Yes) or n(No)>")
while question04 != "y" and question04 != "n":
    question04 = input("初期化しますか？ y(Yes) or n(No)>")

question01 = input("zipファイルを展開しますか？ y(Yes) or n(No)>")
while question01 != "y" and question01 != "n":
    question01 = input("zipファイルを展開しますか？ y(Yes) or n(No)>")
"""

#最後に経過時間を表示するために開始時刻を記録する
start_time = datetime.datetime.now()

"""
print(str(datetime.datetime.now()),"初期化しています")
if question04 == "y":
    try:
        os.remove("all_jsonl_file_path_list.csv")
        print(str(datetime.datetime.now()),"all_jsonl_file_path_list.csvを削除しました")
    except:
        print(str(datetime.datetime.now()),"all_jsonl_file_path_list.csvを削除できませんでした")
    try:
        os.remove("my_model.h5")
        print(str(datetime.datetime.now()),"my_model.h5を削除しました")
    except:
        print(str(datetime.datetime.now()),"my_model.h5を削除できませんでした")
"""

#何件目のzipファイルを読み込んでいるか
count_current_zip = 0
#正常に読み込めたzipファイルの件数
count_success_zip = 0
#読み込めなかったzipファイルの件数
count_error_zip = 0
"""
if question01 == "y":
    print(str(datetime.datetime.now()),"zipファイルを検索しています")
    #zipファイルのファイルパスの一覧を取得し順番に読み込む
    for zip_file_path in sorted(glob.glob((zip_directory) + "\\*.zip")):
        #何件目のzipファイルを読み込んでいるかインクリメントする
        count_current_zip += 1
        print(str(datetime.datetime.now()),zip_file_path + "を展開しています(" + str(count_current_zip) + "件目) " + str(count_success_zip) + "件成功 " + str(count_error_zip) + "件失敗")
        try:
            #zipファイルを展開する
            with zipfile.ZipFile(zip_file_path) as sm_zip:
                sm_zip.extracall(unfolded_zip_directory)
            #正常に読み込めたzipファイルの件数をインクリメントする
            count_success_zip += 1
        except:
            #読み込めなかったzipファイルの件数をインクリメントする
            count_error_zip += 1
"""

#何件目のjsonlファイルを読み込んでいるか
count_current_jsonl = 0
#正常に読み込めたjsonlファイルの件数
count_success_jsonl = 0
#読み込めなかったjsonlファイルの件数
count_error_jsonl = 0
#すべての正解ラベルから重複を取り除いた集合
all_answer_label_set = set()
#定期的に経過時間をリセットし、経過時間を表示するために現在時刻を記録する
temp_time = datetime.datetime.now()
#既に処理済みのcsvファイルがあるか確認する

model = models.Sequential()
model.add(layers.Dense(1000, activation='relu', input_shape=(100000,)))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(128, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#何件目のjsonlファイルを読み込んでいるか
count_current_jsonl = 0
#正常に読み込めたjsonlファイルの件数
count_success_jsonl = 0
#読み込めなかったjsonlファイルの件数
count_error_jsonl = 0
#すべての学習データ
training_data_list = []
#学習データを可視化しやすくする
training_data_list_str = []
#すべての正解ラベル
answer_label_list = []
#処理済みの合計ファイルサイズ
all_file_size = 0
#定期的に経過時間をリセットし、経過時間を表示するために現在時刻を記録する
temp_time = datetime.datetime.now()
#読み込めなかったjsonlファイルを記録する
failed_jsonl_list = []
#読み込めなかった文字列を記録する
skip_string = []
#セーブのタイミングにインターバルを設ける
save_count = 0
#
jpg_count = 0

#停電対策のために処理済みのjsonlファイルのパスの一覧が記録されているcsvファイルを検索する
try:
    print(str(datetime.datetime.now()),"completed_jsonl_files_info.csvをロードしています")
    completed_jsonl_files_info = load_csv("completed_jsonl_files_info.csv")
except:
    #ファイルが存在しなかったからファイルを新規作成する
    print(str(datetime.datetime.now()),"completed_jsonl_files_info.csvをロードできませんでした")
    #処理済みのファイルパスの一覧を保存するリスト
    completed_jsonl_files_info = []

#学習済みのデータと分類器が存在するか判定する
try:
    print(str(datetime.datetime.now()),"my_model.h5をロードしています")
    #学習済みのデータと分類器を読み込む
    model = load_model('my_model.h5')
except:
    print(str(datetime.datetime.now()),"my_model.h5をロードできませんでした")
    print(str(datetime.datetime.now()),"最初から学習します")

try:
    print(str(datetime.datetime.now()),"all_jsonl_file_path_list.csvをロードしています")
    all_jsonl_file_path_list = load_csv("all_jsonl_file_path_list.csv")
except:
    print(str(datetime.datetime.now()),"all_jsonl_file_path_list.csvをロードできませんでした")
    print(str(datetime.datetime.now()),"jsonlファイルを検索しています")
    #jsonlファイルのファイルパスの一覧を取得し順番に読み込む
    all_jsonl_file_path_list = []
    for jsonl_file_path in sorted(glob.glob(unfolded_zip_directory + "\\*\\*.jsonl")):
        all_jsonl_file_path_list.append(jsonl_file_path)
    print(str(datetime.datetime.now()),"all_jsonl_file_path_list.csvをセーブしています")
    save_csv("all_jsonl_file_path_list.csv", all_jsonl_file_path_list)

for jsonl_file_path in all_jsonl_file_path_list:
    save_count += 1
    all_file_size += os.path.getsize(jsonl_file_path)
    #停電対策のために処理済みのjsonlファイルであるか確認する
    if jsonl_file_path in completed_jsonl_files_info:
        continue
    else:
        #try:
            #何件目のjsonlファイルを読み込んでいるかインクリメントする
            count_current_jsonl += 1
            print(str(datetime.datetime.now()),jsonl_file_path + "をロードしています(" + str(count_current_jsonl) + "件目)")
            #jsonlファイルを1つ読み込む
            sm_np = load_json_line(jsonl_file_path)
            #sm_list[:, answer_label_index]とすると一部のjsonlファイルでエラーが出るため、やむを得ずfor文を使う
            print(str(datetime.datetime.now()),"コメントをコードポイントに変換しています")
            for i in range(len(sm_np)):
                #valueがnullだと学習できないので、str型に変換してリストに追加する
                command_index = str(check_command(str(sm_np[i][answer_label_index])))
                if command_index == "10":
                    continue
                training_data_list.append(count_code_point(str(sm_np[i][training_data_index])))
                training_data_list_str.append(str(sm_np[i][training_data_index]))
                answer_label_list.append(command_index)
                #メモリを節約するために途中でインクリメンタル学習を行う
                #物理メモリが64GBの場合100回に1回行うのが適している
                if len(answer_label_list) == 10000:
                    print("\n" + str(datetime.datetime.now()),"インクリメンタル学習をしています")
                    
                    #ベクトル化とカテゴリ化
                    x_train = vectorize_sequences(training_data_list)
                    one_hot_train_labels = to_one_hot(answer_label_list)
                    #one_hot_train_labels = to_categorical(answer_label_list)
                    
                    #学習用と精度の計測用で半分に分ける
                    slice_index = len(x_train) // 2
                    x_val = x_train
                    #x_val = x_train[:slice_index]
                    partial_x_train = x_train
                    #partial_x_train = x_train[slice_index:]
                    y_val = one_hot_train_labels
                    #y_val = one_hot_train_labels[:slice_index]
                    partial_y_train = one_hot_train_labels
                    #partial_y_train = one_hot_train_labels[slice_index:]
                    
                    #学習させる
                    history = model.fit(partial_x_train,
                    partial_y_train,
                    initial_epoch=0,
                    epochs=20,
                    batch_size=1280,
                    validation_data=(x_val, y_val)
                    )
                    
                    #精度を可視化
                    loss = history.history['loss']
                    val_loss = history.history['val_loss']
                    epochs = range(1, len(loss) + 1)
                    plt.plot(epochs, loss, 'bo', label='Training loss')
                    plt.plot(epochs, val_loss, 'b', label='Validation loss')
                    plt.title('Training and validation loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend()
                    jpg_count += 1
                    #損失のグラフを保存する
                    plt.savefig(str(jpg_count) + ".jpg")
                    #リセットする
                    plt.figure()
                    
                    test_status = ""
                    while test_status != "":
                        test_status = str(input("テスト>"))
                        test_list =  model.predict(vectorize_sequences([count_code_point(test_status)]))
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
                    
                    #初期化
                    training_data_list = []
                    answer_label_list = []
                    training_data_list_str = []
                    
            if save_count % 1 == 0:
                print(str(datetime.datetime.now()),"my_model.h5をセーブしています")
                #学習済みデータを保存する
                model.save("my_model.h5")
                #処理済みのファイルパスをcsvファイルに保存する
                #save_csv("completed_jsonl_files_info_" + str(datetime.datetime.now()) + ".csv", completed_jsonl_files_info)
            
            #メモリを節約するために初期化する
            sm_np = []
            #読み込めたjsonlファイルの件数をインクリメントする
            count_success_jsonl += 1
            #処理済みのファイルパスを記録する
            completed_jsonl_files_info.append(jsonl_file_path)
            
            print(str(datetime.datetime.now()),"今までに読み込めなかったjsonlファイルの件数 : " + str(count_error_jsonl) + "件")
            print(str(datetime.datetime.now()),"今までに読み込めなかった文字数 : " + str(len(skip_string)) + "件")
            print(str(datetime.datetime.now()),"経過時間 : " + str(datetime.datetime.now() - start_time))
            print(str(datetime.datetime.now()),"処理済みの合計ファイルサイズ : " + str(all_file_size / 1024 / 1024 / 1024) + "GB")
            print(str(datetime.datetime.now()),"未学習データ : " + str(len(answer_label_list)) + "件")
        #except:
            #読み込めなかったjsonlファイルの件数をインクリメントする
            #count_error_jsonl += 1
            #failed_jsonl_list.append(jsonl_file_path)
            #ave_csv("failed_jsonl_list.csv", failed_jsonl_list)

print(str(datetime.datetime.now()),"インクリメンタル学習をしています")
#ベクトル化とカテゴリ化
x_train = vectorize_sequences(training_data_list)
one_hot_train_labels = to_categorical(answer_label_list)

#学習用と精度の計測用で半分に分ける
slice_index = len(x_train) // 2
x_val = x_train
#x_val = x_train[:slice_index]
partial_x_train = x_train
#partial_x_train = x_train[slice_index:]
y_val = one_hot_train_labels
#y_val = one_hot_train_labels[:slice_index]
partial_y_train = one_hot_train_labels
#partial_y_train = one_hot_train_labels[slice_index:]

batch_size_num = 128
if len(partial_x_train) < 128:
    batch_size_num = len(partial_x_train)

#学習させる
history = model.fit(partial_x_train,
partial_y_train,
epochs=20,
batch_size=batch_size_num,
validation_data=(x_val, y_val))

#精度を可視化
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
jpg_count += 1
#損失のグラフを保存する
plt.savefig(str(jpg_count) + ".jpg")
#リセットする+
plt.figure()

print(str(datetime.datetime.now()),"my_model.h5をセーブしています")
#学習済みデータを保存する
model.save('my_model.h5')
#処理済みのファイルパスをcsvファイルに保存する
#save_csv("completed_jsonl_files_info.csv", completed_jsonl_files_info)
#初期化
training_data_list = []
answer_label_list = []

print(str(datetime.datetime.now()),"☆☆☆　結果報告　☆☆☆")
try:
    print(str(datetime.datetime.now()),"正常に展開できたzipファイルの件数 : " + str(count_success_zip))
except:
    pass
try:
    print(str(datetime.datetime.now()),"展開できなかったzipファイルの件数 : " + str(count_error_zip))
except:
    pass
try:
    print(str(datetime.datetime.now()),"正常に読み込めたjsonlファイルの件数 : " + str(completed_jsonl_files_info))
except:
    pass
try:
    print(str(datetime.datetime.now()),"読み込めなかったjsonlファイルの件数 : " + str(count_error_jsonl))
except:
    pass
try:
    print(str(datetime.datetime.now()),"今までに読み込めなかった文字数 : " + str(len(skip_string)) + "件")
except:
    pass
try:
    print(str(datetime.datetime.now()),"経過時間 : " + str(datetime.datetime.now() - start_time))
except:
    pass
try:
    print(str(datetime.datetime.now()),"処理済みの合計ファイルサイズ : " + str(all_file_size / 1024 / 1024 / 1024) + "GB")
except:
    pass
