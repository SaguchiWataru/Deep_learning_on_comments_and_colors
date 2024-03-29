# PythonとKerasを使って、ニコニコ動画のコメントから色を予測してみた  
## はじめに  

ドワンゴ.LT for student用に作っていたプレゼンテーションをGitHub用に作り直しました。  
2020年はこのイベントが無かったため、GitHubにて公開することにしました。  
最後にソースコードを公開しますので是非ご覧ください。  

![スライド1](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image1.jpg?raw=true)

## 何を作ったのか  

ニコニコ動画にコメントを投稿する際に、おすすめの色を提案する機能を付けたいという思いから、「動画のコメントからコメントに付けられた色を予測する機能」を作成しました。  
今回は、その過程で工夫した点を紹介します。  

![スライド2](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image2.jpg?raw=true)

## 動機  

2018年に、さいたまスーパーアリーナで行われたニコニコ超パーティー2018に行ってきました。  
その際に、前説でダブルMCをしていた百花繚乱さんとドグマ風見さんが、多くのコメントが白だったことを問題視していました。  
その改善策として、おすすめの色(コマンド)を表示する機能があったらカラフルなコメントが増えると思い作ることにしました。  

![スライド3](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image3.jpg?raw=true)

## 何を使ったのか  

### 開発環境 (Kerasとscikit-learnは別々にソースコードを作成しています)  

* Python 3.7  
* Anaconda3 (2019.10)  
* Spyder 3.3.6  
* TensorFlow (GPU) 2.0.0  
* Keras 2.3.1  
* scikit-learn 0.21.3  
* NVIDIA cuda 10.0  
* NVIDIA cuDNN 10.0  
* NVIDIA Graphics Driver 441.28  
* Microsoft Visual Studio 2017 C++ (エディターとしてではなく、Kerasを動作させるため)  
* Microsoft Windows 10 Pro 64bit 1809 (作成したリカバリー用イメージのバージョンに合わせたため少々古くなっています)  

### 沿革  
* 2019/08　独自に作ったアルゴリズム (使うのやめました)  
    * デメリット  
        速度、正確さが限界  
        分類器のダンプ不可  
        GPU非対応  
        深層学習ではない (超単純な機械学習)  

* 2019/09　scikit-learnのGaussianNB (使うのやめました)  
    * デメリット  
        GPU非対応  
        深層学習ではない (機械学習)  

* 2020/02　Keras (今、これを使っています)  
    * デメリット  
        環境構築に時間がかかる  
    * メリット  
        GPU対応なので処理が速い  
        今回の場合、scikit-learnより精度が高い  
        分類器のダンプが可能  
        深層学習である  

最終的に、Kerasを使用した事で実用的なレベルの動作が可能になりました。  

### 参考文献  
* Python：よくわかるPython\[決定版\]  
* scikit-learn：PythonによるAI・機械学習・深層学習アプリのつくり方  
* Keras：PythonとKerasによるディープラーニング  

![スライド4](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image4.jpg?raw=true)

## 学習データと正解ラベル  

学習に用いるデータは、コメントと、色や位置などの情報であるコマンドです。  
コマンドを正解ラベルとして、コメントを学習データとして扱いました。  
データは国立情報学研究所から配布されている「ニコニコ動画コメント等データ」を使用させていただきました。  

![スライド5](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image5.jpg?raw=true)

## データの変換  

学習させるには、学習データと正解ラベルを変換する必要があります。  
どのように変換するかはKerasとscikit-learnで異なります。  
  
Kerasの場合は次のように変換しました。

* 学習データ  
    文字毎にUnicodeの番地に変換し、それをリストにしたものをベクトル化する  

* 正解ラベル  
    コマンドにインデックス番号を割り当て、コマンドをインデックス番号に変換しカテゴリ化する  

ベクトル化とカテゴリ化は難しくて説明ができないので、参考文献の書籍をご覧ください。  

![スライド6](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image6.jpg?raw=true)

scikit-learnのGaussianNBの場合は次のように変換しました。

* 学習データ  
    Unicodeの番地(コードポイント)と対応するリストのインデックスに、文字毎の平均出現回数を格納する  

* 正解ラベル  
    コマンドにインデックス番号を割り当て、コマンドをインデックス番号に変換する  

![スライド7](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image7.jpg?raw=true)

## 正解ラベルの正規化  

ここで問題です。  
今までに、ニコニコ動画に投稿されたコマンドは、何通りあるでしょうか？  

![スライド8](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image8.jpg?raw=true)

正解は約96万通りでした。  
これは独自に調べたデータになりますが、総コメント数が約38億件に対し、正解ラベルであるコマンドは約96万通りでした。  

![スライド9](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image9.jpg?raw=true)

なぜ、約96万通りもあるのか実際にデータを見て確認したところ、フォーマットチェックされていない事が原因でした。 (それでも動くニコニコ動画すごい…)  
実際にあった、フォーマットチェックされていないデータの例をいくつか挙げてみました。  
想定外のデータが沢山ありました。  

![スライド10](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image10.jpg?raw=true)

そこで、約96万通りの正解ラベルを学習させるのは非効率的なので、フォーマットチェックを行い正規化しました。  
Pythonでコマンドを正規化して、色の情報だけ使いました。  
結果、約96万通りの正解ラベルを僅か20通りにできました。  
具体的には、「184 big blueうううううう…」を「blue」に正規化し、さらに、インデックス番号に変更しました。  

```python
#正規化されていないコマンドを1つ引数として受け取る
def check_command(data):
    #全て小文字にする
    data = str.lower(data)
    #色の指定が無かった場合、デフォルトが白になるようにする
    white = True
    #正規化された色の名前のリストを定義する (プレミアム会員は20色使える)
    color_command_list = ["white2","red2","pink2","orange2","yellow2","green2","cyan2","blue2","purple2","black2","white","red","pink","orange","yellow","green","cyan","blue","purple","black"]
    #正規化されていないコマンドの中に、色が含まれているか判別する
    #正規化されていないデータなので、.index()メソッドは使えません
    for i, name in enumerate(color_command_list):
        if name in data:
            command_index = i
            white = False
            break
    #色の指定が無かった場合、デフォルトが白になるようにする
    if white == True:
        #デフォルトの白のインデックス番号は10です (color_command_listの配列の場合)
        command_index = 10
    return command_index
```

![スライド11](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image11.jpg?raw=true)

## メモリ使用量を最適化する  

ここで問題です。  
今までに、ニコニコ動画に投稿されたコメントは、合計何GBあるでしょうか？  

![スライド12](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image12.jpg?raw=true)

正解は、なんと約329GBあります。 (国立情報学研究所より)  

![スライド13](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image13.jpg?raw=true)

今回、学習データと正解ラベルに使用するデータが約329GBありますが、一度に全てを学習させることはコンシューマー向けのパソコンでは不可能でした。  

![スライド14](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image14.jpg?raw=true)

メモリに乗らないサイズのビッグデータを扱うには、分割して学習させる必要があります。  
Kerasはfor文で回すだけで、分割してニューラルネットワークを構築できるので問題ありませんが、scikit-learnのGaussianNBで挑戦する人もいると思うので一応補足すると、scikit-learnのGaussianNBには大きく2種類の学習方法があり、fitとpartial_fitというメソッドがあります。  
気をつけなくてはならないのは、for文を使って分割して学習させるとき、fitメソッドで繰り返し学習させると、前回学習したデータは消えてしまいます。  
つまり、最後の1回の学習しか保存されません。  
そこで、前回の学習データも反映されるpartial_fitメソッドを使って繰り返し学習させる必要があります。  
これが意外と参考書に書いてないので、scikit-learnの公式ドキュメントを読まないと気が付きにくいと思います。  
因みに、繰り返し学習することはインクリメンタル学習と呼ばれています。  

![スライド15](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image15.jpg?raw=true)

しかし、scikit-learnのpartial_fitには大きな壁があります。 (Kerasを使う場合は問題ありません)  

![スライド16](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image16.jpg?raw=true)

scikit-learnの公式ドキュメントを翻訳すると、「インクリメンタル学習(分割して繰り返し学習)を行う場合、1回目のpartial_fitの呼び出しに、全てのパターンの正解ラベルを引数として渡す必要がある。」と書いてあります。  
つまり、過去11年間に投稿された全通りのコマンドを1回で引数に渡す必要があります。  
この問題は、先ほど「正解ラベルの正規化」の見出しで紹介したように、約96万通りのコマンドを正規化して20通りにしたのでこの問題は解決しました。  

![スライド17](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image17.jpg?raw=true)

続いて、実際にfor文を使ってインクリメンタル学習(分割して繰り返し学習)させます。  
そこで、どれくらいのサイズに分割して学習すると、メモリを効率よく使用できるか何回か試行錯誤してみました。  
結論から述べると、物理メモリ64GBのパソコンでは、コメントとコマンドそれぞれKerasの場合は1万件、scikit-learnのGaussianNBの場合は2万件ずつインクリメンタル学習をするとメモリが効率よく安定して使えることがわかりました。  
コメント、コマンド毎に文字数が違う分サイズも異なるので、安定を重視して余裕をもった数にしています。  
データを分割してインクリメンタル学習を行うと、以上のようにメモリがオーバーフローすることなく、学習させることができました。  

![スライド18](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image18.jpg?raw=true)

## 他に工夫したこと (ダイジェスト)  

* 処理時間の短縮① (インクリメンタル学習のタイミングの見直し)  
    これは、「メモリ使用量を最適化する」の見出しの内容の他に、改良した部分があります。  
    改良前は、動画1本毎、または、メモリの使用量が最大に達した時に学習していました。
    一見、効率良く見えますが、コメント数が極端に少ない動画が沢山あった場合は、fitメソッド(Kerasの場合はfitメソッドですが、scikit-learnのGaussianNBの場合はpartial_fitメソッド)が大量に呼び出され、コンピュータのスループットが下がります。  
    それを改良する為に、動画の本数に関係なくコメント1万件毎(Kerasの場合は1万件毎ですが、scikit-learnのGaussianNBの場合は2万件毎)に学習させることにしました。  
    これにより、fitメソッドの呼び出しの回数が大幅に減り、コンピュータのスループットが大幅に上がりました。  

* 処理時間の短縮② (PandasとNumPyを使い短縮)  
    json Linesファイルのデータをfor文で1つずつ整形していたのを、PandasとNumPyで整形することにより処理時間が大幅に削減できました。  
    並行できる処理は並行させるようにしました。  
    さらに、map関数を使うのも同じように型変換の効率を良くする手段ですが、安定版のソースコードのテスト後に気が付き、本記事を書く前に改善を反映できませんでした。  
    次に安定版の動作テストをするタイミングで反映させたいと思っています。  

* 停電後に、自動で途中から復帰  
    これは、万が一デスクトップパソコンの電源が切れてしまった場合、どこまで処理したかを常に出力しているCSV形式のジャーナルを元に、最後に出力したCSVファイルからロールフォワードを自動で行います。  
    CSVファイルの出力直後にリネームしているので、CSVファイルの出力中に電源が切れていないかどうかは、ファイル名で判別できます。  
    万が一、CSVの出力中に電源が切れてしまった事が確認できた場合は、1つ前のバージョンのCSVファイルを元にロールフォワードを自動で行うようにしています。整合性を保つのが大変でした。  

![スライド19](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image19.jpg?raw=true)

## 解決した問題①  

問題点：
白いコメントが大多数なので、過学習が起きる

解決策：
白いコメントを学習の対象外にしました

![スライド20](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image20.jpg?raw=true)

## 解決した問題②  

問題点：
「自分でうｐした動画に自分でコメントしまくってみる」などの動画が原因で過学習が起きる

解決策：
モデルを動画毎に作成するパターンも考えました

![スライド21](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image21.jpg?raw=true)

## 間に合わなかったこと  

* Pythonの動作がWindowsよりも効率が良い、Linux(Ubuntu)に移行できませんでした  
    UbuntuにNVIDIAのドライバーをインストールする際にエラーが発生し、画面が暗転すると復帰できなくなりました。原因は調査中です。

![スライド22](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image22.jpg?raw=true)

## 実際に予測してみた  

今回の製作物を実際に動かしてみたいと思います。  
今回は、ニコニコ動画で公開されている最も古い動画である、[「新・豪血寺一族 -煩悩解放 - レッツゴー！陰陽師」](https://www.nicovideo.jp/watch/sm9)という動画のコメントを学習させました。  
この動画は、現場でも動作確認によく使われているそうです。  
実演は、精度の高かったKerasを使ったソースコードで行いました。  

![test_full_2](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/test_full_2.png?raw=true)

![test_default_size_2_big](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/test_default_size_2_big.png?raw=true)

すると、このような結果が出ました。  

* ああああああああ  
    red  

* 悪霊退散☆悪霊退散☆  
    red  

* ☆☆☆☆☆☆☆☆  
    yellow  

* はんぺん煮  
    cyan

* やばい倒れる陰陽師  
    red

* 成仏しろＹＯ！  
    yellow

この有名な動画を閲覧した事がある方はきっと「大体合ってる」と思っていただけたかと思います。  
個人的には、☆☆☆☆☆☆☆☆がyellowと予測できたことに感動しました。 (半年かけただけあって感動しました)  

![スライド23](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image23.jpg?raw=true)

![スライド24](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image24.jpg?raw=true)

## 今後の抱負  

今後、もしも挑戦するとしたら2019年8月29日にサービスが再開したニコるのデータを元により親しみやすい配色ができるようにしたいと思っています。  

![スライド25](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image25.jpg?raw=true)

## ソースコードと学習済みモデルを公開！  

このプロジェクトをご自身のPC環境で試したいと思う方のために、ソースコードと学習済みモデルを公開することにしました。  
ソースコードにお見苦しい点も多々あるかと思いますが試行錯誤して作りました。  
試用を目的として公開しているので、私的利用の範囲内でご使用ください。 (著作権は佐口航に帰属します)  
特に就活を目的とした、そのままの転用はご遠慮ください。  
また、不具合や損害などが発生しても責任は負えないのでご了承ください。  
開発環境はお使いのパソコンに合わせて構築してください。  

### ダウンロード  

[Keras_color_learn.py (学習用)](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/Keras_color_learn.py)  

[Keras_color_predict.py (予測用)](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/Keras_color_predict.py)  

[my_model.h5 (Keras用の学習済みモデル)](https://drive.google.com/file/d/1hJjKD-KZFYQJZwwG2sXPTDW09Ka3PD7y/view?usp=sharing)  

学習済みモデルは[「新・豪血寺一族 -煩悩解放 - レッツゴー！陰陽師」](https://www.nicovideo.jp/watch/sm9)のコメントとコマンドを学習したデータです。  

![スライド26](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image26.jpg?raw=true)

### 開発環境 (Kerasとscikit-learnは別々にソースコードを作成しています)  

* Python 3.7  
* Anaconda3 (2019.10)  
* Spyder 3.3.6  
* TensorFlow(GPU) 2.0.0  
* Keras 2.3.1  
* scikit-learn 0.21.3  
* NVIDIA cuda 10.0  
* NVIDIA cuDNN 10.0  
* NVIDIA Graphics Driver 441.28  
* Microsoft Visual Studio 2017 C++ (エディターとしてではなく、Kerasを動作させるため)  
* Microsoft Windows 10 Pro 64bit 1809 (作成したリカバリー用イメージのバージョンに合わせたため少々古くなっています)  

## jsonlファイルについて補足  

国立情報学研究所から配布されている「ニコニコ動画コメント等データ」はjsonl形式です。  
jsonlとはjson Linesの略で、jsonファイルとは構造が違うので気を付けてください。  
このファイルを読み込むには、PandasというライブラリとNumPyというライブラリを使いました。  
まず、Pandasというライブラリでデータを整形し、整形されたデータをNumPyというライブラリを使ってNumPy配列を作りました。  
「ニコニコ動画コメント等データ」のjsonlファイルは、Pandasでキーワード引数である"orient"を"records"に、"lines"を"True"に、"encoding"を"utf-8"に、"dtype"を"object"にすると読み込めました。  
続いて、Pandasだと後に列を抽出する際に、エラー(オーバーフロー)が出るのでNumPy配列であるasarryに変換しました。  
arrayではなくasarrayを使用した理由は、配列をコピーした際に参照制約を効かせ、メモリを節約するためです。  

```Python
def load_json_line(file_path):
    return np.asarray(pd.read_json(file_path, orient="records", lines=True, encoding="utf8", dtype="object"))
```

## 最後に  

このリポジトリをご閲覧いただきありがとうございます。  
もしも、このプロジェクトに興味を持って頂けた方がいらっしゃったら幸いです。  

最近は、AtCoderというプログラミングコンテストに毎週参加しています。  
Rating最高値は311です。  
コンテスト実績：[https://atcoder.jp/users/K019C1053](https://atcoder.jp/users/K019C1053)  
保有資格は、基本情報技術者試験です。  
2023年3月卒業予定です。  


![スライド27](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image27.jpg?raw=true)

![スライド28](https://github.com/SaguchiWataru/Deep_learning_on_comments_and_colors/blob/master/images/presentation/ver2020_05_25_18_37/image28.jpg?raw=true)
