# libtorch semantic segmentation

このレポジトリはLibTorchでセマンティックセグメンテーション（推論のみ）を行うものです。

## サンプルの動かし方
### LibTorchの動作確認
1. LibTorchを公式サイトからダウンロード
2. ダウンロードしたファイルを展開する
3. 以下のコマンドを実行
```
$ cd libtorch_semantic_segmentation/hello_world
$ mkdir build
$ cd build
$ cmake ..
```
4. 上記が問題なくできれば、./libtorch_testを実行

### セマンティックセグメンテーションの実行
1. modelディレクトリにtorchscript化したモデルファイル、imagesディレクトリに推論対象の画像をそれぞれ格納
2. 動作確認時と同様にcmakeを実行
3. ./libtorch_semasegで推論が実行可能
