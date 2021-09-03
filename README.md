# NLP若手の会 (YANS) 第16回シンポジウム ハッカソン Dチーム

- [NLP若手の会 (YANS) 第16回シンポジウム ハッカソン](https://yans.anlp.jp/entry/yans2021hackathon) におけるDチームのソースコードです．
- タスクはWikipedia記事から各カテゴリに設定された属性の値を抽出するタスクです．
- Dチームの最終提出したシステムは我々のモデルに加えて，[森羅2019ベースラインシステム](https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/F4-2.pdf)と[BERTベースラインシステム](https://github.com/ujiuji1259/shinra-attribute-extraction)とのアンサンブルしたものになっています．詳細は後日ハッカソンページ内で公開されるスライドをご確認ください．
- [リーダーボード](https://yans2021hackathon.pythonanywhere.com/)の順位は3位，最終順位は3位でした．

## モデルの概要

- 文字ベースのモデルを用い，IOB2形式のラベルによる系列ラベリングとしてタスクを解いています．
- モデルは[東北大BERT](https://huggingface.co/cl-tohoku/bert-base-japanese-char-whole-word-masking)と[SHIBA](https://github.com/octanove/shiba)を利用しています．

## 学習
```bash
sh train.sh
```

| args | candidates |
|---|---|
| --category | City / Company |
| --block | line / char |
| --model | shiba / charbert |

- `--block`は，`line`は一行ずつ処理，`char`は最大文字数ずつ処理する仕様になっています．
- `--max_length`は，`--model`が`shiba`ならば`2048`が上限，`charbert`ならば`512`が上限になっています．

### train.shの例
```bash
python ./src/train.py \
    --input_plain_path ./dataset/yans2021hackathon_plain/ \
    --input_annotation_path ./dataset/yans2021hackathon_annotation/ \
    --output_path ./output/ \
    --category Company \
    --block line \
    --model shiba \
    --max_length 2048 \
    --batch_size 8 \
    --max_epoch 200 \
    --learning_rate 1e-5 \
    --grad_clip 1.0 \
    --seed 0 \
    --cuda 0
```

- `--input_plain_path`にある`./dataset/yans2021hackathon_plain/`の内部には，`(カテゴリ名)/(ページID).txt`となっています．
- `--input_annotatioin_path`にある`./dataset/yans2021hackathon_annotation/`の内部には，`(カテゴリ名)_dist.json`となっています．

## 予測
```bash
sh predict.sh
```

### predict.shの例
```bash
python ./src/predict.py \
    --input_plain_path ./dataset/yans2021hackathon_plain/ \
    --input_annotation_path ./dataset/yans2021hackathon_annotation/ \
    --output_path ./output/ \
    --category Company \
    --block line \
    --model shiba \
    --batch_size 32 \
    --cuda 0
```

## Reference
- [NLP若手の会 (YANS) 第16回シンポジウム ハッカソン - NLP 若手の会](https://yans.anlp.jp/entry/yans2021hackathon)
- [森羅2020-JPでNER入門 - うしのおちちの備忘録](https://kuroneko1259.hatenablog.com/entry/2021/08/12/163855)
- [Wikipedia構造化プロジェクト「森羅2019-JP」 (森羅2019ベースライン)](https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/F4-2.pdf)
- [ujiuji1259/shinra-attribute-extraction (サブワードベースのBERT)](https://github.com/ujiuji1259/shinra-attribute-extraction)
- [upura/yans2021-hackathon (Aチームの取り組み)](https://github.com/upura/yans2021-hackathon)
- [octanove/shiba](https://github.com/octanove/shiba)
- [文字ベースのBERT](https://huggingface.co/cl-tohoku/bert-base-japanese-char-whole-word-masking)
