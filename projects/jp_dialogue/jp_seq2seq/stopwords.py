#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# From here https://www.ranks.nl/stopwords/japanese
sw = [
    'これ',
    'それ',
    'あれ',
    'この',
    'その',
    'あの',
    'ここ',
    'そこ',
    'あそこ',
    'こちら',
    'どこ',
    'だれ',
    'なに',
    'なん',
    '何',
    '私',
    '貴方',
    '貴方方',
    '我々',
    '私達',
    'あの人',
    'あのかた',
    '彼女',
    '彼',
    'です',
    'あります',
    'ある',
    'おります',
    'おる',
    'います',
    'いる',
    'は',
    'が',
    'の',
    'に',
    'を',
    'で',
    'え',
    'から',
    'まで',
    'より',
    'も',
    'どの',
    'と',
    'し',
    'それで',
    'しかし'
]

# add punctuation
STOPWORDS = sw + [".", "?", "!", ",", '。', '？', '！', '、', '・']
