# -*- encoding: utf-8 -*-
'''
@create_time: 2023/04/18 16:07:43
@author: lichunyu
'''
import json
from copy import copy


def gen_chat_data(data: list):
    result = []
    history = []
    for _, d in enumerate(data):
        q, a = d["q"], d["a"]
        result.append({"q": q.replace("<n>", "\n"), "a": a.replace("<n>", "\n"), "history": copy(history)})
        history.append([q, a])
    return result


def jsonline(file_path: str):
    with open(file_path, "r") as f:
        while 1:
            if i := f.readline():
                yield json.loads(i)
            else:
                return


def get_data(file_path: str):
    data = []
    for i in jsonline(file_path):
        data.extend(gen_chat_data(i))
    return data


if __name__ == "__main__":
    for i in get_data("../data/data_example.jsonl"):
        print(i)
    ...