import jieba
import numpy as np
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import logging

def read_raw(train_file):
    print('reading_csv...')
    train_x = np.genfromtxt(train_file, delimiter=',',
                            skip_header=1, usecols=1, dtype='U20')
    return train_x


def cut_raw(train_file, test_file):
    train_x = read_raw(train_file)
    output = []
    for in_seq in train_x:
        out_seq = list(jieba.cut(in_seq, cut_all=False, HMM=False))
        # while ' ' in  out_seq:
        #     out_seq.remove(' ')
        output.append(out_seq)
    
    test_x = read_raw(test_file)
    for test_in_seq in test_x:
        out_seq = list(jieba.cut(test_in_seq, cut_all=False, HMM=False))
        # while ' ' in  out_seq:
        #     out_seq.remove(' ')
        output.append(out_seq)
    
    return output
def my_tset(model_path):
    model = KeyedVectors.load(model_path, mmap = 'r')
    print(model.get_vector('<unk>'))

def test_dict(model_path):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec.load(model_path)

    print("提供 3 種測試模式\n")
    print("輸入一個詞，則去尋找前一百個該詞的相似詞")
    print("輸入兩個詞，則去計算兩個詞的餘弦相似度")
    print("輸入三個詞，進行類比推理")

    while True:
            try:
                query = input()
                q_list = query.split()
                # q_list[0] = ' '
                if len(q_list) == 1:
                    print("相似詞前 10 排序")
                    res = model.most_similar(q_list[0], topn=10)
                    for item in res:
                        print(item[0]+","+str(item[1]))

                elif len(q_list) == 2:
                    print("計算 Cosine 相似度")
                    res = model.similarity(q_list[0], q_list[1])
                    print(res)
                else:
                    print("%s之於%s，如%s之於" % (q_list[0], q_list[2], q_list[1]))
                    res = model.most_similar([q_list[0], q_list[1]], [
                                             q_list[2]], topn=10)
                    for item in res:
                        print(item[0]+","+str(item[1]))
                print("----------------------------")
            except Exception as e:
                print(repr(e))

def main():

    out_cut = cut_raw('hw6_data/train_x.csv', 'hw6_data/test_x.csv')
    model = Word2Vec(out_cut, size=100, window=5, min_count=1, workers=4)
    model.save("word2vec_noHMM.model")
    model.wv.save("word2vec_noHMM.wv")
    test_dict("word2vec.model")
    # my_tset("word2vec.wv")


if __name__ == '__main__':
    jieba.set_dictionary('dict.txt.big')
    main()
