import pickle 
from gensim.models import KeyedVectors
from argparse import ArgumentParser

parser = ArgumentParser(description='gensimのWECoherenceのために、キャッシュを作る')   
parser.add_argument('source', help='path of source file of pre-trained embeddings. ex."/workdir/datasets/glove.6B/glove.6B.300d.txt"')    # 必須の引数を追加
parser.add_argument('--target', default=None, help='path of cache file"')    # 必須の引数を追加
parser.add_argument("--binary",default=False)
parser.add_argument("--no_header",default=True)
kwargs = parser.parse_args()

def make_wv_cache(source:str,target=None, binary=False, no_header=True):
    kv = KeyedVectors.load_word2vec_format(source,binary=binary,no_header=no_header)
    if target is None:
        target = source.replace(".txt", ".bin.pkl")
    with open(target, "bw") as f:
        pickle.dump(kv, f)

if __name__ == "__main__":
    #make_wv_cache("/workdir/datasets/glove.6B/glove.6B.300d.txt")
    make_wv_cache(**kwargs)