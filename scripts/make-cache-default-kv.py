import pickle
from octis.evaluation_metrics.coherence_metrics import WECoherencePairwise
from omegaconf import OmegaConf
import rootpath
import os 

"""
/workdir/datasets/w2v.default/w2v.3M.300d.bin.pkl のようなキャッシュファイルが存在しないときに、キャッシュファイルを作成するために利用するスクリプト。
最低限dummy_kv.txtがあることが前提。
このdummy_kv.txtを作るところからやるのが一番いいが、コレ自体はgistにでも上げておけばいい。
https://gist.githubusercontent.com/rsimd/f07cd4891f7e0e0a01dff3fa72093769/raw/595f56e55aff1c9b67e6b80f5182ab14e14ee8a6/dummy_kv.txt

"""


if __name__ == "__main__":
    a = WECoherencePairwise()
    # prj_root = rootpath.detect()
    # fpath = os.path.join(prj_root, "datasets.yaml")
    # cfg = OmegaConf.load(fpath)
    # save_path = os.path.join(cfg.dataroot, "")
    save_path = "/workdir/datasets/w2v.default/w2v.3M.300d.bin.pkl"
    with open(save_path, "bw") as f:
        pickle.dump(a._wv, f)