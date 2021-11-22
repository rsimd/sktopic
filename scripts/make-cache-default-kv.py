import pickle
from octis.evaluation_metrics.coherence_metrics import WECoherencePairwise

if __name__ == "__main__":
    a = WECoherencePairwise()
    save_path = "/workdir/datasets/w2v.default/w2v.3M.300d.bin.pkl"
    with open(save_path, "bw") as f:
        pickle.dump(a._wv, f)