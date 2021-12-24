import subprocess
import argparse
import os 
import numpy as np 


def get_cohline(fname=None, outputs=None):
    if fname is not None and outputs is None:
        with open(fname) as f:
            d = f.read().splitlines()
    else:
        d = outputs.splitlines()
    d = [float(line.split("\t")[1]) for line in d[1:]]
    return np.array(d)

def get_cohs(fpath,
    program_path= "palmetto-0.1.0-jar-with-dependencies.jar",
    corpus_path= "wikipedia_bd",
    methods = ["c_p", "c_a", "c_v", "npmi", "uci"])->dict[str,float]:
    """get topic coherence based on wikipedia

    Parameters
    ----------
    fpath : str
        path of file of topic top words
    program_path : str, optional
        path of palmetto.jar, by default "palmetto-0.1.0-jar-with-dependencies.jar"
    corpus_path : str, optional
        path of index file based on wikipedia, by default "wikipedia_bd"

    Returns
    -------
    dict(str,float)
        TC method name and TC score pair
    """
    #fpath = "/workdir/scripts/outputs/2021-11-29/03-36-18/wandb/latest-run/files/topics.txt"
    with open(fpath) as f:
        d = f.read()
    tmp = []
    for line in d.split("\n"):
        a = " ".join(line.split()[:10])
        tmp.append(a)
    tmp = "\n".join(tmp)
    topicsN10_path = fpath.replace("topics.txt", "topicsN10.txt")
    
    with open(topicsN10_path, "w") as f:
        f.write(tmp)
    print(os.getcwd())
    print("Save topicsN10.txt...")
    print("------------------------")
    results = {}
    #methods = ["c_p", "c_a", "c_v", "npmi", "uci"]
    for ix, method in enumerate(methods):
        cmd = f"java -jar {program_path} {corpus_path} {method} {topicsN10_path}"
        outputs = subprocess.check_output(cmd,shell=True).decode()
        #print(outputs)
        output_path = fpath.replace("topics.txt", f"topicsN10_{method}.txt")
        with open(output_path, "w") as f:
            f.write(outputs)
        score = get_cohline(outputs=outputs)
        #print(score.dtype)
        #print(score)
        key = f"TopicCoherenceOnWikipedia({method})"
        results[key] = np.nanmean(score)
        print(f"({ix+1}/{len(methods)}) {key}== {round(results[key], 3)}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fpath",type=str)
    parser.add_argument("-p","--program_path",type=str,default="palmetto-0.1.0-jar-with-dependencies.jar")
    parser.add_argument("-c", "--corpus_path",type=str,default="wikipedia_bd")
    kwargs = parser.parse_args()

    get_cohs(
        fpath=kwargs.fpath, 
        program_path=kwargs.program_path, 
        corpus_path=kwargs.corpus_path,
        )