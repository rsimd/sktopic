import subprocess
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("fpath",)
parser.add_argument("program_path",default="palmetto-0.1.0-jar-with-dependencies.jar")
parser.add_argument("corpus_path",default="wikipedia_bd")
kwargs = parser.parse_args()

def get_cohs(
    fpath:str,
    program_path:str = "palmetto-0.1.0-jar-with-dependencies.jar",
    corpus_path:str = "wikipedia_bd")->None:
        
    #fpath = "/workdir/scripts/outputs/2021-11-29/03-36-18/wandb/latest-run/files/topics.txt"
    with open(fpath) as f:
        d = f.read()

    tmp = []
    for line in d.split("\n"):
        a = " ".join(line.split()[:10])
        tmp.append(a)
    tmp = "\n".join(tmp)
    fpath2 = fpath.replace("topics.txt", "topicsN10.txt")
    with open(fpath2, "w") as f:
        f.write(tmp)
    for method in ["c_p","c_p","c_v","npmi","uci"]:
        method = "c_v"
        cmd = f"""java -jar {program_path} {corpus_path} {method} {fpath2}"""
        outputs = subprocess.check_output(cmd)
        fpath3 = fpath.replace("topics.txt", f"topicsN10_{method}.txt")    
        with open(fpath3, "w") as f:
            f.write(outputs)
    return

if __name__ == "__main__":
    get_cohs(**kwargs)