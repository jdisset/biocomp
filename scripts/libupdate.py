from pathlib import Path
import scriptutils as ut
import argparse
import pickle

def main(libpath):
    libpath = Path(libpath)
    libpath.parent.mkdir(parents=True, exist_ok=True)
    print("Updating biocomp lib from google sheets...")
    lib = ut.getLibFromGoogleSheet()
    with open(libpath, "wb") as f:
        pickle.dump(lib, f)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="where to save the biocomp lib")
    args = parser.parse_args()
    main(args.path)
