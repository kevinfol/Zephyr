# Tests the Zephyr installation for any issues by trying to import the main
# zephyr processing script as well as other ancillary scripts
import sys, os

sys.path.append(os.getcwd())
from cli import process_pipeline
import mkdocs

if __name__ == "__main__":
    print("-> Zephyr installed correctly")
