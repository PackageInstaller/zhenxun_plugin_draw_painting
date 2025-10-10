import os

if __name__ == "__main__":
    TARGET_DIRECTORY = "temp"
    ORIGINAL_PREFIX = "NKM_UNIT_S"
    NEW_PREFIX = "星之救援" 
    for fn in os.listdir(TARGET_DIRECTORY):
        if fn.endswith(".png") and ORIGINAL_PREFIX in fn:
            nf = fn.replace(ORIGINAL_PREFIX, NEW_PREFIX, 1)
            os.rename(os.path.join(TARGET_DIRECTORY, fn), os.path.join(TARGET_DIRECTORY, nf))
            print(f"{fn} -> {nf}")