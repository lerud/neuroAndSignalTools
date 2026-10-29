import os
import subprocess
import numpy as np

def doRsync(inputFile):

    sourceHome = "/Volumes/Seagate/map/"
    os.chdir(sourceHome)
    inputFile = inputFile.replace(sourceHome, "")
    print(inputFile)
    subprocess.run('''rsync -avhR --progress "''' + inputFile + '''" /Users/karl/map''', shell=True)
    return "/Users/karl/map/" + inputFile

def calcPC(mat, doPC, comparisonSignal, whichPC=0):
    if doPC:
        outmat = np.zeros((mat.shape[2], mat.shape[0]))
        for i in range(mat.shape[0]):
            submat = mat[i, :, :].T
            u, s, _ = np.linalg.svd(submat)
            PC = u[:, whichPC] * s[whichPC]
            if comparisonSignal is None:
                comparisonSignal = submat.mean(axis=1)
            pearsonR = np.corrcoef(PC, comparisonSignal)[1, 0]
            # print(pearsonR)
            PC *= np.sign(pearsonR)
            outmat[:, i] = PC
    else:
        outmat = mat.mean(axis=1).T
    return outmat
