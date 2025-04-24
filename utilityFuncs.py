import os
import subprocess

def doRsync(inputFile):

    sourceHome = "/Volumes/Seagate/map/"
    os.chdir(sourceHome)
    inputFile = inputFile.replace(sourceHome, "")
    print(inputFile)
    subprocess.run("rsync -avhR --progress " + inputFile + " /Users/karl/map", shell=True)
    return "/Users/karl/map/" + inputFile
