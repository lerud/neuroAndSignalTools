import os
import subprocess

def doRsync(inputFile):

    sourceHome = "/Volumes/Seagate/map/"
    os.chdir(sourceHome)
    inputFile = inputFile.replace(sourceHome, "")
    subprocess.run(["rsync", "-avhR", "--progress", inputFile, "/Users/karl/map"])
    return "/Users/karl/map/" + inputFile
