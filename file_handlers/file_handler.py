import os

class FileHandler:
    def saveData(fileName, data, reset=False):
        mode = 'w' if reset or not os.path.exists(fileName) else 'a'
        
        with open(fileName, mode) as f:
            f.write(f"{data}\n")