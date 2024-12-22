import pandas as pd
from config import Config

class Logs:
    @staticmethod
    def save_output_txt(label, predict, epoch):
        path = f"{Config.RESULT_PATH}/val_{epoch}.txt"
        with open(path, 'w') as f:
            f.write("-"*10 + f"Validation result {epoch}" + "-"*10)
            f.write("-"*10 + "> LABEL:\n")
            f.write(label)
            f.write("-"*10 + "> PREDICT:\n")
            f.write(predict)
            f.write("-"*25)
        print(f"File Saved in {path} !")
    
    @staticmethod
    def save_output_csv(ds, epoch):
        data = pd.DataFrame(ds)
        path = f"{Config.RESULT_PATH}/loss_{epoch}.csv"
        data.to_csv(path)
        print(f"Loss CSV saved in {path} !")
        