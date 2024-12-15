import torch

class Config:
    SEQ_LEN = 24196
    BATCH_SIZE = 2
    RESULT_PATH = 'result'
    D_MODEL = 512
    D_FF = 2048
    N = 8
    HEAD_SIZE = 8
    WORKERS = 0
    LR = 1e-5
    
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
