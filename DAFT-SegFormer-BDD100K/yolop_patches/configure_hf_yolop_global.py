from pathlib import Path
import re

p = Path("lib/config/default.py")
text = p.read_text()

replacements = [
    (r"_C\.GPUS = \(0,1\)", "_C.GPUS = (0,)"),
    (r"_C\.WORKERS = 8", "_C.WORKERS = 4"),
    (r"_C\.MODEL\.PRETRAINED = \"\"", "_C.MODEL.PRETRAINED = \"weights/End-to-end.pth\""),
    (r"_C\.DATASET\.DATAROOT = '.*?'", "_C.DATASET.DATAROOT = '/Users/berkay/projects/daft-drive/DAFT-SegFormer-BDD100K/data/hf_bdd100k_yolop_global/images'"),
    (r"_C\.DATASET\.LABELROOT = '.*?'", "_C.DATASET.LABELROOT = '/Users/berkay/projects/daft-drive/DAFT-SegFormer-BDD100K/data/hf_bdd100k_yolop_global/labels'"),
    (r"_C\.DATASET\.MASKROOT = '.*?'", "_C.DATASET.MASKROOT = '/Users/berkay/projects/daft-drive/DAFT-SegFormer-BDD100K/data/hf_bdd100k_yolop_global/masks'"),
    (r"_C\.DATASET\.LANEROOT = '.*?'", "_C.DATASET.LANEROOT = '/Users/berkay/projects/daft-drive/DAFT-SegFormer-BDD100K/data/hf_bdd100k_yolop_global/lanes'"),
    (r"_C\.DATASET\.DATASET = '.*?'", "_C.DATASET.DATASET = 'HfBddTxtDataset'"),
    (r"_C\.DATASET\.TRAIN_SET = '.*?'", "_C.DATASET.TRAIN_SET = 'train'"),
    (r"_C\.DATASET\.TEST_SET = '.*?'", "_C.DATASET.TEST_SET = 'val'"),
    (r"_C\.TRAIN\.END_EPOCH = 240", "_C.TRAIN.END_EPOCH = 10"),
    (r"_C\.TRAIN\.BATCH_SIZE_PER_GPU =24", "_C.TRAIN.BATCH_SIZE_PER_GPU = 8"),
    (r"_C\.TEST\.BATCH_SIZE_PER_GPU = 24", "_C.TEST.BATCH_SIZE_PER_GPU = 8"),
]

for pattern, repl in replacements:
    text = re.sub(pattern, repl, text)

text = re.sub(r"_C\.TRAIN\.DET_ONLY = False", "_C.TRAIN.DET_ONLY = True", text)

p.write_text(text)
print("patched:", p)
