# import torch
# from transformers import AutoModel, AutoTokenizer
# import time
#
# model = AutoModel.from_pretrained("roberta-base").cuda()
# tokenizer = AutoTokenizer.from_pretrained("roberta-base")
#
# inputs = tokenizer(["This is a test sentence."] * 256, return_tensors="pt", padding=True, truncation=True).to("cuda")
#
# print("✅ GPU:", torch.cuda.get_device_name(0))
# print("⏳ Start inferencji...")
#
# start = time.time()
# for _ in range(100):
#     with torch.no_grad():
#         _ = model(**inputs)
# end = time.time()
#
# print(f"✅ Czas 100 iteracji: {end - start:.2f} sek")


import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))