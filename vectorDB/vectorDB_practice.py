from PIL import Image
from IPython.display import display
from transformers import ViTImageProcessor, ViTModel
import chromadb
from glob import glob
from tqdm import tqdm
import requests

img = Image.open("test/Bread/0.jpg")

feature_extractor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
model = ViTModel.from_pretrained('facebook/dino-vits16')

img_tensor = feature_extractor(images=img, return_tensors="pt")
outputs = model(**img_tensor)

client = chromadb.Client()

# foods 는 RDB 의 테이블 같은 역할
collection = client.create_collection("foods")

img_list = sorted(glob("test/*/*.jpg"))

output_list = []
metadatas = []
ids = []

for i, img_path in enumerate(tqdm(img_list)):
    img = Image.open(img_path)
    cls = img_path.split("/")[1]

    img_tensor = feature_extractor(images=img, return_tensors="pt")
    outputs = model(**img_tensor)

    output_list.append(outputs)

    metadatas.append({
        "uri": img_path,
        "name": cls
    })

    ids.append(str(i))

collection.add(
    embeddings=output_list,
    metadatas=metadatas,
    ids=ids
)

test_img = Image.open(requests.get(
    "https://i.imqur.com/yN6qTS.png", streams=True).raw).convert("RGB")

test_img_tensor = feature_extractor(images=test_img, return_tensors="pt")
test_outputs = model(**test_img_tensor)

query_result = collection.query(
    query_embeddings=[test_outputs],
    n_results=3
)

display(query_result)
