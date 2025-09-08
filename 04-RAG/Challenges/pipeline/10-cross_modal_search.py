import os
import glob
import faiss
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

IMAGES_PATHS = 'data/raw/images'
OUTPUT_INDEX_PATH = 'data/embeddings/cross-model-search.index'

model = SentenceTransformer('clip-ViT-B-32')

def generate_clip_embeddings(images_path, model):
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(images_path, '**', ext), recursive=True))
    print(f"Found {len(image_paths)} images: {image_paths}")
    embeddings = []
    valid_image_paths = []
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            embedding = model.encode(image)
            embeddings.append(embedding)
            valid_image_paths.append(img_path)
        except Exception as e:
            print(f"Error on {img_path}: {e}")
    return embeddings, valid_image_paths

embeddings, image_paths = generate_clip_embeddings(IMAGES_PATHS, model)

def faiss_create_index(embeddings,image_paths,vector_path):
    dimession = len(embeddings[0])
    index = faiss.IndexFlatIP(dimession)
    index = faiss.IndexIDMap(index)

    vectors = np.array(embeddings).astype(np.float32)
    index.add_with_ids(vectors,np.array(range(len(embeddings))))

    faiss.write_index(index, vector_path)

    with open(vector_path + '.paths','w') as f:
        for img_paths in image_paths:
            f.write(img_paths +'\n')
    return index 

index = faiss_create_index(embeddings,image_paths,OUTPUT_INDEX_PATH)

# --- Retrival via image ---
def retrieve_similar_image(query,model,index,image_paths,top_k = 3):
    if query.endswith(('.jpg','.png', '.jpeg')):
        query = Image.open(query)
    
    query_encode = model.encode(query)
    query_encode = query_encode.astype(np.float32).reshape(1,-1)


    distance, indices = index.search(query_encode,top_k)

    retrieved_images = [image_paths[int(idx)] for idx in indices[0]]

    return query, retrieved_images

query = "data/raw/images/image_005.png"
# query = "Kurutha"
query, retrived_image = retrieve_similar_image(query, model,index,image_paths,top_k = 3)


for i,imageName in enumerate(retrived_image):
    img = Image.open(imageName)
    plt.subplot(1,len(retrived_image), i+1)
    plt.imshow(img)
    plt.title(f"image {i+1}")
plt.show()



