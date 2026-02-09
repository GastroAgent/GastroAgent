import json
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from sentence_transformers import SentenceTransformer
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from transformers import ViTModel, ViTConfig, AutoTokenizer, AutoModel
# from utils_ import _get_vector_norm

queries = os.listdir('/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/our_eval_data/new_cropped-2004-2010-endovit/processed_sim_data')
documents = os.listdir('/mnt/inaisfs/data/home/tansy_criait/data2/tsy/EndoViT/our_eval_data/new_cropped-2004-2010-endovit/processed_sim_data')

print(queries)

# clip_text_path = '/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/big_clip_trained_weight_disease/CLIPModel_5'
# clip = AutoModel.from_pretrained(clip_text_path).to('cuda').eval()
# tokenizer = AutoTokenizer.from_pretrained(clip_text_path)
# text_inputs = tokenizer(queries, return_tensors="pt", padding=True).to(clip.device)
# text_embeds = clip.get_text_features(**text_inputs)
# text_embeds = text_embeds / _get_vector_norm(text_embeds)
# queries_embeds = text_embeds.cpu().detach().numpy()

# text_inputs = tokenizer(documents, return_tensors="pt", padding=True).to(clip.device)
# text_embeds = clip.get_text_features(**text_inputs)
# doc_embeds = text_embeds / _get_vector_norm(text_embeds)
# doc_embeds = doc_embeds.cpu().detach().numpy()
# similarity = queries_embeds @ doc_embeds.T

# # ### nomic-embed-text-v1.5
# # model = SentenceTransformer("/home/dalhxwlyjsuo/criait_tansy/weight/nomic-embed-text-v1.5", trust_remote_code=True)
# # query_embeddings = model.encode(queries)
# # document_embeddings = model.encode(documents)
# # print(query_embeddings.shape, document_embeddings.shape)
# # similarity = model.similarity(query_embeddings, document_embeddings)

# # ### text2vec
# # model = SentenceTransformer("/home/dalhxwlyjsuo/criait_tansy/weight/text2vec")
# # query_embeddings = model.encode(queries)
# # document_embeddings = model.encode(documents)
# # similarity = model.similarity(query_embeddings, document_embeddings)

# # ### Qwen3-0.6B
# # model = SentenceTransformer("/home/dalhxwlyjsuo/criait_tansy/weight/Qwen3-Embedding-0.6B")
# # query_embeddings = model.encode(queries, prompt_name="query")
# # document_embeddings = model.encode(documents)
# # similarity = model.similarity(query_embeddings, document_embeddings)

# # ### medical_embedded_v4
# # model = SentenceTransformer('/home/dalhxwlyjsuo/criait_tansy/weight/medical_embedded_v4')
# # query_embeddings = model.encode(queries)
# # document_embeddings = model.encode(documents)
# # similarity = model.similarity(query_embeddings, document_embeddings)

# print(similarity)
# sim_matrix = similarity
# # scaler = StandardScaler()
# # sim_matrix = scaler.fit_transform(sim_matrix)

# plt.figure(figsize=(16, 14))
# # sns.heatmap(sim_matrix, annot=False, fmt=".2f", cmap="viridis", cbar=True)
# sns.heatmap(sim_matrix, cmap="coolwarm", vmin=sim_matrix.min(), vmax=sim_matrix.max())
# plt.title("Similarity Matrix Heatmap")
# plt.xlabel("Sample Index")
# plt.ylabel("Sample Index")

# # 保存为图片
# plt.savefig("/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/src/similarity_clip_caption.png", dpi=300, bbox_inches='tight')
# plt.close()
# print(queries)

# # 转换为 DataFrame
# df = pd.DataFrame(sim_matrix, index=queries, columns=documents)
# # 保存为 Excel 文件
# output_filename = '/home/dalhxwlyjsuo/criait_tansy/jmf/EndoViT/similarity_clip_caption.xlsx'
# df.to_excel(output_filename, index=True, header=True)
# print(f"相似度矩阵已成功保存到 {output_filename}")
