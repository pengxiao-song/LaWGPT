from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import sentence_transformers
import numpy as np
import re, os

__all__ = ["Knowledge"]

class Knowledge(object):
    def __init__(self, knowledge_path="./knowledge", embedding_name='GanymedeNil/text2vec-large-chinese') -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_name)
        self.knowledge = FAISS.load_local(knowledge_path, embeddings=self.embeddings)
        # EMBEDDINGS.client = sentence_transformers.SentenceTransformer("/home/wnjxyk/Projects/wenda/model/text2vec-large-chinese", device="cuda")

    def render_index(self, idx, score):      
        indices = self.knowledge.index_to_docstore_id[idx]
        doc = self.knowledge.docstore.search(indices)
        meta_content  = doc.metadata
        return {"title": meta_content['source'], "score": int(score), "content": meta_content["content"]}

    def query_prompt(self, prompt, topk=3, threshold=700):
        embedding = self.knowledge.embedding_function(prompt)
        scores, indices = self.knowledge.index.search(np.array([embedding], dtype=np.float32), topk)
        docs = []
        titles = set()
        for j, i in enumerate(indices[0]):
            if i == -1: continue
            if scores[0][j] > threshold: continue
            item = self.render_index(i, scores[0][j])
            if item["title"] in titles: continue
            titles.add(item["title"])
            docs.append(item)
        return docs

    def get_response(self, output: str) -> str:
        first, res = True, ""
        for doc in output:
            if not first: res += "---\n"
            res += doc["content"]
            first = False
        return res
    
# knowledge = Knowledge()
# answer = knowledge.query_prompt("强奸男性犯法吗？")
# print(answer)
# print(knowledge.get_response(answer))