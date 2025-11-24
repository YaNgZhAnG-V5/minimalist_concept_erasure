import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import random

import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import FluxPipeline
from torch.autograd import Variable
from tqdm import tqdm
from utils import gumbel_softmax, save_to_dict

LEN_EN_3K_VOCAB = 3000


def get_english_tokens():
    data_path = "data/english_3000.csv"
    df = pd.read_csv(data_path)
    vocab = {}
    for ir, row in df.iterrows():
        vocab[row["word"]] = ir
    assert len(vocab) == LEN_EN_3K_VOCAB
    return vocab


def detect_special_tokens(text):
    text = text.lower()
    for i in range(len(text)):
        if text[i] not in "abcdefghijklmnopqrstuvwxyz</>":
            return True
    return False


def retrieve_embedding_token(model_name, query_token, vocab="EN3K"):
    if vocab == "EN3K":
        if model_name == "SD-v1-4":
            embedding_matrix = torch.load("models/embedding_matrix_dict_EN3K.pt")
        elif model_name == "SD-v2-1":
            embedding_matrix = torch.load("models/embedding_matrix_dict_EN3K_v2-1.pt")
        elif model_name == "Flux-schnell":
            embedding_matrix = torch.load("models/embedding_matrix_dict_EN3K_schnell.pt")
        else:
            raise ValueError(
                "model_name should be either 'SD-v1-4', 'SD-v2-1', 'Flux-schnell', 'Flux-dev', 'SD-v3-0', 'SD-v3-5'"
            )
        if query_token in embedding_matrix:
            return embedding_matrix[query_token]
    else:
        raise ValueError("vocab should be either 'Flux' or 'EN3K'")


@torch.no_grad()
def create_embedding_matrix(
    model, start=0, end=1000, model_name="Flux-schnell", save_mode="array", remove_end_token=False, vocab="EN3K"
):

    tokenizer_vocab = get_vocab(model, vocab=vocab)

    if save_mode == "array":
        all_embeddings = []
        for token in tqdm(tokenizer_vocab.keys()):
            print(token, tokenizer_vocab[token])
            if tokenizer_vocab[token] < start or tokenizer_vocab[token] >= end:
                continue
            # print(token, tokenizer_vocab[token])
            if remove_end_token:
                token_ = token.replace("</w>", "")
            else:
                token_ = token
            emb_, pooled_, ids_ = model.encode_prompt(
                prompt=[token_],
                prompt_2=[token_],
                device=model.device,
                num_images_per_prompt=1,
                max_sequence_length=256,
            )
            all_embeddings.append(emb_)
        return torch.cat(all_embeddings, dim=0)  # shape (49408, 77, 768)
    elif save_mode == "dict":
        all_embeddings = {}
        for token in tqdm(tokenizer_vocab.keys()):
            if tokenizer_vocab[token] < start or tokenizer_vocab[token] >= end:
                continue
            # print(token, tokenizer_vocab[token])
            if remove_end_token:
                token_ = token.replace("</w>", "")
            else:
                token_ = token
            emb_, pooled_, ids_ = model.encode_prompt(
                prompt=[token_],
                prompt_2=[token_],
                device=model.device,
                num_images_per_prompt=1,
                max_sequence_length=256,
            )
            all_embeddings[token] = emb_
        return all_embeddings
    else:
        raise ValueError("save_mode should be either 'array' or 'dict'")


@torch.no_grad()
def search_closest_tokens(
    concept,
    model,
    k=10,
    reshape=True,
    sim="cosine",
    model_name="Flux-schnell",
    ignore_special_tokens=True,
    vocab="EN3K",
):
    """
    Given a concept, i.e., "nudity", search for top-k closest tokens in the embedding space
    """
    tokenizer_vocab = get_vocab(model, vocab=vocab)
    # inverse the dictionary
    tokenizer_vocab_indexing = {v: k for k, v in tokenizer_vocab.items()}

    # Get the embedding of the concept
    central_concept_embeds, _, _ = model.encode_prompt(
        prompt=[concept],
        prompt_2=[concept],
        device=model.device,
        num_images_per_prompt=1,
        max_sequence_length=256,
    )

    # Calculate the cosine similarity between the concept and all tokens
    # load the embedding matrix
    all_similarities = []
    if vocab == "EN3K":
        if model_name == "Flux-schnell":
            embedding_matrix = torch.load("models/embedding_matrix_array_EN3K_schnell.pt")
        elif model_name == "Flux-dev":
            embedding_matrix = torch.load("models/embedding_matrix_array_EN3K_dev.pt")
        else:
            raise ValueError("model_name should be either 'Flux-schnell' or 'Flux-dev'")

        central_concept_embeds = central_concept_embeds[:, 0, :]
        embedding_matrix = embedding_matrix[:, 0, :].to(central_concept_embeds.device)
        # import pdb; pdb.set_trace()

        if reshape:
            central_concept_embeds = central_concept_embeds.view(central_concept_embeds.size(0), -1)
            embedding_matrix = embedding_matrix.view(embedding_matrix.size(0), -1)
        if sim == "cosine":
            similarities = F.cosine_similarity(central_concept_embeds, embedding_matrix, dim=-1)
        elif sim == "l2":
            similarities = -F.pairwise_distance(central_concept_embeds, embedding_matrix, p=2)
        all_similarities.append(similarities)
    else:
        raise ValueError("vocab should be either 'CLIP' or 'EN3K'")

    similarities = torch.cat(all_similarities, dim=0)
    # sorting the similarities
    sorted_similarities, indices = torch.sort(similarities, descending=True)
    print(f"sorted_similarities: {sorted_similarities[:10]}")
    print(f"indices: {indices[:10]}")

    sim_dict = {}
    for im, i in enumerate(indices):
        if ignore_special_tokens:
            if detect_special_tokens(tokenizer_vocab_indexing[i.item()]):
                continue
        token = tokenizer_vocab_indexing[i.item()]
        sim_dict[token] = sorted_similarities[im]

    top_k_tokens = list(sim_dict.keys())[:k]
    # print(f"Top-{k} closest tokens to the concept {concept} are: {top_k_tokens}")
    return top_k_tokens, sim_dict


def save_embedding_matrix(model, model_name="Flux-schnell", save_mode="array", vocab="EN3K"):
    if vocab == "EN3K":
        embedding_matrix = create_embedding_matrix(
            model, start=0, end=LEN_EN_3K_VOCAB, model_name=model_name, save_mode=save_mode, vocab="EN3K"
        )
        if model_name == "Flux-schnell":
            print("[Flux-schnell] embedding")
            torch.save(embedding_matrix, f"models/embedding_matrix_{save_mode}_EN3K_schnell.pt")
        elif model_name == "Flux-dev":
            print("[Flux-dev] embedding")
            torch.save(embedding_matrix, f"models/embedding_matrix_{save_mode}_EN3K_dev.pt")
    else:
        raise ValueError("vocab should be either 'T5' or 'EN3K'")


@torch.no_grad()
def get_vocab(pipe, vocab="EN3K"):

    if pipe is not None:
        # Flux
        tokenizer_vocab = pipe.tokenizer_2.vocab  # vocab_size 32,100
    if vocab == "EN3K":
        tokenizer_vocab = get_english_tokens()  # vocab_size 3,000

    return tokenizer_vocab


def my_kmean(sorted_sim_dict, num_centers, compute_mode):
    if compute_mode == "numpy":
        import numpy as np
        from sklearn.cluster import KMeans

        similarities = np.array([sorted_sim_dict[token].item() for token in sorted_sim_dict])
        similarities = similarities.reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(similarities)
        # print(f"Cluster centers: {kmeans.cluster_centers_}")
        # print(f"Cluster labels: {kmeans.labels_}")
        cluster_centers = kmeans.cluster_centers_
    elif compute_mode == "torch":
        from torch_kmeans import KMeans

        similarities = torch.stack([sorted_sim_dict[token] for token in sorted_sim_dict])
        similarities = torch.unsqueeze(similarities, dim=0)
        similarities = torch.unsqueeze(similarities, dim=2)  # [1, N, 1]
        # print('similarities shape:', similarities.shape)
        kmeans = KMeans(n_clusters=num_centers).fit(similarities)
        import pdb

        pdb.set_trace()
        # print(f"Cluster centers: {kmeans.cluster_centers}")
        # print(f"Cluster labels: {kmeans.labels}")
        cluster_centers = kmeans.cluster_centers

    # find the closest token to each cluster center
    cluster_dict = {}
    for i, center in enumerate(cluster_centers):
        closest_token = None
        closest_similarity = -float("inf")
        for j, token in enumerate(sorted_sim_dict):
            similarity = sorted_sim_dict[token].item()
            if abs(similarity - center) < abs(closest_similarity - center):
                closest_similarity = similarity
                closest_token = token
        cluster_dict[closest_token] = (closest_token, closest_similarity, i)
    # print(f"Cluster dictionary: {cluster_dict}")

    return cluster_dict


@torch.no_grad()
def learn_k_means_from_input_embedding(sim_dict, num_centers=5, compute_mode="numpy"):
    """
    Given a model, a set of tokens, and a concept, learn k-means clustering on the search_closest_tokens's output
    """
    if num_centers <= 0:
        print("Number of centers should be greater than 0. Returning the tokens themselves.")
        return list(sim_dict.keys())
    if len(list(sim_dict.keys())) <= num_centers:
        print("Number of tokens is less than the number of centers. Returning the tokens themselves.")
        return list(sim_dict.keys())

    return list(my_kmean(sim_dict, num_centers, compute_mode).keys())


def create_prompt(word, retrieve=True, vocab="EN3K"):
    if retrieve:
        return retrieve_embedding_token(model_name="Flux-schnell", query_token=word, vocab=vocab)


if __name__ == "__main__":
    flux_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")

    # ddd = save_embedding_matrix(pipe)
    # top_k_tokens, sim_dict = search_closest_tokens('soccer', pipe)
    gumbel_k_closest = 1000
    gumbel_num_centers = 50
    history_dict = {}

    if not os.path.exists("models/embedding_matrix_dict_EN3K_schnell.pt"):
        save_embedding_matrix(flux_model, model_name="Flux-schnell", save_mode="dict", vocab="EN3K")

    erased_words = ["nude"]
    # (b) similarities between tokens
    tokens_embedding = []
    all_sim_dict = dict()
    for word in erased_words:
        top_k_tokens, sorted_sim_dict = search_closest_tokens(word, flux_model, k=gumbel_k_closest)
        tokens_embedding.extend(top_k_tokens)
        all_sim_dict[word] = {key: sorted_sim_dict[key] for key in top_k_tokens}

    # (c) perserved
    if gumbel_num_centers > 0:
        assert (
            gumbel_num_centers % len(erased_words) == 0
        ), "Number of centers should be divisible by number of erased words"
    preserved_dict = dict()

    # (d) k means
    for word in erased_words:
        temp = learn_k_means_from_input_embedding(sim_dict=all_sim_dict[word], num_centers=gumbel_num_centers)
        preserved_dict[word] = temp

    history_dict = save_to_dict(preserved_dict, "preserved_set_0", history_dict)

    # (e) create a matrix of embeddings for the preserved set
    print("Creating preserved matrix")
    one_hot_dict = dict()
    preserved_matrix_dict = dict()
    for erase_word in erased_words:
        preserved_set = preserved_dict[erase_word]
        pbar = tqdm(preserved_set)
        for i, word in enumerate(pbar):
            if i == 0:
                preserved_matrix = create_prompt(word)
            else:
                preserved_matrix = torch.cat((preserved_matrix, create_prompt(word)), dim=0)
            pbar.set_description("Index: {0}, Word: {1}, Dimesion: {2}".format(i, word, preserved_matrix.shape))
            # print(i, word, preserved_matrix.shape)
        # preserved_matrix = torch.cat([create_prompt(word) for word in preserved_set], dim=0) # [n, 77, 768]
        preserved_matrix = preserved_matrix.flatten(start_dim=1)  # [n, 77*768]
        one_hot = torch.zeros((1, preserved_matrix.shape[0]), device="cuda:0", dtype=preserved_matrix.dtype)  # [1, n]
        one_hot = one_hot + 1 / preserved_matrix.shape[0]
        one_hot = Variable(one_hot, requires_grad=True)
        print(one_hot.shape, preserved_matrix.shape)
        print(one_hot)
        one_hot_dict[erase_word] = one_hot
        preserved_matrix_dict[erase_word] = preserved_matrix

    print("one_hot_dict:", one_hot_dict)  # [bs, 50]
    history_dict = save_to_dict(one_hot_dict, "one_hot_dict_0", history_dict)

    word = random.sample(erased_words, 1)[0]
    abc = torch.matmul(
        gumbel_softmax(one_hot_dict[word], temperature=2, hard=1), preserved_matrix_dict[word]
    ).unsqueeze(0)
    import pdb

    pdb.set_trace()
