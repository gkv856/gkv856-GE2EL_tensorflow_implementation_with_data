# import torch
# import torch.autograd as grad
# import torch.nn.functional as F
#
# from config.hyper_parameters import hyper_paramaters as hp
#
#
# # eq. (1)
# def get_centroids(embeddings):
#     # calculating centroid of all the utterances  as per eq. (1)
#     # centroid for neg similarity
#     centroids = embeddings.mean(dim=1)
#     return centroids
#
#
# # eq (8)
# def get_centroid(embeddings, speaker_num, utterance_num):
#     # c(j−i) = 1  # mmX=1  # ejm, (8)
#     # eq 8
#     centroid = 0
#     for utterance_id, utterance in enumerate(embeddings[speaker_num]):
#         if utterance_id == utterance_num:
#             continue
#         centroid = centroid + utterance
#     centroid = centroid / (len(embeddings[speaker_num]) - 1)
#     return centroid
#
#
# def get_utterance_centroids(embeddings):
#     """
#     Returns the centroids for each utterance of a speaker, where
#     the utterance centroid is the speaker centroid without considering
#     this utterance
#
#     Shape of embeddings should be:
#         (speaker_ct, utterance_per_speaker_ct, embedding_size)
#     """
#     sum_centroids = embeddings.sum(dim=1)
#     # we want to subtract out each utterance, prior to calculating the
#     # the utterance centroid
#     sum_centroids = sum_centroids.reshape(sum_centroids.shape[0], 1, sum_centroids.shape[-1])
#     # we want the mean but not including the utterance itself, so -1
#     num_utterances = embeddings.shape[1] - 1
#     centroids = (sum_centroids - embeddings) / num_utterances
#     return centroids
#
#
# # calculates cos similarities of eq. (5)
# def get_cos_sim(embeddings, centroids):
#     # number of utterances per speaker
#     num_utterances = embeddings.shape[1]
#     utterance_centroids = get_utterance_centroids(embeddings)
#
#     # flatten the embeddings and utterance centroids to just utterance,
#     # so we can do cosine similarity
#     utterance_centroids_flat = utterance_centroids.view(utterance_centroids.shape[0] * utterance_centroids.shape[1], -1)
#     embeddings_flat = embeddings.view(embeddings.shape[0] * num_utterances, -1)
#     # the cosine distance between utterance and the associated centroids
#     # for that utterance
#     # this is each speaker's utterances against his own centroid, but each
#     # comparison centroid has the current utterance removed
#     cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)
#
#     # now we get the cosine distance between each utterance and the other speakers' centroids
#     # to do so requires comparing each utterance to each centroid. To keep the
#     # operation fast, we vectorize by using matrices L (embeddings) and
#     # R (centroids) where L has each utterance repeated sequentially for all
#     # comparisons and R has the entire centroids frame repeated for each utterance
#     centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1))
#     embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1)
#     embeddings_expand = embeddings_expand.view(
#         embeddings_expand.shape[0] * embeddings_expand.shape[1],
#         embeddings_expand.shape[-1]
#     )
#     cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
#     cos_diff = cos_diff.view(
#         embeddings.size(0),
#         num_utterances,
#         centroids.size(0)
#     )
#     # assign the cosine distance for same speakers to the proper idx
#     same_idx = list(range(embeddings.size(0)))
#     cos_diff[same_idx, :, same_idx] = cos_same.view(embeddings.shape[0], num_utterances)
#     cos_diff = cos_diff + hp.small_err
#     return cos_diff
#
#
# def calc_loss(sim_matrix):
#     same_idx = list(range(sim_matrix.size(0)))
#
#     # eq. (6)
#     pos = sim_matrix[same_idx, :, same_idx]
#     neg = (torch.exp(sim_matrix).sum(dim=2) + hp.small_err).log_()
#     per_embedding_loss = -1 * (pos - neg)  # this is the loss as per eq. (6)
#
#     # eq. (10) final loss
#     # the final GE2E loss LG is
#     # the sum of all losses over the similarity matrix (1 ≤ j ≤ N, and 1 ≤ i ≤ M)
#     loss = per_embedding_loss.sum()
#     return loss, per_embedding_loss
#
#
# # code below will not be executed when this file is imported using from xyz import abc
# if __name__ == "__main__":
#     w = grad.Variable(torch.tensor(1.0))
#     b = grad.Variable(torch.tensor(0.0))
#
#     lst = [[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]
#     embeddings = torch.tensor(lst).to(torch.float).reshape(3, 2, 3)
#     centroids = get_centroids(embeddings)
#     cos_sim = get_cos_sim(embeddings, centroids)
#
#     # this is eq (5)
#     sim_matrix = w * cos_sim + b
#     loss, per_embedding_loss = calc_loss(sim_matrix)
#
#     print(loss)
#     print(per_embedding_loss)
