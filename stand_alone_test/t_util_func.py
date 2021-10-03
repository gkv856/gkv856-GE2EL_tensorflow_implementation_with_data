from gkv_implement.util import *


# note that util file is defined in the string constant
# testing if random_batch function works as expected.
# it is supposed to return a nparray of size - (150, 20, 40)
# 150 length of audio of 5 sample from 4 people of t_mels = 40
random_batch()

# testing the calc_similarity function
w = tf.constant([1], dtype=tf.float32)
b = tf.constant([0], dtype=tf.float32)
embedded = tf.constant([[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=tf.float32)
print(embedded.shape)

# def similarity(embedded, w, b, N=config["speaker_num"], M=config["utter_num"],
#                P=config["embed_dim"], centroid_with_m_utter=None):
sim_matrix = calc_similarity(embedded, w=w, b=b, N=3, M=2, P=3)
print(sim_matrix)


#testing the loss functions
loss1 = calculate_loss(sim_matrix, type="softmax", N=3, M=2)
loss2 = calculate_loss(sim_matrix, type="contrast", N=3, M=2)

print(loss1)
print(loss2)
