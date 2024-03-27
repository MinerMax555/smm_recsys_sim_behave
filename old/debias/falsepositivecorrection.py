import numpy as np


def FalsePositiveCorrection(scores, count, k_num, accepted_songs_last_round, top_k_last_round, all_accept_songs, all_top_K):
    count_index = count - 1
    n_users = scores.shape[0]

    # out of the pandas dataframes we recreate 2 np arrays that store the information of all of the accepted songs and top K of the shape [iteration x user x rank (of recommendation)]
    # Create a numpy array filled with np.nan
    user_item_matrix = np.zeros((1,n_users, k_num), dtype=np.int64)
    user_item_matrix[:,:,:] = -1

    # Fill the numpy array based on user-item interactions
    for i in range(n_users):
        tokens = accepted_songs_last_round[accepted_songs_last_round['user_id:token'] == i]["item_id:token"].values
        user_item_matrix[0,i,:len(tokens)] = tokens
        
    all_accept_songs = np.concatenate((all_accept_songs, user_item_matrix), axis=0)
    
    top_K_songs = top_k_last_round.reshape(1,n_users,-1)
    all_top_K = np.concatenate((all_top_K, top_K_songs), axis=0)
    

    # this is where we store the denominator of the formula of the paper. This concept of calculating got thaken out of the code that was at with the paper
    # youf find it on Simulation_FPC.py, line 81
    denominator = np.ones_like(scores)

    delta_kf = 1. / np.log2(np.arange(k_num) + 2)

    for u in range(n_users):
        for k in range(k_num):
            iid = int(all_top_K[count_index, u, k])
            if iid not in all_accept_songs[count_index, u]:
                denominator[u, iid] *= (1-delta_kf[k])
    
    predict_scaled = 1. - (1 - scores) / denominator
    
    return predict_scaled, all_accept_songs, all_top_K