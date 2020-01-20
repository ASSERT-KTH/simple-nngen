# encoding=utf-8

import os
import time
import fire
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu

def load_data(path):
    """load lines from a file"""
    with open(path, 'r') as f:
        lines = f.read().split('\n')[0:-1]
    lines = [l.strip() for l in lines]
    return lines

def find_mixed_nn(simi, diffs, test_diff, bleu_thre :int =5) -> int:
    """Find the nearest neighbor using cosine simialrity and bleu score"""
    candidates = simi.argsort()[-bleu_thre:][::-1]
    max_score = 0
    max_idx = -1
    cnt = 0
    for j in candidates:
        if simi[j] < 0:
            return max_idx
        score = sentence_bleu([diffs[j].split()], test_diff.split())
        if score > max_score:
            max_score = score
            max_idx = j
        cnt = cnt + 1
    return max_idx

def nngen(train_diffs :List[str], train_msgs :List[str], test_diffs :List[str],
          train_repos :List[str], test_repos :List[str],
          mode :"'exc': excludes test commit repo, 'inc': only includes test commit repo" ='def',
    bleu_thre :"how many candidates to consider before calculating bleu_score" =5) -> List[str]:
    """NNGen
    NOTE: currently, we haven't optmize for large dataset. You may need to split the 
    large training set into several chunks and then calculate the similarities between
    train set and test set to speed up the algorithm. You may also leverage GPU through
    pytorch or other libraries.
    """
    counter = CountVectorizer()
    train_matrix = counter.fit_transform(train_diffs)
    # print(len(counter.vocabulary_))
    test_matrix = counter.transform(test_diffs)
    similarities = cosine_similarity(test_matrix, train_matrix)
    test_msgs = []
    test_selected_repos = []
    for idx, test_simi in enumerate(similarities):
        if mode == 'exc':
            for i in range(test_simi.size):
                if train_repos[i] == test_repos[idx] or train_repos[i] == 'UNKNOWN' or test_repos[idx] == 'UNKNOWN':
                    test_simi[i] = -1
        if mode == 'inc':
            for i in range(test_simi.size):
                if train_repos[i] != test_repos[idx] or train_repos[i] == 'UNKNOWN' or test_repos[idx] == 'UNKNOWN':
                    test_simi[i] = -1
        if (idx + 1) % 100 == 0:
            print(idx+1)
        max_idx = find_mixed_nn(test_simi, train_diffs, test_diffs[idx], bleu_thre)
        if max_idx == -1:
            test_msgs.append("UNKONWN")
            test_selected_repos.append("UNKNOWN")
        else:
            test_msgs.append(train_msgs[max_idx])
            test_selected_repos.append(train_repos[max_idx])
    return (test_msgs, test_selected_repos)

def main(train_diff_file :str, train_msg_file :str, train_repos_file :str, 
         test_diff_file :str, test_repos_file :str):
    """Run NNGen with default given dataset using default setting"""
    start_time = time.time()
    test_dirname = os.path.dirname(test_diff_file)
    test_basename = os.path.basename(test_diff_file)
    
    train_diffs = load_data(train_diff_file)
    train_msgs = load_data(train_msg_file)
    train_repos = load_data(train_repos_file)
    test_diffs = load_data(test_diff_file)
    test_repos = load_data(test_repos_file)
    
    inc_full_out_repos_file =  "./inc_full_nngen." + test_basename.replace('.diff', '.repos')
    inc_out_repos_file =  "./inc_nngen." + test_basename.replace('.diff', '.repos')
    exc_out_repos_file =  "./exc_nngen." + test_basename.replace('.diff', '.repos')
    out_repos_file =  "./nngen." + test_basename.replace('.diff', '.repos')
    
    inc_full_out_file =  "./inc_full_nngen." + test_basename.replace('.diff', '.msg')
    inc_full_out_res = nngen(train_diffs, train_msgs, test_diffs, train_repos, test_repos, 'inc', len(train_diffs))
    with open(inc_full_out_file, 'w') as out_f:
        out_f.write("\n".join(inc_full_out_res[0]) + "\n")
    with open(inc_full_out_repos_file, 'w') as out_f:
        out_f.write("\n".join(inc_full_out_res[1]) + "\n")
        
    inc_out_file =  "./inc_nngen." + test_basename.replace('.diff', '.msg')
    inc_out_res = nngen(train_diffs, train_msgs, test_diffs, train_repos, test_repos, 'inc')
    with open(inc_out_file, 'w') as out_f:
        out_f.write("\n".join(inc_out_res[0]) + "\n")
    with open(inc_out_repos_file, 'w') as out_f:
        out_f.write("\n".join(inc_out_res[1]) + "\n")
    
    exc_out_res = nngen(train_diffs, train_msgs, test_diffs, train_repos, test_repos, 'exc')
    exc_out_file =  "./exc_nngen." + test_basename.replace('.diff', '.msg')
    with open(exc_out_file, 'w') as out_f:
        out_f.write("\n".join(exc_out_res[0]) + "\n")
    with open(exc_out_repos_file, 'w') as out_f:
        out_f.write("\n".join(exc_out_res[1]) + "\n")
    
    out_file =  "./nngen." + test_basename.replace('.diff', '.msg')
    out_res = nngen(train_diffs, train_msgs, test_diffs, train_repos, test_repos)
    with open(out_file, 'w') as out_f:
        out_f.write("\n".join(out_res[0]) + "\n")
    with open(out_repos_file, 'w') as out_f:
        out_f.write("\n".join(out_res[1]) + "\n")
    
    
    time_cost = time.time() -start_time
    print("Done, cost {}s".format(time_cost))



#     test_repos = load_data("./files/data/test.projectIds")
#     unknown_test_repos = 0
#     for i in range(len(test_repos)):
#         if test_repos[i] == 'UNKNOWN':
#             unknown_test_repos = unknown_test_repos + 1
#     print ("Number of known test repos: " + str(len(test_repos) - unknown_test_repos))
#     
#     nngen_repos = load_data("./files/data/nngen.test.repos")
#     cnt = 0
#     for i in range(len(test_repos)):
#         if nngen_repos[i] == test_repos[i] and nngen_repos[i] != 'UNKNOWN':
#             cnt = cnt + 1
#     print ("Number of known nngen repos similar to the test repos: " + str(cnt))
#     
#     inc_repos = load_data("./files/data/inc_nngen.test.repos")
#     cnt = 0
#     for i in range(len(test_repos)):
#         if inc_repos[i] == test_repos[i] and inc_repos[i] != 'UNKNOWN':
#             cnt = cnt + 1
#     print ("Number of known inc_nngen repos similar to the test repos: " + str(cnt))
#     
#     exc_repos = load_data("./files/data/exc_nngen.test.repos")
#     cnt = 0
#     for i in range(len(test_repos)):
#         if exc_repos[i] == test_repos[i] and exc_repos[i] != 'UNKNOWN':
#             cnt = cnt + 1
#     print ("Number of known exc_nngen repos similar to the test repos: " + str(cnt))

def compute_bleu_scores(algorithm :str):
    algo_repos = load_data("./files/data/" + algorithm + ".test.repos")
    test_repos = load_data("./files/data/test.projectIds")
    algo_msgs = load_data("./files/data/" + algorithm + ".test.msg")
    test_msgs = load_data("./files/data/test.msg")
    same_repo_cnt = 0
    same_bleu_sum = 0
    different_repo_cnt = 0
    different_bleu_sum = 0
    unknown_repo_cnt = 0
    unknown_bleu_sum = 0
    all_repo_cnt = 0
    all_bleu_sum = 0
    for i in range(len(algo_msgs)):
        bleu = sentence_bleu([test_msgs[i].split()], algo_msgs[i].split())
        all_repo_cnt = all_repo_cnt + 1
        all_bleu_sum = all_bleu_sum + bleu
        if test_repos[i] == 'UNKNOWN':
            unknown_repo_cnt = unknown_repo_cnt + 1
            unknown_bleu_sum = unknown_bleu_sum + bleu
        elif algo_repos[i] == test_repos[i]:
            same_repo_cnt = same_repo_cnt + 1
            same_bleu_sum = same_bleu_sum + bleu
        else:
            different_repo_cnt = different_repo_cnt + 1
            different_bleu_sum = different_bleu_sum + bleu
    print("avg bleu for messages selected from the same,different,all,unknown repos: " 
          + str(same_bleu_sum / same_repo_cnt) + "," 
          + str(different_bleu_sum / different_repo_cnt) + ","
          + str(all_bleu_sum / all_repo_cnt) + "," 
          + str(unknown_bleu_sum / unknown_repo_cnt))

if __name__ == "__main__":
    fire.Fire({
        'main':main 
    })
