import csv
import json
import numpy as np
import pdb
from tqdm import tqdm

def check_k(queries):
    return len(queries[0]['mentions'][0]['candidates'])

def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    queries = data['queries']
    k = check_k(queries)

    errors = [] #['truth', 'prediction']

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i+1] # to get acc@(i+1)
                mention_hit += np.any([candidate['label'] for candidate in candidates])
            
                if not np.any([candidate['label'] for candidate in candidates]):
                    errors.append([mention['golden_cui'], mention['mention'], mention['candidates'][0]['cui'], mention['candidates'][0]['name']])
            # When all mentions in a query are predicted correctly,
            # we consider it as a hit 
            if mention_hit == len(mentions):
                hit +=1
        
        data['acc{}'.format(i+1)] = hit/len(queries)

    return data, errors

def check_label(predicted_cui, golden_cui):
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|"))))>0)

def predict_topk(biosyn, eval_dictionary, eval_queries, topk, score_mode='hybrid'):
    """
    Parameters
    ----------
    score_mode : str
        hybrid, dense, sparse
    """
    encoder = biosyn.get_dense_encoder()
    tokenizer = biosyn.get_dense_tokenizer()
    sparse_encoder = biosyn.get_sparse_encoder()
    sparse_weight = biosyn.get_sparse_weight().item() # must be scalar value
    sent_weight = biosyn.get_sent_weight().item()
    print(sent_weight)
    
    # embed dictionary
    dict_sparse_embeds = biosyn.embed_sparse(names=eval_dictionary[:,0], show_progress=True)
    dict_dense_embeds = biosyn.embed_dense(names=eval_dictionary[:,0], show_progress=True)
    dict_sent_embeds = biosyn.embed_sent(names=eval_dictionary[:,0], show_progress=True)
    
    queries = []
    for eval_query in tqdm(eval_queries, total=len(eval_queries)):
        mentions = eval_query[0].replace("+","|").split("|")
        golden_cui = eval_query[1].replace("+","|")
        
        dict_mentions = []
        for mention in mentions:
            mention_sparse_embeds = biosyn.embed_sparse(names=np.array([mention]))
            mention_dense_embeds = biosyn.embed_dense(names=np.array([mention]))
            mention_sent_embeds = biosyn.embed_sent(names=np.array([mention]))
            
            # get score matrix
            sparse_score_matrix = biosyn.get_score_matrix(
                query_embeds=mention_sparse_embeds, 
                dict_embeds=dict_sparse_embeds
            )
            dense_score_matrix = biosyn.get_score_matrix(
                query_embeds=mention_dense_embeds, 
                dict_embeds=dict_dense_embeds
            )
            sent_score_matrix = biosyn.get_score_matrix(
                query_embeds=mention_sent_embeds, 
                dict_embeds=dict_sent_embeds
            )
            if score_mode == 'hybrid':
                score_matrix = sparse_weight * sparse_score_matrix + sent_weight * sent_score_matrix + dense_score_matrix
            elif score_mode == 'dense':
                score_matrix = dense_score_matrix
            elif score_mode == 'sparse':
                score_matrix = sparse_score_matrix
            elif score_mode == 'no-sent':
                score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
            else:
                raise NotImplementedError()

            candidate_idxs = biosyn.retrieve_candidate(
                score_matrix = score_matrix, 
                topk = topk
            )
            np_candidates = eval_dictionary[candidate_idxs].squeeze()
            dict_candidates = []
            for np_candidate in np_candidates:
                dict_candidates.append({
                    'name':np_candidate[0],
                    'cui':np_candidate[1],
                    'label':check_label(np_candidate[1],golden_cui)
                })
            dict_mentions.append({
                'mention':mention,
                'golden_cui':golden_cui, # golden_cui can be composite cui
                'candidates':dict_candidates
            })
        queries.append({
            'mentions':dict_mentions
        })
    
    result = {
        'queries':queries
    }

    return result

def getLCAStatistics(disease_dict, tree_map, errors):
    n_errors = 0
    n_child_guessed = 0
    n_ancestor_guessed = 0
    distances = []
    family_errors = []
    lcas = []
    vs = []
    for e in errors:
        #Within family errors
        if e[0] != e[2]:
            n_errors += 1
            if e[0] in tree_map[e[2]]:
                if tree_map[e[2]][e[0]] > 0:
                    n_ancestor_guessed += 1
                else:
                    n_child_guessed += 1
                distances.append(abs(tree_map[e[2]][e[0]]))
                family_errors.append((e[0], e[2], tree_map[e[2]][e[0]]))
            
        #LCA distances
        family_gt = tree_map[e[0]]
        print(family_gt)
        family_pred = tree_map[e[2]]
        print(family_pred)
        overlap = family_gt.keys() & family_pred.keys()
        lca = -float('inf')
        dpred = -float('inf')
        dgt = -float('inf')
        ancestor = 'C'
        print(overlap)
        for k in overlap:
            if family_gt[k] + family_pred[k] < 0:
                if family_gt[k] + family_pred[k] > lca:
                    lca = family_gt[k] + family_pred[k]
                    dpred = family_pred[k]
                    dgt = family_gt[k]
                    ancestor = k
        v = list(e)
        v.append(ancestor)
        v.append(dgt)
        v.append(dpred)
        if e[0] in family_pred:
            v.append(1)
        else:
            v.append(0)
        vs.append(v)
        lcas.append(lca)
        if lca == -float('inf'):
            print(v)
        print()
    
    print("Precent of errors within family: %.03f"%(len(distances) / n_errors))
    print("Average distance within family: %.03f"%(sum(distances) / len(distances)))
    print("Percent child guessed: %.03f"%(n_child_guessed / len(distances)))
    print("Percent ancestor guessed: %.03f"%(n_ancestor_guessed / len(distances)))
    dists = np.asarray(distances)
    print("Percentage of errors which are a child or parent: %.04f"%(len(dists[dists == 1]) / len(dists)))
    
    print("Average LCA distance: %.04f"%(abs(sum(lcas) / len(lcas))))


def evaluate(biosyn, eval_dictionary, eval_queries, topk, score_mode='hybrid'):
    """
    predict topk and evaluate accuracy
    
    Parameters
    ----------
    biosyn : BioSyn
        trained biosyn model
    eval_dictionary : str
        dictionary to evaluate
    eval_queries : str
        queries to evaluate
    topk : int
        the number of topk predictions
    score_mode : str
        hybrid, dense, sparse

    Returns
    -------
    result : dict
        accuracy and candidates
    """
    result = predict_topk(biosyn,eval_dictionary,eval_queries, topk, score_mode)
    result, errors = evaluate_topk_acc(result)
    
    return result, errors