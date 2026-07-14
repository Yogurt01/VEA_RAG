import sys
from typing import List
from collections import OrderedDict
from ast import literal_eval as make_tuple
import os
import json


def read_discourse_merge(file):
    """
    Return a list of list of tuples
    [ sent_i [ tuples (start, end (inclusive)
    :param file:
    :return: sent_disco_boundary: [ sent0 [(0,2),(3,9)], sent1 [(0,1),(2,10),(11,19),], ...]
            EDU_to_sent dict: {'1':0, '2': 0, ....} # discourse start from 1!!!
            edu_nsubj: dict determine if an EDU contains nsubj
    """
    with open(file, 'r') as fd:
        lines = fd.read().splitlines()
    lines = [l for l in lines if len(l) > 0]

    edu_dict = OrderedDict()

    last_line = lines[-1].split("\t")
    total_sent_num = int(last_line[0]) + 1
    sent_with_edu_spans = [[] for _ in range(total_sent_num)]
    for l in lines:
        items = l.split("\t")
        edu_id = str(items[-1])
        sent_id = int(items[0])
        nsubj = str(items[5])
        word_index_in_sent = int(items[1]) - 1

        if edu_id in edu_dict:
            _t = edu_dict[edu_id]
            _t[2] = word_index_in_sent
            edu_dict[edu_id][3] = edu_dict[edu_id][3] + [nsubj]
        else:
            edu_dict[edu_id] = [sent_id, word_index_in_sent, word_index_in_sent, [nsubj]]
    for k, v in edu_dict.items():
        sent_id, start_idx, end_idx, nsubj = v
        sent_with_edu_spans[sent_id].append((start_idx, end_idx))

    EDU_pool = {}
    edu_nsubj = {}
    for k, v in edu_dict.items():
        EDU_pool[k] = v[0]
        edu_nsubj[k] = v[3]

    return sent_with_edu_spans, EDU_pool, edu_nsubj


def determine_head(left_node, right_node):
    if left_node['type'] == 'n':
        return left_node['head']
    elif right_node['type'] == 'n':
        return right_node['head']
    else:
        return left_node['head']


def new_return_tree(d, EDU_pool, EDU_nsubj):
    root_node = d.popitem()  # root node
    root_node_sidx, root_node_eidx, root_node_node, root_node_rel = root_node[1]
    root_node_node = 'n' if root_node_node.startswith('N') else 's'
    if len(d) == 0:
        if root_node_sidx == 1 and 'punct' == EDU_nsubj['{}'.format(root_node_sidx)][0]:
            root_node_node = 's'

        return {
            'left': None,
            'right': None,
            's': root_node_sidx,
            'e': root_node_eidx,
            'type': root_node_node,
            'rel': root_node_rel,
            'head': root_node_sidx,
            'dep': [],
            'link': []
        }

    listed_items = list(d.items())

    right_child = listed_items[-1]
    r_sidx, r_eidx, r_node, r_rel = right_child[1]

    l_sidx = root_node_sidx
    l_eidx = r_sidx - 1

    keys = list(d.keys())
    cut_point = keys.index("{}_{}".format(l_sidx, l_eidx)) + 1
    left = listed_items[:cut_point]
    right = listed_items[cut_point:]

    left_node = new_return_tree(OrderedDict(left), EDU_pool, EDU_nsubj)
    right_node = new_return_tree(OrderedDict(right), EDU_pool, EDU_nsubj)

    my_head = determine_head(left_node, right_node)

    deps = left_node['dep'] + right_node['dep']

    if left_node['type'] == 's':
        if EDU_pool['{}'.format(left_node['s'])] == EDU_pool['{}'.format(my_head)] == EDU_pool[
            '{}'.format(left_node['e'])]:
            deps.append((left_node['head'], my_head))
    if right_node['type'] == 's':
        if EDU_pool['{}'.format(right_node['s'])] == EDU_pool['{}'.format(right_node['e'])] == EDU_pool[
            '{}'.format(my_head)]:
            deps.append((right_node['head'], my_head))
    elif right_node['type'] == 'n':
        if (EDU_pool['{}'.format(right_node['s'])]
            == EDU_pool['{}'.format(right_node['e'])]
            == EDU_pool['{}'.format(my_head)]) \
                and (right_node['head'] != my_head):
            nsubj_sign = False

            nsubj = EDU_nsubj['{}'.format(my_head)]
            if 'nsubj' in nsubj:
                nsubj_sign = True

            if nsubj_sign == False:
                deps.append((right_node['head'], my_head))

    links = left_node['link'] + right_node['link']
    links.append((left_node['head'], right_node['head'], root_node_rel))

    return {
        'left': left_node,
        'right': right_node,
        's': root_node_sidx,
        'e': root_node_eidx,
        'type': root_node_node,
        'rel': root_node_rel,
        'head': my_head,
        'dep': deps,
        'link': links
    }


def new_read_bracket(bracket_file, EDU_pool, EDU_nsubj):
    with open(bracket_file, 'r') as fd:
        lines = fd.read().splitlines()
    treebank = [None for _ in range(1000)]
    d = OrderedDict()
    max_num = -1

    for l in lines:
        tup = make_tuple(l)
        index, node, relation = tup
        sidx, eidx = index
        d['{}_{}'.format(sidx, eidx)] = [sidx, eidx, node, relation]
        max_num = max(max_num, eidx)
        if sidx == eidx:
            treebank[sidx] = [node, relation, None]
    d['{}_{}'.format(1, max_num)] = [1, max_num, 'Nucleus', 'ROOT']
    x = new_return_tree(d, EDU_pool, EDU_nsubj)

    return x['dep'], x['link']


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Please provide the merge_folder, brackets_folder, and output_file as command-line arguments.")
        sys.exit(1)

    merge_folder = sys.argv[1]
    brackets_folder = sys.argv[2]
    output_file = sys.argv[3]

    result_dict = {}

    merge_files = os.listdir(merge_folder)
    for merge_file in merge_files:
        merge_file_path = os.path.join(merge_folder, merge_file)
        brackets_file = merge_file.replace('.merge', '.brackets')
        brackets_file_path = os.path.join(brackets_folder, brackets_file)

        if os.path.isfile(brackets_file_path):
            sent_with_edu, edu_pool, edu_nsubj = read_discourse_merge(merge_file_path)
            dep, link = new_read_bracket(brackets_file_path, edu_pool, edu_nsubj)
            result_dict[merge_file] = {'dep': dep, 'link': link}

    with open(output_file, 'w') as f:
        json.dump(result_dict, f)
