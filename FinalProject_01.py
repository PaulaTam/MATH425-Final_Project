#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from collections import Counter

filename = "./FinalProjectFiles/top250movies.txt"

l = []
filtered_ranks = {}

#open top250movies file, read  only, with utf-8 encoding
with open(filename, mode='r', encoding='utf-8') as f:
    for line in f:
        l.append(line.rstrip())
        #for every line in the file, strip trailing characters and append line to list

#this function is to split each line by the "/" delimiter
def slash_delimiter(string):
    split_string = string.split('/')
    return split_string

#this function is to apply the slash_delimiter function to a list
def delimited_list(movie_list):
    list_of_actors = []
    for item in movie_list:
        list_of_actors.append(slash_delimiter(item))
    return list_of_actors

#this funciton is to create a flatlist of all tuple combinations where actor1 appears before actor2
def create_actor_tuples(movie_list):
    list_of_tuples = []
    for movie_item in movie_list:
        movie_tuples = []
        for actor1, actor2 in itertools.combinations(movie_item, 2):
            #https://docs.python.org/3/library/itertools.html#itertools.combinations
            #we use itertools combinations to get all possible combinations in order
            #e.g. ABCD -> resulting combinations: AB AC AD BC BD CD
            movie_tuples.append(tuple((actor1, actor2)))
        list_of_tuples.append(movie_tuples)
        flatlist = [element for sublist in list_of_tuples for element in sublist]
    return flatlist

#this function returns all tuples with weights that are greater than 1
def highest_weights(dictionary):
    for k, v in dictionary.items():
        if (v > 1):
            filtered_ranks[k] = dictionary[k]
            
#this function compares any tuples with it's reversed tuple and keeps the tuple with the highest weight            
def remove_low_weights(dict1):
    output, seen = {}, set(filtered_ranks.keys())
    #initializing seen set with the tuples that are in filtered_ranks
    #set comparison as time complexity is near O(1)
    for key, value in dict1.items():
        k = tuple(key)
        if k not in seen and tuple(reversed(key)) not in seen:
            seen.add(k)
            output[key] = value
    return output

#this function merges 2 dictionaries together into 1
def merge_dicts(dict1, dict2):
    return (dict2.update(dict1))
        
dl = delimited_list(l)
rm = [l.pop(0) for l in dl] #list comprehension to remove movie titles from each line

at = create_actor_tuples(dl)
actor_ranks = dict(Counter(at)) #use of collections.Counter() to get weights of tuples, 880639 keys
highest_weights(actor_ranks)
ans = remove_low_weights(actor_ranks)
merge_dicts(ans, filtered_ranks) #filtered_ranks now 879433 keys

node_list = [(k[1], k[0], v) for k, v in filtered_ranks.items()]
#reverse order of the key tuple to point to higher billed actors
#we will import node_list to FinalProject_01_1.py in order to create network
