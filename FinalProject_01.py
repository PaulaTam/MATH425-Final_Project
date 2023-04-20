#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:54:55 2023

@author: PaulaTam
"""

import itertools
from collections import defaultdict, Counter

filename = "./FinalProjectFiles/top250movies.txt"

l = []
filtered_ranks = {}
#nested_dict = defaultdict(dict)

#open top250movies file, read  only, with utf-8 encoding
#put ids into set -> list -> can get index --> can check pagerank file as resource
with open(filename, mode='r', encoding='utf-8') as f:
    #print(f.read()) #read() to check every line in file
    for line in f:
        l.append(line.rstrip())
   
#this is to see if the list actually worked or not
#test = [l[0]]        
#print(test)

def slash_delimiter(string):
    split_string = string.split('/')
    return split_string
    
def delimited_list(movie_list):
    list_of_actors = []
    for item in movie_list:
        list_of_actors.append(slash_delimiter(item))
    return list_of_actors

def create_actor_tuples(movie_list):
    list_of_tuples = []
    for movie_item in movie_list:
        movie_tuples = []
        for actor1, actor2 in itertools.combinations(movie_item, 2):
            movie_tuples.append(tuple((actor1, actor2)))
        list_of_tuples.append(movie_tuples)
        flatlist = [element for sublist in list_of_tuples for element in sublist]
    return flatlist
            
def highest_weights(dictionary):
    for k, v in dictionary.items():
        if (v > 1):
            filtered_ranks[k] = dictionary[k]
            #filtered_ranks is a dictionary with all weights > 1
                        
def remove_low_weights(dict1):
    output, seen = {}, set(filtered_ranks.keys())
    for key, value in dict1.items():
        k = tuple(key)
        if k not in seen and tuple(reversed(key)) not in seen:
            seen.add(k)
            output[key] = value
    return output

def merge_dicts(dict1, dict2):
    return (dict2.update(dict1))
        
dl = delimited_list(l)
rm = [l.pop(0) for l in dl] #list comprehension for all the movie titles

at = create_actor_tuples(dl)
actor_ranks = dict(Counter(at))
highest_weights(actor_ranks)
ans = remove_low_weights(actor_ranks)
merge_dicts(ans, filtered_ranks) #filtered_ranks now 879433

node_list = [(k[1], k[0], v) for k, v in filtered_ranks.items()]
#reverse order of the key to point to higher billed actors
#print(list(filtered_ranks.keys()))

#test = set(filtered_ranks.keys())
#print(actor_ranks) #commented this part out since it's 880639 entries in dict()
