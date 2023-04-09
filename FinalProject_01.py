#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:54:55 2023

@author: PaulaTam
"""

import itertools

filename = "./FinalProjectFiles/top250movies.txt"

l = []
actor_ranks = dict()

#open top250movies file, read only, with utf-8 encoding
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

def create_dict_of_tuples(actor_tuples):
    for tuple_item in actor_tuples:
        if tuple_item in actor_ranks:
            actor_ranks[tuple_item] += 1
        else:
            actor_ranks[tuple_item] = 1
    
        
dl = delimited_list(l)
rm = [l.pop(0) for l in dl] #list comprehension for all the movie titles

at = create_actor_tuples(dl)
create_dict_of_tuples(at)
#print(actor_ranks) #commented this part out since it's 880639 entries in dict()
