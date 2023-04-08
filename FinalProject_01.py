#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:54:55 2023

@author: PaulaTam
"""

filename = "./FinalProjectFiles/top250movies.txt"

l = []

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
        
dl = delimited_list(l)
#print(dl[0])