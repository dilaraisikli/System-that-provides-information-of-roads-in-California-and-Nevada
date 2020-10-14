import pandas as pd
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import gmplot
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from statistics import mean
import numpy as np
import math
import random
import folium
from IPython.display import IFrame
from sklearn.manifold import TSNE
import itertools
import functionality1
import functionality2
import functionality3

import importlib
importlib.reload(functionality1)


def map_choice(G):
    print("Choose the functionality. Enter: \n -'1' to find the neighbors! \n -'2' to find the smartest network!, \n -'3' to find the shortest ordered route!, \n -'4' to find the shortest route! \n Enter 'esc' to exit", end = "")
    enter = input()
    if enter == '1':

        node=input("Choose a node ")
        measure =input('Choose a distance')
        threshold=input('Choose a threshold ')
        coordinates, distance, travel_time = functionality1.get_data()
        #Creating the graph
        l1=distance['Id_Node1']
        l2=distance['Id_Node2']
        graph = {}
        for i, j in zip(l1, l2):
            graph.setdefault(i, []).append(j)
        
        functionality1.funct1(node,measure,threshold, graph, coordinates, distance, travel_time)
        return

        
    elif enter == '2':
        network, distances, time, coordinates, weighted_network, df = functionality2.get_data()
        nodes=list(map(int,input("Choose a set of nodes (just enter the nodes id with spaces between them) ").split()))
        measure=input("Choose a distance type " )
        path = functionality2.find_smartest_path(nodes, measure, network, distances, time, coordinates, weighted_network)
        functionality2.draw_graph(path,G, coordinates, df)
        functionality2.gmaps(coordinates, path)
        res=IFrame('mapCC.html', width=700, height=600)
        res
        
        
    elif enter == '3':
        coordinates, df= functionality3.get_coordinates()
        nodes=list(map(int,input("Choose a set of nodes (just enter the nodes id with spaces between them) ").split()))
        measure=input("Choose a distance type " )
        path = functionality3.find_shortest_path(nodes, measure)
        functionality3.draw_graph(path,G, coordinates, df)
        functionality3.gmaps(coordinates, path)
        res=IFrame('mapTT.html', width=700, height=600)
        res

    elif enter == '4':
        coordinates, df= functionality3.get_coordinates()
        nodes=list(map(int,input("Choose a set of nodes (just enter the nodes id with spaces between them) ").split()))
        measure=input("Choose a distance type " )
        path = functionality4.shorthest_unordered_path(nodes, measure)
        functionality4.draw_graph(path,G, coordinates, df)
        functionality4.gmaps(coordinates, path)
        res=IFrame('mapTT.html', width=700, height=600)
        res

    elif enter == 'esc':
        return
    else:
        print("Please, enter again one of those: '1', '2', '3','4' or 'esc'.", '\n')
        return map_choice()




G = functionality2.get_graph()
map_choice(G)