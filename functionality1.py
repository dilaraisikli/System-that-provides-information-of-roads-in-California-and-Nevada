import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import folium
from IPython.display import IFrame
from sklearn.manifold import TSNE
import itertools
#Importing the data

def get_data():

    #Coordinates data
    coordinates= open("USA-road-d.CAL.co")
    coordinates=pd.DataFrame(coordinates)
    coordinates=coordinates.iloc[7:]
    coordinates.columns=['a']
    coordinates=coordinates.a.str.split(expand=True)
    del coordinates[0]
    coordinates.columns= ['Id_Node','Latitude','Longitude']
    coordinates= coordinates.set_index('Id_Node')


    #Distance data
    distance= open("USA-road-d.CAL.gr")
    distance=pd.DataFrame(distance)
    distance=distance.iloc[7:]
    distance.columns=['a']
    distance=distance.a.str.split(expand=True)
    del distance[0]
    distance.columns= ['Id_Node1','Id_Node2','Distance']
    distance= distance.reset_index()
    del distance['index']


    #Travel_time
    travel_time= open("USA-road-t.CAL.gr")
    travel_time=pd.DataFrame(travel_time)
    travel_time=travel_time.iloc[7:]
    travel_time.columns=['a']
    travel_time=travel_time.a.str.split(expand=True)
    del travel_time[0]
    travel_time.columns= ['Id_Node1','Id_Node2','Travel time']
    travel_time= travel_time.reset_index()
    del travel_time['index']
    return coordinates, distance, travel_time
    
    
    
#The visualization function
def viz1(node,neighbors,paths,measure,coordinates, distance, travel_time):
    #The coordinates of the neighbors
    long=[int(coordinates['Longitude'][k-1])*(10**-6) for k in neighbors]
    lat=[int(coordinates['Latitude'][k-1])*(10**-6) for k in neighbors]
    # We create empty map zoomed in on the principal node
    m = folium.Map(location=(int(coordinates['Longitude'][int(node)-1])*(10**-6),int(coordinates['Latitude'][node-1])*10**-6), zoom_start=20)
    #We put a red marker for the principal node 
    folium.Marker(location=(int(coordinates['Longitude'][int(node)-1])*(10**-6),int(coordinates['Latitude'][node-1])*10**-6), popup='Node '+str(node),icon=folium.Icon(color='red', icon='info-sign')).add_to(m)
    #We put green markers for the neighbors nodes
    for i in range(len(neighbors)):
        folium.Marker([long[i], lat[i]], popup='Node '+str(neighbors[i]),icon=folium.Icon(color='green', icon='info-sign')).add_to(m)
    #Based on the choice of the measure , we chose the color of the edges (the streets)
    if measure=='n':
        col='blue'
    elif measure=='d':
        col='black'
    elif measure =='t':
        col='purple'
    points=[[] for i in range(len(paths))]
    #We plot the lines that represent the edges 
    for i in range(len(paths)):
        for j in range(len(paths[i])):
            pointlong=int(coordinates['Longitude'][int(paths[i][j])-1])*(10**-6)
            pointlat=int(coordinates['Latitude'][int(paths[i][j])-1])*(10**-6)
            points[i].append(tuple([pointlong, pointlat]))
        folium.PolyLine(points[i], color=col, weight=4, opacity=1).add_to(m)
    #The result
    m.save('map1.html')
    res=IFrame(src='map1.html', width=700, height=600)
    return(res)
  
        
        
#The d(x,y) function
def d(x,y, distance):
    dist=int(distance.loc[(distance['Id_Node1']==str(x)) & (distance['Id_Node2']==str(y)),'Distance'])
    return(dist)
#The t(x,y) function
def t(x,y, travel_time, distance):
    dist=int(travel_time.loc[(distance['Id_Node1']==str(x)) & (travel_time['Id_Node2']==str(y)),'Travel time'])
    return(dist)
#We also create the paths because we will need them for the visualization 
#The paths_n, paths_d and paths_t functions are recursive functions that save the neighbors of the node when the distance
#is still inferior than the threshold AND the neighbor node wasn't already visited. The output of these functions is a list of all the paths 
#starting by the principal node and in which the threshold is not exceeded
def paths_n(graph,node,threshold,visited = None): 
    if visited == None: 
        visited = set([node]) 
    else: 
        visited.add(node) 
    if threshold==0: 
        return [[node]] 
    paths = [[node]+path for neighbor in graph.get(node) if neighbor not in visited for path in paths_n(graph,neighbor,threshold-1,visited)] 
    visited.remove(node) 
    return paths 

def paths_d(graph,distance,node,threshold,visited = None): 
    if visited == None: 
        visited = set([node]) 
    else: 
        visited.add(node) 
    if threshold<min([d(node,k, distance) for k in graph.get(node)]): 
        return [[node]] 
    paths = [[node]+path for neighbor in graph.get(node) if neighbor not in visited for path in paths_d(graph,distance,neighbor,threshold-d(neighbor,node),visited)] 
    visited.remove(node) 
    return paths 

def paths_t(graph,travel_time, distance,node,threshold,visited = None): 
    if visited == None: 
        visited = set([node]) 
    else: 
        visited.add(node) 
    if threshold<min([t(node,k, travel_time, distance) for k in graph.get(node)]): 
        return [[node]] 
    paths = [[node]+path for neighbor in graph.get(node) if neighbor not in visited for path in paths_t(graph,travel_time, distance, neighbor,threshold-d(neighbor,node, distance),visited)] 
    visited.remove(node) 
    return paths 

#The functionality 1
def funct1(node,measure,threshold, graph, coordinates, distance, travel_time):
    #Based on the choice of the measure , we execute our code
    if measure=='n':
        paths=paths_n(graph,str(node),threshold,visited = None)
        #Based on our paths, we create neighbors list that will contain the nodes mentioned in the paths without repetition  
        neighbors=[]
        for i in range(len(paths)):
            neighbors=set(itertools.chain(neighbors, paths[i]))
        neighbors.remove(str(node))
        neighbors=sorted([int(i) for i in neighbors])
        viz1(node,neighbors,paths,measure)
        res='The neighbors of the node '+str(node)+' are : '+str(neighbors)
        print(res)
    elif measure=='d':
        #Based on our paths, we create neighbors list that will contain the nodes mentioned in the paths without repetition
        paths=paths_d(graph,distance,str(node),threshold,visited = None, )
        neighbors=[]
        for i in range(len(paths)):
            neighbors=set(itertools.chain(neighbors, paths[i]))
        neighbors.remove(str(node))
        neighbors=sorted([int(i) for i in neighbors])
        viz1(node,neighbors,paths,measure)
        res='The neighbors of the node '+str(node)+' are : '+str(neighbors)
        print(res)
    elif measure=='t':
        #Based on our paths, we create neighbors list that will contain the nodes mentioned in the paths without repetition
        paths=paths_t(graph,travel_time,distance, str(node),threshold,visited = None)
        neighbors=[]
        for i in range(len(paths)):
            neighbors=set(itertools.chain(neighbors, paths[i]))
        neighbors.remove(str(node))
        neighbors=sorted([int(i) for i in neighbors])
        viz1(node,neighbors,paths,measure, coordinates,distance, travel_time)
        res='The neighbors of the node '+str(node)+' are : '+str(neighbors)
        print(res)
    
    else:
        print('The type of distance is unkwown')
        
    return viz1(node,neighbors,paths,measure,coordinates, distance, travel_time)

