from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import gmplot
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import functionality2
from statistics import mean

# -------------------------------------------------BASIC FUNCTIONS-----------------------------------------------
def checkIfDuplicates(list):
    ''' Check if given list contains any duplicates '''    
    for elem in list:
        if list.count(elem) > 1:
            return True
    return False

def removeDuplicates(list):
    ''' Remove from a given list any duplicates ''' 
    list.reverse()
    for elem in list:
        while list.count(elem) > 1:
            list.remove(elem)
    list.reverse()
    return list

def contained(list1, list2):
    for el in list1:
        if el in list2:
            continue
        else:
            return False
    return True

# return the index of a given node in the list of neightbours of another node
def look_for_it(element, list_of_list):
    for i in range(len(list_of_list)):
        if list_of_list[i][0]== element:
            return i

def look_for_it_network(element, list_of_list):
    for i in range(len(list_of_list)):
        if list_of_list[i]== element:
            return i
        
def find_weight(path, distances, time, small_coords, measure):
    if measure== "d":
        d=0
        for i in range(1,len(path)):
            d+=distances[path[i-1]][path[i]]
        return d
        
    elif measure== "t":
        d=0
        for i in range(1,len(path)):
            d+=time[path[i-1]][path[i]]
        return d
        
    elif measure== "n":
        d=0
        for i in range(1,len(path)):
            d+=small_network[path[i-1]][path[i]]
        return d
    

def nodes_non_in_line(paths, nodes):
    ot=[]
    for i in range(len(paths)):
        for j in range(len(paths)):
            if i!= j:
                if (paths[i][-1]== paths[j][0]) & (paths[i][0]!= paths[j][-1]):
                    ot.append(paths[i][:-1]+paths[j])
                    
    new=[]
    for path in ot:
        if contained(nodes, path):
            new.append(path)
    return new
        
#------------------------------------------------ GET DATA -------------------------------------------    
# network, distances, time, coordinates, weighted_network = functionality2.get_data()
def get_data():
    n_small = 75000
    distances = defaultdict(dict)
    weighted_network=defaultdict(dict)
    network = defaultdict(list)
    with open('USA-road-d.CAL.gr', 'r') as f:
        for _ in range(n_small):
            if f.readline()[0] == 'a': 
                n1, n2, d= list(map(int, f.readline()[2::].split()))
                distances[n1][n2] = d
                distances[n2][n1]= d
                weighted_network[n1][n2] = 1
                weighted_network[n2][n1]= 1
                network[n1].append(n2)
                network[n2].append(n1)
    
    time = defaultdict(dict)
    with open('USA-road-t.CAL.gr', 'r') as f:
        for _ in range(n_small):
            if f.readline()[0]=='a':
                n1, n2,t = list(map(int, f.readline()[2::].split()))
                time[n1][n2]= t
                time[n2][n1]=t
    coordinates = defaultdict(list)
    with open('USA-road-d.CAL.co', 'r') as f:
        for line in tqdm(f):
            if line[0]=='v':
                n, lat, long= list(map(int, line[2::].split()))
                coordinates[n]=[lat, long]
                
    df = pd.DataFrame(coordinates).T
    df.rename(columns={0:'latitude', 1:'longitude'}, inplace = True)
    return network, distances, time, coordinates, weighted_network, df
    
    
# ---------------------------------------------------IF PATH EXISTS-----------------------------------
# visits all the nodes of a graph (connected component) starting from a given node using BFS
def bfs(network, start):
    # keep track of all visited nodes
    explored = []
    # keep track of nodes to be checked
    queue = [start]

    levels = {}         # this dict keeps track of levels
    levels[start]= 0    # depth of start node is 0

    visited= [start]     # to avoid inserting the same node twice into the queue

    # keep looping until there are nodes still to be checked
    while queue:
       # pop shallowest node (first node) from queue
        node = queue.pop(0)
        explored.append(node)
        neighbours = network[node]

        # add neighbours of node to queue
        for neighbour in neighbours:
            if neighbour not in visited:
                queue.append(neighbour)
                visited.append(neighbour)

                levels[neighbour]= levels[node]+1
    return explored

def has_Path(nodes, network):
    for el in nodes:
        start= nodes[0]
        ans = bfs(network,start)
        if contained(nodes,ans):
            return True
        else:
            return False
        
#--------------------------------------------------- DIJSTRA -----------------------------------------------------

def dijkstra(graph,start,target, visited=[],distances={},predecessors={}):    
    if start == target:
        # We build the shortest path and display it
        path=[]
        pred=target
        while pred != None:
            path.append(pred)
            pred=predecessors.get(pred,None)
        # reverses the array, to display the path nicely
        return path
    else :
        # if it is the initial  run, initializes the cost
        if not visited: 
            distances[start]=0
            print
        # visit the neighbors
        for neighbor in graph[start] :
            if neighbor not in visited:
                new_distance = distances[start] + graph[start][neighbor]
                if new_distance < distances.get(neighbor,float('inf')):
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = start
        # mark as visited
        visited.append(start)
        # now that all neighbors have been visited: recurse                         
        # select the non visited node with lowest distance 'x'
        # run Dijskstra with src='x'
        unvisited={}
        for k in graph:
            if k not in visited:
                unvisited[k] = distances.get(k,float('inf'))        
        x=min(unvisited, key=unvisited.get)
        return dijkstra(graph,x,target,visited,distances,predecessors)
    
    
    
# ------------------------------------------------------ FIND THE MINUMU PATH ACCORDING TO A TYPE OF DISTANCE------------------------
def physical_distance(all_paths, distances):
    how_far = []
    for path in all_paths:
        d=0
        for i in range(1,len(path)):
            d+=distances[path[i-1]][path[i]]
        how_far.append(d)
    idx= how_far.index(min(how_far))
    shortest_path= all_paths[idx]
    return shortest_path, min(how_far)

def time_distance(all_paths, time):
    how_far = []
    for path in all_paths:
        d=0
        for i in range(1,len(path)):
            d+=time[path[i-1]][path[i]]
        how_far.append(d)
    idx= how_far.index(min(how_far))
    shortest_path= all_paths[idx]
    return shortest_path, min(how_far)

def network_distance(all_paths, network):
    how_far = []
    for path in all_paths:
        d=0
        for i in range(1,len(path)):
            indx= look_for_it_network(path[i],network[path[i-1]])
            d+=1
        how_far.append(d)
    idx= how_far.index(min(how_far))
    shortest_path= all_paths[idx]
    return shortest_path, min(how_far)


# --------------------------- FIND SMARTEST PATH-------------------------------------------
def find_smartest_path(nodes, measure,network, distances, time, coordinates, small_network):    
    print("check if there are duplicates in the given set of nodes")
    if checkIfDuplicates(nodes):
        print("There are duplicates in the given set of nodes")
        nodes = removeDuplicates(nodes)
        print("New set of nodes:", nodes)
    else :
        print("There are no duplicates in the given set of nodes")
        
    
    # check if there is a path between
    print('Check if there is a path between these nodes')
    if has_Path(nodes, network) == False:
        print("There is no path bewteen these nodes: ", nodes)
        return
    print('There is a path bewteen the these nodes: ', nodes)
    
    if len(nodes) == 2:
        print('Looking for paths between the two given nodes') 
        if measure=='d':
            path = dijkstra(distances, nodes[0], nodes[1],visited=[],distances={},predecessors={})
            minimum = find_weight(path, distances, time, coordinates, measure)
            print('The minumum weight is:', minimum)
            print('The shortest path according to the pysical distance is:', path)
            return path
        elif measure=='t':
            path = dijkstra(time, nodes[0], nodes[1],visited=[],distances={},predecessors={})
            minimum = find_weight(path, distances, time, coordinates, measure)
            print('The minumum weight is:', minimum)
            print('The shortest path according to the time distance is:', path)
            return path
        elif measure=='n':
            path = dijkstra(small_network, nodes[0], nodes[1],visited=[],distances={},predecessors={})
            minimum = find_weight(path, distances, time, coordinates, measure)
            print('The minumum weight is:', minimum)
            print('The shortest path according to the network distance is:', path)
            return path
    else:
        #if there are paths, let's look for all them
        print('Looking for all paths bewteen of the given nodes')
        all_paths=[]
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if j != i:
                    all_paths.append(dijkstra(small_network, nodes[i], nodes[j],visited=[],distances={},predecessors={}))
        # keep only those that goes throught all the nodes
        # if the nodes are not in a straight line the function give me error
        new_all = nodes_non_in_line(all_paths, nodes)
    
        print('Choosing the best path among all') 
        #choose the best one according to the distance   
        if measure=='d':
            path, minimum = physical_distance(new_all, distances)
            print('The minumum weight is:', minimum)
            print('The shortest path according to the pysical distance is:', path)
            return path
        elif measure=='t':
            path, minimum = time_distance(new_all, time)
            print('The minumum weight is:', minimum)
            print('The shortest path according to the time distance is:', path)
            return path
        elif measure=='n':
            path, minimum = network_distance(new_all, network)
            print('The minumum weight is:', minimum)
            print('The shortest path according to the network distance is:', path)
            return path
        
        
# ---------------------------------------- VISUALIZE ---------------------------------------

def draw_graph(path,G, coordinates, df):
    H = G.subgraph(path)

    latitude=[]
    longitude=[]
    for node in path:
        latitude.append(coordinates[node][1]/1000000)
        longitude.append(coordinates[node][0]/1000000)
        
    BBox = [df.latitude.min()/1000000,df.latitude.max()/1000000,df.longitude.min()/1000000,df.longitude.max()/1000000]
    plt.figure(figsize=(14,12))
    # Creates the map
    ca_map = plt.axes(projection=ccrs.PlateCarree())
    ca_map.add_feature(cfeature.LAND)
    ca_map.add_feature(cfeature.OCEAN)
    ca_map.add_feature(cfeature.COASTLINE)
    ca_map.add_feature(cfeature.BORDERS, edgecolor = 'lightgray',linestyle=':')
    ca_map.add_feature(cfeature.LAKES, alpha=0.5)
    ca_map.add_feature(cfeature.RIVERS)
    ca_map.add_feature(cfeature.STATES.with_scale('10m'))
    plt.scatter(longitude,latitude,
         color='blue', s=2, marker='o',
         transform=ccrs.PlateCarree())
    plt.scatter(mean(longitude), mean(latitude), 
                color='red', s=5000,facecolors='none',
                transform=ccrs.PlateCarree())
    ca_map.set_extent(BBox)
    plt.show()
    

def gmaps(coordinates, paths):
    latitude=[]
    longitude=[]
    for node in paths:
        latitude.append(coordinates[node][1]/1000000)
        longitude.append(coordinates[node][0]/1000000)
        
    lat_center = sum(latitude)/len(latitude)
    long_center = sum(longitude)/len(longitude)
    gmap2 = gmplot.GoogleMapPlotter(lat_center, long_center, 13)
    gmap2.scatter(latitude, longitude, '#0000FF', size=20, marker=True)
    gmap2.marker(latitude[0],longitude[0],"cornflowerblue", title="Flaminia & Dilara & Mechket")
    gmap2.plot(latitude, longitude, 'yellow', edge_width=2.5)
    gmap2.draw("mapCC.html")
   