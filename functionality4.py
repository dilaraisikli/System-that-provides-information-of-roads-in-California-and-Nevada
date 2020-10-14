import pandas as pd
from collections import defaultdict
from queue import PriorityQueue
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import gmplot
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import functionality2
from statistics import mean

# ----------------------------------------BASIC FUNCTIONS------------------------------------
def findmin (x):# this function fails if the list length is 0 
        if x[0] != 0 :
            minimum = x[0]
        else:
            minimum  = x[1]
        for i in x[1:]:
            if i < minimum and i != 0 and x.index(i) not in visitedNodeList : 
                minimum = i
        visitedNodeList.append(x.index(minimum))
        return (x.index(minimum))


def get_coordinates():
    coordinates = defaultdict(list)
    with open('USA-road-d.CAL.co', 'r') as f:
        for line in f:
            if line[0]=='v':
                n, lat, long= list(map(int, line[2::].split()))
                coordinates[n]=[lat, long]
                
    df = pd.DataFrame(coordinates).T
    df.rename(columns={0:'latitude', 1:'longitude'}, inplace = True)
    return coordinates, df

def get_network():
    network = defaultdict(list)
    with open('USA-road-d.CAL.gr', 'r') as f:
        for line in range(f):
            if line[0] == 'a': 
                n1, n2, d= list(map(int, line[2::].split()))
                network[n1].append(n2)
                network[n2].append(n1)
    return network


#-------------------------------------- SHORTEST PATH --------------------------------------------
# this function try to find the shortest path for begining and destination nodes
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
    for i in range(1,len(nodes)):
        ans = bfs(network, nodes[i-1])
        if nodes[i] in ans:
            continue
        else:
            return False
    return True
#-------------------------------------- SHORTEST PATH --------------------------------------------
# this function try to find the shortest path for begining and destination nodes

def djikstra(g, begining, destination):
    
    add_distance = {}
    follow_node = {}
    limitP = PriorityQueue()
    limitP.put(begining, 0)
    add_distance[begining] = 0
    follow_node[begining] = None
   
    while not limitP.empty():
        curr = limitP.get()
        if curr == destination:
            break
        for i in [l[0] for l in g[curr]]:
            value = add_distance[curr] + [l[1] for l in g[curr] if l[0]==i][0]
            if i not in add_distance or value < add_distance[i]:
                add_distance[i] = value
                priority = value
                limitP.put(i, priority)
                follow_node[i] = curr
    point = destination
    graph_route = []
    graph_route.append(point)
    while point != begining:
        graph_route = [follow_node[point]] + graph_route
        point = follow_node [point]
    final_dest= add_distance[destination]
    return graph_route, final_dest  #retuns optimal route between two nodes and add them as final_dest



def shorthest_unordered_path(sequence_of_nodes, distances_function):
    start_node= sequence_of_nodes[0] #start point from nodes list
    totalRouteForVisual= []  #this list has all route for using on visual. function 
    routeListForVisual=[]

    columnsName=['first', 'second', distances_function + '(first,second)']
    if distances_function == "travel_distance": #if distances_function is distance read distance file and create a data frame 
        with open("USA-road-d.CAL.gr", 'r') as f:
            reader = f.readlines()
            reader = reader[7:]
            NodeDataFrame = pd.DataFrame([map(int, i.strip('\n').strip("a ").split()) for i in reader], columns=columnsName)


    elif distances_function == "travel_time":  #if distances_function is time read time file and create a data frame 
        with open("USA-road-t.CAL.gr", 'r') as f:
            reader = f.readlines()
            reader = reader[7:]
            NodeDataFrame = pd.DataFrame([map(int, i.strip('\n').strip("a ").split()) for i in reader], columns=columnsName)

    elif distances_function == "network_distance ": # Here we get the data we need for creating network_distance from travel distance data
        with open("USA-road-d.CAL.co", 'r') as f:
            reader = f.readlines()
            reader = reader[7:]
            NodeDataFrame = pd.DataFrame([map(int, i.strip('\n').strip("a ").split()) for i in reader], columns=columnsName)


    if distances_function == "travel_distance" or distances_function == "travel_time": #if distances_function is time or distance create a dictinoary from our dataframes
        nodesDic = defaultdict(list)
        for index in NodeDataFrame.index:
            nodesDic[NodeDataFrame["first"].iloc[index]].append((NodeDataFrame["second"].iloc[index], NodeDataFrame[distances_function +"(first,second)"].iloc[index]))

        for key, value in nodesDic.items():
            list_temp=[]
            for i in range(len(value)):
                list_temp.append(value[i][0])
            network[key]=list_temp


    elif distances_function == "network_distance" :  #if distances_function is network create a dictinoary from our dataframes with all costs are 1
        nodesDic = defaultdict(list)
        for index in NodeDataFrame.index:
            nodesDic[NodeDataFrame["first"].iloc[index]].append((NodeDataFrame["second"].iloc[index], 1))
        for key, value in nodesDic.items():
            list_temp=[]
            for i in range(len(value)):
                list_temp.append(value[i][0])
            network[key]=list_temp

    totalmatrix= []
    #create a matrix with elements which are optimal distance between each nodes
    connection = has_Path(sequence_of_nodes,network) #if there is conn. use djikstra
    if connection == True:
        for j in range(len(sequence_of_nodes)):
            routeList=[]
            for i in range(len(sequence_of_nodes)):

                route, cost = djikstra(nodesDic, sequence_of_nodes[j], sequence_of_nodes[i]) 
                routeList.append(cost)
            totalmatrix.append(routeList) 
    else:
        print("Not possible")

    visitedNodeList=[0] #take visited node to avoid go again 
    compareList= [] 
    sortedList= []
    #find min and update minNode
    minnode = findmin(totalmatrix[0]) 
    for i in range(1,len(totalmatrix)-2):
        newMin = findmin(totalmatrix[minnode])
        minnode= newMin
    #compare index list for find index of last node
    for i in range(len(totalmatrix)):
        compareList.append(i)
    for i in compareList:
        if i not in visitedNodeList:
            visitedNodeList.append(i)
#match index and actual nodes and create a new list for ORDERED NODE LIST    
for i in range(len(sequence_of_nodes)):
        if visitedNodeList[i] == sequence_of_nodes.index(sequence_of_nodes[i]):
            sortedList.append(sequence_of_nodes[i])

    #make djikstra for ordered list
    for i in range(len(sortedList)-1):
        route, cost = djikstra(nodesDic, sortedList[i], sortedList[i+1]) 
        routeListForVisual.append(route)
        routeList.append((route, cost))

    print(routeList) #recive optimal route map list
    #Since it's not in the right structure (the path we have) we put it into something that is ready to be Visualized
    #Start to Visualize the Shortest Ordered Route
    for i in range(len(routeListForVisual)):
        for j in range(len(routeListForVisual[i])):
            totalRouteForVisual.append(routeListForVisual[i][j])
        totalRouteForVisual.remove(routeListForVisual[i][-1])
    totalRouteForVisual.append(routeListForVisual[-1][-1])
    return totalRouteForVisual  #merge optimal route map list




# ---------------------------------------------VISUALIZE-----------------------------------------------

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
    gmap2.draw("mapPP.html")
