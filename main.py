import networkx as nx
import heapq

class Graph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node, heuristic):
        self.graph.add_node(node, heuristic=heuristic)

    def add_edge(self, node1, node2, weight):
        self.graph.add_edge(node1, node2, weight=weight)

    def set_heuristic(self, node, heuristic):
        self.graph.nodes[node]['heuristic'] = heuristic

    def get_weight(self, node1, node2):
        return self.graph.edges[node1, node2]['weight']

    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))

class Graph1:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node, heuristic):
        self.graph.add_node(node, heuristic=heuristic)

    def add_edge(self, node1, node2, cost):
        self.graph.add_edge(node1, node2, cost=cost)

    def set_heuristic(self, node, heuristic):
        self.graph.nodes[node]['heuristic'] = heuristic

    def set_cost(self, node1, node2, cost):
        self.graph.edges[node1, node2]['cost'] = cost

    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))    

def uniform_cost_search(graph, start, goal):
    frontier = [(0, start)] # file de priorité (coût, nœud)
    explored = {start: (None, 0)} # ensemble des nœuds explorés

    while frontier:
        cost, node = heapq.heappop(frontier) # prendre le nœud avec le coût le plus faible
        if node == goal:
            # retourner le chemin trouvé
            path = [node]
            while start != path[0]:
                parent = explored[path[0]][0]
                path.insert(0, parent)
            return path, cost

        for neighbor in graph.get_neighbors(node):
            neighbor_cost = cost + graph.get_weight(node, neighbor)
            if neighbor not in explored or neighbor_cost < explored[neighbor][1]:
                explored[neighbor] = (node, neighbor_cost)
                heapq.heappush(frontier, (neighbor_cost, neighbor))

    return None

def greedy_best_first_search(graph, start, goal):
    frontier = [(graph.graph.nodes[start]['heuristic'], start)] # file de priorité (heuristique, nœud)
    explored = set() # ensemble des nœuds explorés
    
    while frontier:
        _, node = heapq.heappop(frontier) # prendre le nœud avec l'heuristique la plus faible
        if node == goal:
            # retourner le chemin trouvé
            path = [node]
            while node != start:
                node, _, _ = explored[node]
                path.append(node)
            path.reverse()
            return path
        
        if node not in explored:
            explored.add(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in explored:
                    heuristic = graph.graph.nodes[neighbor]['heuristic']
                    heapq.heappush(frontier, (heuristic, neighbor))
                    explored.add(neighbor)
                    explored[neighbor] = (node, heuristic)
                
    return None # pas de chemin trouvé


    
def a_star_search(graph, start, goal):
    frontier = [(0, start)] # file de priorité (f = g + h, nœud)
    explored = {} # ensemble des nœuds explorés, avec leur coût et leur parent
    
    while frontier:
        _, node = heapq.heappop(frontier) # prendre le nœud avec la plus petite valeur de f
        if node == goal:
            # retourner le chemin trouvé
            if goal not in explored:
                return None # Le nœud de destination n'a pas été atteint
            path = [node]
            while node != start:
                node = explored[node][1]
                path.append(node)
            path.reverse()
            return path, explored[goal][0]
        
        if node not in explored:
            explored[node] = (float('inf'), None) # initialiser le coût et le parent
            for neighbor in graph.get_neighbors(node):
                g = explored[node][0] + graph.graph[node][neighbor]['cost'] # coût à partir du nœud de départ
                h = graph.graph.nodes[neighbor]['heuristic'] # heuristique du nœud courant
                f = g + h # somme du coût et de l'heuristique
                heapq.heappush(frontier, (f, neighbor))
                if g < explored.get(neighbor, (float('inf'), None))[0]:
                    explored[neighbor] = (g, node)
                
    return None # pas de chemin trouvé


# Create graph
graph = Graph1()
graph.add_node('A', 5)
graph.add_node('B', 2)
graph.add_node('C', 3)
graph.add_node('D', 8)
graph.add_node('E', 1)
graph.add_edge('A', 'B', 2)
graph.add_edge('A', 'C', 3)
graph.add_edge('B', 'D', 1)
graph.add_edge('C', 'D', 3)
graph.add_edge('C', 'E', 4)
graph.set_heuristic('A', 5)
graph.set_heuristic('B', 2)
graph.set_heuristic('C', 3)
graph.set_heuristic('D', 1)
graph.set_heuristic('E', 0)

# Find path from A to D using A* search

result = a_star_search(graph, 'A', 'D')
if result is not None:
    path, cost = result
    print('Path:', path)
    print('Cost:', cost)
else:
    print('No path found')






graph = Graph()

# ajouter des nœuds
graph.add_node('A', 10)
graph.add_node('B', 7)
graph.add_node('C', 3)
graph.add_node('D', 2)
graph.add_node('E', 0)

# ajouter des arêtes
graph.add_edge('A', 'B', 3)
graph.add_edge('A', 'C', 1)
graph.add_edge('B', 'D', 5)
graph.add_edge('C', 'D', 4)
graph.add_edge('C', 'E', 2)
graph.add_edge('D', 'E', 1)

# effectuer la recherche
path, cost = uniform_cost_search(graph, 'A', 'E')

# afficher le chemin et le coût
print(path)  # ['A', 'C', 'E']
print(cost)  #


