def fix_layout(G):
    for n in G.nodes():
        node = G.nodes(data=True)[n]
        if node["bipartite"] == "func":
            node["shape"] = '"square"'
            node["width"] = "1"
        elif node["bipartite"] == "data":
            node["shape"] = '"circle"'
        else:
            pass

    return G
