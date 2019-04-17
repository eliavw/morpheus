import plotly as py
import plotly.graph_objs as go


def get_edge_trace(G):

    # Init and layout
    edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=4, color="black"), hoverinfo="none", mode="lines"
    )

    # Collect data from graph
    for edge in G.edges():
        x0, y0 = G.node[edge[0]]["pos"]
        x1, y1 = G.node[edge[1]]["pos"]
        edge_trace["x"] += tuple([x0, x1, None])
        edge_trace["y"] += tuple([y0, y1, None])
    return edge_trace


def get_node_trace(G, bipartite="func", marker_size=100, marker_symbol="square-open"):

    node_trace = go.Scatter(
        x=[],
        y=[],
        mode="markers+text",
        marker=dict(
            color="white",
            size=marker_size,
            symbol=marker_symbol,
            line=dict(width=4, color="black"),
        ),
    )

    relevant_nodes = [n for n in G.nodes() if G.nodes()[n]["bipartite"] == bipartite]

    for node in relevant_nodes:
        x, y = G.node[node]["pos"]
        node_trace["x"] += tuple([x])
        node_trace["y"] += tuple([y])

    # Labels separately
    node_trace["text"] = [node for node in relevant_nodes]

    return node_trace


def model_graph_traces(G):

    edge_trace = get_edge_trace(G)

    func_node_trace = get_node_trace(
        G, bipartite="func", marker_size=100, marker_symbol="square"
    )

    data_node_trace = get_node_trace(
        G, bipartite="data", marker_size=50, marker_symbol="circle"
    )

    return edge_trace, func_node_trace, data_node_trace


def model_graph_layout(title="Network graph"):
    return go.Layout(
        title=title,
        titlefont=dict(size=16),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
