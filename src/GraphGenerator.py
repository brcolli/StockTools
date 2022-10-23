import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import pandas as pd


class GraphGenerator:

    @staticmethod
    def root_node_with_branches(root_node: str, branch_nodes: list):

        graph = nx.Graph()

        all_nodes = [root_node] + branch_nodes
        for node in all_nodes:
            graph.add_node(node)

        for branch_node in branch_nodes:
            graph.add_edges_from([(root_node, branch_node)])

        graph = GraphGenerator._define_spring_layout(graph)

        edge_trace = GraphGenerator._convert_networkx_graph_edges_to_plotly(graph)
        node_trace = GraphGenerator._convert_networkx_graph_nodes_to_plotly(graph)

        fig = GraphGenerator._generate_plotly_networkx_figure(root_node, node_trace, edge_trace)

        return fig

    @staticmethod
    def _define_spring_layout(graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:

        pos = nx.spring_layout(graph, k=0.5, iterations=100)
        for n, p in pos.items():
            graph.nodes[n]['pos'] = p

        return graph

    @staticmethod
    def _convert_networkx_graph_edges_to_plotly(graph: nx.classes.graph.Graph):

        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        for edge in graph.edges():

            x0, y0 = graph.nodes[edge[0]]['pos']
            x1, y1 = graph.nodes[edge[1]]['pos']

            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        return edge_trace

    @staticmethod
    def _convert_networkx_graph_nodes_to_plotly(graph: nx.classes.graph.Graph):

        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                colorscale='pinkyl',
                reversescale=True,
                color=[],
                size=37,
                line=dict(width=0)))

        for node in graph.nodes():

            x, y = graph.nodes[node]['pos']
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])

        for node, adjacencies in enumerate(graph.adjacency()):

            node_trace['marker']['color'] += tuple([len(adjacencies[1])])
            node_info = adjacencies[0]
            node_trace['text'] += tuple([node_info])

        return node_trace

    @staticmethod
    def _generate_plotly_networkx_figure(query: str, node_trace, edge_trace):

        title = f"Closely Related Topics to {query}"
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=title,
                            titlefont=dict(size=16),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=21, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False,
                                       showticklabels=False, mirror=True),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, mirror=True)))

        return fig


if __name__ == '__main__':

    beta_companies = pd.read_csv('../doc/beta_companies_keywords.csv').T
    for company, row in beta_companies.iterrows():

        print(f'Generating image for {company}')

        company_graph = GraphGenerator.root_node_with_branches(company, row.to_list())
        company_graph.write_image(f'../doc/CloselyRelatedTopicsGraphs/{company}CloselyRelatedTopicsGraph.png')
