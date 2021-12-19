import argparse
import matplotlib.pyplot as plt
from matplotlib import path
import networkx as nx

"""
This code is for plotting the stylized scene graphs in matplotlib 
"""

from scene_graph_svg import SceneGraph, VideoGraph, load_data_from_predictions


def custom_draw_networkx_labels(
        G,
        pos,
        labels=None,
        font_size=12,
        font_color="k",
        font_family="sans-serif",
        font_weight="normal",
        alpha=None,
        bbox_list=None,
        horizontalalignment="center",
        verticalalignment="center",
        ax=None,
):
    """Draw node labels on the graph G.
    Same thing as the version in networkX, but here I use a
    list for nodes' boundary boxes instead of just one boundary box for all nodes
   """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Matplotlib required for draw()") from e
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax = plt.gca()

    if labels is None:
        labels = {n: n for n in G.nodes()}

    text_items = {}  # there is no text collection so we'll fake one
    count = 0
    for n, label in labels.items():
        (x, y) = pos[n]
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same
        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            transform=ax.transData,
            bbox=bbox_list[count],
            clip_on=True,
        )
        text_items[n] = t
        count += 1

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


def custom_draw_networkx_edges(
        G,
        pos,
        edgelist=None,
        edge_positions=None,
        width=1.0,
        edge_color="k",
        style="solid",
        alpha=None,
        arrowstyle="-|>",
        arrowsize=10,
        edge_cmap=None,
        edge_vmin=None,
        edge_vmax=None,
        ax=None,
        arrows=True,
        label=None,
        node_size=300,
        nodelist=None,
        node_shape="o",
        connectionstyle=None,
        min_source_margin=0,
        min_target_margin=0,
):
    """Draw the edges of the graph G.

    This is the draw_edges from NetworkX, but edited so I can draw
    edges in desired positions. I added #%%%%%%%%%%%%%# before and after
    each block of code I added/edited so it's clear what I changed.

    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import colorConverter, Colormap, Normalize
        from matplotlib.collections import LineCollection
        from matplotlib.patches import FancyArrowPatch
        import numpy as np
    except ImportError as e:
        raise ImportError("Matplotlib required for draw()") from e
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = list(G.edges())

    if len(edgelist) == 0:  # no edges!
        if not G.is_directed() or not arrows:
            return LineCollection(None)
        else:
            return []

    if nodelist is None:
        nodelist = list(G.nodes())

    # FancyArrowPatch handles color=None different from LineCollection
    if edge_color is None:
        edge_color = "k"

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    # %%%%%%%%%%%%%#
    # A list of all the points in the edges
    edge_curves = [np.array(edge_positions[(e[0], e[1])]) for e in edgelist]
    # %%%%%%%%%%%%%#

    # Check if edge_color is an array of floats and map to edge_cmap.
    # This is the only case handled differently from matplotlib
    if (
            np.iterable(edge_color)
            and (len(edge_color) == len(edge_pos))
            and np.alltrue([isinstance(c, Number) for c in edge_color])
    ):
        if edge_cmap is not None:
            assert isinstance(edge_cmap, Colormap)
        else:
            edge_cmap = plt.get_cmap()
        if edge_vmin is None:
            edge_vmin = min(edge_color)
        if edge_vmax is None:
            edge_vmax = max(edge_color)
        color_normal = Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

    if not G.is_directed() or not arrows:
        edge_collection = LineCollection(
            edge_pos,
            colors=edge_color,
            linewidths=width,
            antialiaseds=(1,),
            linestyle=style,
            transOffset=ax.transData,
            alpha=alpha,
        )

        edge_collection.set_cmap(edge_cmap)
        edge_collection.set_clim(edge_vmin, edge_vmax)

        edge_collection.set_zorder(1)  # edges go behind nodes
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)

        return edge_collection

    arrow_collection = None

    if G.is_directed() and arrows:
        # Note: Waiting for someone to implement arrow to intersection with
        # marker.  Meanwhile, this works well for polygons with more than 4
        # sides and circle.

        def to_marker_edge(marker_size, marker):
            if marker in "s^>v<d":  # `large` markers need extra space
                return np.sqrt(2 * marker_size) / 2
            else:
                return np.sqrt(marker_size) / 2

        # Draw arrows with `matplotlib.patches.FancyarrowPatch`
        arrow_collection = []
        mutation_scale = arrowsize  # scale factor of arrow head

        # FancyArrowPatch doesn't handle color strings
        arrow_colors = colorConverter.to_rgba_array(edge_color, alpha)
        for i, (src, dst) in enumerate(edge_pos):
            # %%%%%%%%%%%%%#
            # we ignore the first point because it causes the path to close on itself for some reason
            points = edge_curves[i][1:]
            # print(len(points))

            codes = [path.Path.MOVETO] + [path.Path.CURVE4] * (len(points) - 1)
            curve = path.Path(points, codes, closed=False)

            # # some edge position annotation code for debugging
            # plt.scatter(points[:, 0], points[:, 1])
            # nums = range(len(points))
            # for j, txt in enumerate(nums):
            #     ax.annotate(txt, (points[j][0], points[j][1]))
            # %%%%%%%%%%%%%#

            x1, y1 = src
            x2, y2 = dst
            shrink_source = 0  # space from source to tail
            shrink_target = 0  # space from  head to target
            if np.iterable(node_size):  # many node sizes
                source, target = edgelist[i][:2]
                source_node_size = node_size[nodelist.index(source)]
                target_node_size = node_size[nodelist.index(target)]
                shrink_source = to_marker_edge(source_node_size, node_shape)
                shrink_target = to_marker_edge(target_node_size, node_shape)
            else:
                shrink_source = shrink_target = to_marker_edge(node_size, node_shape)

            if shrink_source < min_source_margin:
                shrink_source = min_source_margin

            if shrink_target < min_target_margin:
                shrink_target = min_target_margin

            if len(arrow_colors) == len(edge_pos):
                arrow_color = arrow_colors[i]
            elif len(arrow_colors) == 1:
                arrow_color = arrow_colors[0]
            else:  # Cycle through colors
                arrow_color = arrow_colors[i % len(arrow_colors)]

            if np.iterable(width):
                if len(width) == len(edge_pos):
                    line_width = width[i]
                else:
                    line_width = width[i % len(width)]
            else:
                line_width = width

            # %%%%%%%%%%%%%#
            arrow = FancyArrowPatch(
                path=curve,
                arrowstyle=arrowstyle,
                mutation_scale=mutation_scale,
                color=arrow_color,
                linewidth=line_width,
                connectionstyle=connectionstyle,
                linestyle=style,
                zorder=1,
            )  # arrows go behind nodes
            # %%%%%%%%%%%%%#
            # arrow = FancyArrowPatch(
            #     (x1, y1),
            #     (x2, y2),
            #     arrowstyle=arrowstyle,
            #     shrinkA=shrink_source,
            #     shrinkB=shrink_target,
            #     mutation_scale=mutation_scale,
            #     color=arrow_color,
            #     linewidth=line_width,
            #     connectionstyle=connectionstyle,
            #     linestyle=style,
            #     zorder=1,
            # )  # arrows go behind nodes

            # There seems to be a bug in matplotlib to make collections of
            # FancyArrowPatch instances. Until fixed, the patches are added
            # individually to the axes instance.
            arrow_collection.append(arrow)
            ax.add_patch(arrow)

    # update view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))

    w = maxx - minx
    h = maxy - miny
    padx, pady = 0.05 * w, 0.05 * h
    corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return arrow_collection


def set_node_edge_positions(union_graph_nx, union_graph_gviz):
    """
    Get the node/edge position info from graphviz
    """
    # parse the dot data from pygraphviz to extract the node and edge positions
    node_positions = {}
    for node in union_graph_nx.nodes:
        node_info = union_graph_gviz.get_node(node)
        raw_positions = node_info.attr['pos'].split(',')
        node_positions[node] = [float(pos) for pos in raw_positions]

    edge_positions = {}
    for edge in union_graph_nx.edges:
        edge_info = union_graph_gviz.get_edge(edge[0], edge[1])
        raw_pos = edge_info.attr['pos']
        raw_pos = (raw_pos[2:]).split(' ')  # remove first 2 characters and split

        points = [pair.split(',') for pair in raw_pos]
        points = [[float(pair[0]), float(pair[1])] for pair in points]
        edge = (edge[0], edge[1])
        edge_positions[edge] = points

    return node_positions, edge_positions


def find_plot_limits(node_positions, edge_positions, side_buffer, top_buffer):
    """
    Find the axis limits for the plot, and add a buffer so node labels aren't cutoff
    """
    Xlim = [100000, -100000]  # the bounds of the plot for the total graph
    Ylim = [100000, -100000]  # find the furthest out node/edge point

    for anode in node_positions:
        x, y = node_positions[anode]
        Xlim = [min(Xlim[0], x), max(Xlim[1], x)]
        Ylim = [min(Ylim[0], y), max(Ylim[1], y)]

    for _, anedge in edge_positions.items():
        for point in anedge:
            x, y = point[0], point[1]
            Xlim = [min(Xlim[0], x), max(Xlim[1], x)]
            Ylim = [min(Ylim[0], y), max(Ylim[1], y)]

    Xlim = [Xlim[0] - side_buffer, Xlim[1] + side_buffer]
    Ylim = [Ylim[0] - top_buffer, Ylim[1] + top_buffer]
    return Xlim, Ylim


def display_scene_graphs(video_graph: VideoGraph, args):
    """
    Display the frames of video graph
    :param video_graph:
    :param step: which frames to visualize (e.g. 1 vis per 30 frames)
    """
    # Create a union-graph that includes all nodes and edges from all the needed frames in video
    union_graph_nx = nx.MultiDiGraph()

    union_graph_nx.add_nodes_from(video_graph.total_nodes)
    union_graph_nx.add_edges_from(video_graph.total_edges)

    union_graph_gviz = nx.drawing.nx_agraph.to_agraph(union_graph_nx)

    # make the font size high and nodes spread apart, so there's less node overlap
    for node_name in video_graph.total_nodes:
        node = union_graph_gviz.get_node(node_name)
        node.attr['label'] = video_graph.total_label_map[node_name]
        node.attr['fontsize'] = 12
        node.attr['shape'] = 'rectangle'
        node.attr['style'] = 'rounded, filled'
        if node_name in video_graph.total_obj_nodes:
            node.attr['fillcolor'] = '#FFB0B9'
        else:
            node.attr['fillcolor'] = '#C3E2B3'

        if len(node_name) >= 6:
            node.attr['margin'] = '0.4,0.1'

    for edge_name in video_graph.total_edges:
        edge = union_graph_gviz.get_edge(edge_name[0], edge_name[1])
        # edge.attr['arrowsize'] = 1    # these params mess more things up, not worth it
        # edge.attr['minlen'] = 1
        # edge.attr['headport'] = '_'
        # edge.attr['tailport'] = '_'

    union_graph_gviz.graph_attr['nodesep'] = 3.0
    union_graph_gviz.graph_attr['root'] = 'person'
    union_graph_gviz.graph_attr['overlap'] = False
    union_graph_gviz.graph_attr['splines'] = True
    union_graph_gviz.graph_attr['K'] = 3.0

    # Use pygraphviz for generating the node (and edge) layouts, then draw with NetworkX
    union_graph_gviz.layout(prog='sfdp')

    ### Uncomment below to write out the pygraphviz vis to a file for fun
    # union_graph_gviz.draw('union_graph.png')
    # union_graph_gviz.write('union_graph.dot')
    # union_graph_gviz.draw('union_graph.svg', format='svg')

    node_positions, edge_positions = set_node_edge_positions(union_graph_nx, union_graph_gviz)

    Xlim, Ylim = find_plot_limits(node_positions, edge_positions, side_buffer=250, top_buffer=30)

    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_axes([0, 0, 1, 1])  # position: left, bottom, width, height
    ax.set_axis_off()  # this gets rid of whitespace around plot

    # draw out the total_graph in a raw, un-styled format for fun
    nx.draw_networkx(union_graph_nx, pos=node_positions, labels=video_graph.total_label_map,
                     connectionstyle="arc3,rad=0.1", min_target_margin=25)

    plt.xlim(Xlim)
    plt.ylim(Ylim)
    plt.axis('scaled')

    plt.draw()
    plt.show()

    for i in range(0, len(video_graph), args.step):
        # Rather than visualizing a subset of nodes and edges of the the union graph,
        # We create a brand new graph. It's (probably) easier this way
        sg: SceneGraph = video_graph[i]

        current_graph = nx.MultiDiGraph()
        current_graph.add_nodes_from(sg.nodes)
        current_graph.add_edges_from(sg.edges)

        color_list = []
        bbox_list = []
        for curr_node in current_graph:
            if curr_node in sg.obj_nodes:  # RGBA
                color = (1, 0, 0, .3)
            elif curr_node in sg.rel_nodes:
                color = (0, 1, 0, .3)
            elif curr_node in sg.attr_nodes:
                color = (0, 0, 1, .3)
            else:
                raise ValueError('Incorrect Node: ' + curr_node)

            bbox = dict(color=color, edgecolor='black', boxstyle='round,pad=0.13')
            color_list.append(color)
            bbox_list.append(bbox)

        fig = plt.figure()  # To plot the next graph in a new figure
        fig.tight_layout()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        # We only need to draw the labels and edges. No need to draw the nodes
        custom_draw_networkx_labels(G=current_graph, pos=node_positions, labels=sg.label_map,
                                    ax=plt.gca(), bbox_list=bbox_list, font_size=11)

        custom_draw_networkx_edges(G=current_graph, pos=node_positions, ax=plt.gca(),
                                   edge_positions=edge_positions, node_size=100)

        plt.axis('scaled')

        plt.xlim(Xlim)
        plt.ylim(Ylim)

        plt.draw()
        plt.show()


def main(args):
    video_graph: VideoGraph = load_data_from_predictions(args)

    display_scene_graphs(video_graph, args)


if __name__ == '__main__':
    example_videos = ['5INX3', '3VH9O', '00T1E']

    example_vid = example_videos[0]

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="scene_graph_data/" + example_vid + '/')
    parser.add_argument("--step", type=int, default=10)

    args = parser.parse_args()
    main(args)
