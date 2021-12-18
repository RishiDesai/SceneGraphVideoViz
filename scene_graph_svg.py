import argparse
import os
import pickle
import cairosvg
import numpy as np
import scipy
from xml.dom import minidom
import pygraphviz

"""
This code is for using GraphViz to create svg pictures of the scene graphs 
"""


class SceneGraph:
    """
    Contains the scene graph for a single frame
    """

    def __init__(self, obj_nodes, rel_nodes, attr_nodes, rel_edges, attr_edges):
        self.obj_nodes = list(obj_nodes)
        self.rel_nodes = list(rel_nodes)
        self.attr_nodes = list(attr_nodes)

        self.rel_edges = list(rel_edges)
        self.attr_edges = list(attr_edges)

        self.nodes = self.obj_nodes + self.rel_nodes + self.attr_nodes
        self.edges = self.rel_edges + self.attr_edges

        self.label_map = {}  # we need a lab
        for label in self.nodes:
            split = label.split('#')
            self.label_map[label] = split[1] if len(split) == 3 else label

    def __str__(self):
        print(str(self.nodes) + '\n' + str(self.edges))


class VideoGraph:
    """
    A wrapper for a list of scene graphs objects
    step: tells which frames to add to the video_graph. (e.g. every 15th or 30th frame)
    """

    def __init__(self, scene_graphs, step, title):
        self.scene_graphs: list[SceneGraph] = scene_graphs

        self.total_nodes = set()
        self.total_edges = set()
        self.total_rel_nodes = set()
        self.total_obj_nodes = set()
        self.total_attr_nodes = set()
        self.total_label_map = {}
        self.step = step
        self.title = title

        # only add sg's from the frames we are going to use, so it's less crowded overall
        for i in range(0, len(self.scene_graphs), step):
            sg = self.scene_graphs[i]

            self.total_nodes.update(sg.nodes)
            self.total_edges.update(sg.edges)

            self.total_rel_nodes.update(sg.rel_nodes)
            self.total_obj_nodes.update(sg.obj_nodes)
            self.total_attr_nodes.update(sg.attr_nodes)
            self.total_label_map.update(sg.label_map)

        self.total_edges = list(self.total_edges)
        self.total_nodes = list(self.total_nodes)

        self.total_rel_nodes = list(self.total_rel_nodes)
        self.total_obj_nodes = list(self.total_obj_nodes)
        self.total_attr_nodes = list(self.total_attr_nodes)

        self.raw_signal = {}
        self.new_signal = {}

    def median_filter(self, kernel_size):
        """
        Predictions have a lot of noise, so median_filter helps the scene graph video ook less jittery
        """
        self._create_signal()
        self._filter_signal(kernel_size)
        self._enforce_signal()

    def _create_signal(self):
        """
        Create a temporal signal for each of the elements (nodes and edges)
        """
        num_frames = len(self)
        for node in self.total_nodes:
            self.raw_signal[node] = np.zeros(num_frames)
        for edge in self.total_edges:
            self.raw_signal[edge] = np.zeros(num_frames)

        for i in range(num_frames):
            sg: SceneGraph = self[i]
            for node in sg.nodes:
                self.raw_signal[node][i] = 1
            for edge in sg.edges:
                self.raw_signal[edge][i] = 1

    def _filter_signal(self, kernel_size):
        """
        Run a median filter to smooth things out
        """
        for elem, signal in self.raw_signal.items():
            filtered_signal = scipy.signal.medfilt(signal, kernel_size=kernel_size)
            self.new_signal[elem] = filtered_signal

    def _enforce_signal(self):
        """
        Update the per-frame scene graphs with the filtered signal, so they're smoother
        """
        for i in range(len(self)):
            sg: SceneGraph = self[i]

            def _enforce(elem, struct):
                old_status = elem in struct
                new_status = bool(self.new_signal[elem][i])

                if new_status == old_status:
                    return
                elif new_status and not old_status:
                    struct.append(elem)
                elif not new_status and old_status:
                    struct.remove(elem)
                else:
                    raise ValueError  # should not happen

            for node in self.total_nodes:
                _enforce(node, sg.nodes)
            for edge in self.total_edges:
                _enforce(edge, sg.edges)

    def __getitem__(self, item):
        return self.scene_graphs[item]

    def __len__(self):
        return len(self.scene_graphs)


def load_scene_graph_json(graphs):
    """
    Creates video graph from a list of json scene graph files
    """
    scene_graphs_raw = []
    for graph in graphs:
        scene_graph = {'objects': [], 'attributes': [], 'relationships': [], 'url': graph['url']}
        for obj in graph['objects']:
            name = ''
            if 'name' in obj:
                name = obj['name']
            elif 'names' in obj and len(obj['names']) > 0:
                name = obj['names'][0]
            scene_graph['objects'].append({'name': name})
        scene_graph['attributes'] = graph['attributes']
        scene_graph['relationships'] = graph['relationships']

        scene_graphs_raw.append(scene_graph)

    # Now to convert the scene graph encoding into a usable network
    scene_graphs = []
    for sg in scene_graphs_raw:
        objs_raw, rels_raw, attrs_raw = sg['objects'], sg['relationships'], sg['attributes']

        obj_nodes = []
        idx_to_obj = {}
        for idx, obj in enumerate(objs_raw):
            idx_to_obj[idx] = obj['name']
            obj_nodes.append(obj['name'])

        rel_edges = []
        rel_nodes = []
        for rel_entry in rels_raw:
            pred = rel_entry['predicate']
            subj_idx = rel_entry['subject']
            obj_idx = rel_entry['object']

            subj = idx_to_obj[subj_idx]
            obj = idx_to_obj[obj_idx]

            rel_edges.append((subj, pred))  # these are the two edges for each predicate
            rel_edges.append((pred, obj))
            rel_nodes.append(pred)

        attr_edges = []
        attr_nodes = []
        for attr_entry in attrs_raw:
            attr = attr_entry['attribute']
            obj_idx = attr_entry['object']

            obj = idx_to_obj[obj_idx]

            attr_edges.append((obj, attr))  # only one edge for an attribute
            attr_nodes.append(attr)

        scene_graph = SceneGraph(obj_nodes, rel_nodes, attr_nodes, rel_edges, attr_edges)

        scene_graphs.append(scene_graph)

    video_graph = VideoGraph(scene_graphs)
    return video_graph


def load_data_from_predictions(args):
    """
    Load the scene graphs from a list of triplets defining the edges/nodes
    """
    # helps with readability for multi-word node names
    node_label_map = {
        'notlookingat': 'not looking at',
        'lookingat': 'looking at',
        'sittingon': 'sitting on',
        'lyingon': 'lying on',
        'notcontacting': 'not contacting',
        'infrontof': 'in front of',
        'onthesideof': 'on the side of',
        'sofacouch': 'sofa/couch',
        'cupglassbottle': 'cup/glass/bottle'
    }
    with open(os.path.join(args.video, 'hor_triplets.pkl'), 'rb') as f:
        triplets = pickle.load(f)

    triplets_raw = [trip.split('\n') for trip in triplets]
    # len(triplets_raw) is the number of frames in video

    scene_graph_list = []
    for triplets in triplets_raw:
        obj_nodes, rel_nodes = set(), set()
        rel_edges = []
        for trip in triplets:  # Each triplet is in the form "Object-Relationship-Object"
            trip = trip.replace('-', '')
            trip = trip.split()
            if len(trip) < 3:
                continue

            # e.g. replace 'lyingon' with 'lying on'
            trip = [node_label_map[obj_label] if obj_label in node_label_map else obj_label for obj_label in trip]

            # In order to distinguish repeated relationships, we make the relationship label the whole triplet itself.
            # Separate with hash (#), so we can easily split string later
            trip[1] = "%s#%s#%s" % (trip[0], trip[1], trip[2])

            rel_edges.append((trip[0], trip[1]))  # add edges
            rel_edges.append((trip[1], trip[2]))

            obj_nodes.add(trip[0])  # add nodes
            obj_nodes.add(trip[2])
            rel_nodes.add(trip[1])

        curr_sg = SceneGraph(obj_nodes=obj_nodes, rel_nodes=rel_nodes, attr_nodes=[], rel_edges=rel_edges, attr_edges=[])
        scene_graph_list.append(curr_sg)

    if args.video[-1] == '/':
        video_title = args.video.split('/')[-2]  # get the video name
    else:
        video_title = args.video.split('/')[-1]

    video_graph = VideoGraph(scene_graph_list, step=args.step, title=video_title)
    return video_graph


def frames_to_video(video_graph: VideoGraph):
    """
    Make a video from the svg images
    """
    video = video_graph.title
    folder = './%s_svg' % video
    fnames = sorted(os.listdir(folder))
    out_folder = './%s_png' % video
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for fname in fnames:
        cairosvg.svg2png(
            file_obj=open(os.path.join(folder, fname), "r"),
            write_to=os.path.join(out_folder, fname.split('.')[0] + '.png')
        )
    os.system("ffmpeg -i %s_png/frame%%03d.png %s.mp4" % (video, video))


def create_union_graph(video_graph: VideoGraph):
    """
    Create the union graph of all frame scene graphs, so we can output the its svg and dot files
    :return: the file_name of the union svg file
    """
    # Create a union-graph that includes all nodes and edges from all the needed frames in video
    union_graph_gviz = pygraphviz.AGraph(directed=True)
    union_graph_gviz.add_nodes_from(video_graph.total_nodes)
    union_graph_gviz.add_edges_from(video_graph.total_edges)

    # stylistic stuff for nodes/edges
    for node_name in video_graph.total_nodes:
        node = union_graph_gviz.get_node(node_name)
        node.attr['label'] = video_graph.total_label_map[node_name]
        if node_name in video_graph.total_obj_nodes:
            node.attr['fillcolor'] = '#FFB0B9'
            node.attr['fontsize'] = 20
        else:
            node.attr['fillcolor'] = '#C3E2B3'

    union_graph_gviz.node_attr['shape'] = 'rectangle'
    union_graph_gviz.node_attr['style'] = 'rounded, filled'

    union_graph_gviz.edge_attr['arrowsize'] = .5
    union_graph_gviz.edge_attr['arrowhead'] = 'vee'
    union_graph_gviz.edge_attr['color'] = '#7F7F7F'

    union_graph_gviz.graph_attr['root'] = 'person'
    union_graph_gviz.graph_attr['overlap'] = False
    union_graph_gviz.graph_attr['splines'] = True

    filename = video_graph.title + '_union'
    union_graph_gviz.layout(prog='sfdp')

    # We write out the union to SVG, so we can parse it for the per-Frame SVGs

    # union_graph_gviz.write(filename + '.dot')
    union_graph_gviz.draw(filename + '.svg', format='svg')

    # create_frame_graphs(filename + '.svg', video_graph)

    return filename + '.svg'


def create_frame_graphs(union_graph_filename, video_graph: VideoGraph):
    """
    :param union_graph_filename: The union graph's svg file
    Parse the union svg file, remove unnecessary elements and write out all the
    per-frame svg files
    This allows us to bypass GraphViz's layout system, which ignores node positional info
    """
    for i in range(0, len(video_graph), video_graph.step):
        # Go through the union graph's svg file (it's xml) and remove unneeded elements
        sg: SceneGraph = video_graph[i]

        xmldoc = minidom.parse(union_graph_filename)
        itemlist = xmldoc.getElementsByTagName('g')

        to_remove = []  # nodes/edges to remove!
        for item in itemlist:
            item_class = item.attributes['class'].value
            item_title_raw = item.getElementsByTagName('title')
            item_title = item_title_raw[0].firstChild.data

            if item_class == 'node' and item_title not in sg.nodes:
                to_remove.append(item)

            elif item_class == 'edge':
                edge = tuple(item_title.split('->'))  # tuple should be length 2
                if edge not in sg.edges:
                    to_remove.append(item)

        for dead in to_remove:  # delete the unneeded nodes and edges
            parent = dead.parentNode
            parent.removeChild(dead)

        viddir = video_graph.title + '_svg'
        if not os.path.isdir(viddir):
            os.mkdir(viddir)

        loc = os.path.join(viddir, 'frame%s.svg' % str(i).zfill(3))
        with open(loc, "w") as fs:
            fs.write(xmldoc.toxml())
            fs.close()


def main(args):
    video_graph: VideoGraph = load_data_from_predictions(args)

    video_graph.median_filter(kernel_size=25)

    # plt.figure()
    # plt.plot(video_graph.raw_signal['box'])
    # plt.plot(video_graph.new_signal['box'])
    # plt.show()
    #
    # plt.figure()
    # plt.plot(video_graph.raw_signal['person#not contacting#box'])
    # plt.plot(video_graph.new_signal['person#not contacting#box'])
    # plt.show()

    union_graph_file = create_union_graph(video_graph)

    create_frame_graphs(union_graph_filename=union_graph_file, video_graph=video_graph)

    frames_to_video(video_graph)


if __name__ == '__main__':

    # example_videos = ['5INX3', '3VH9O', '00T1E']
    example_videos = ['5INX3']

    for example_vid in example_videos:
        parser = argparse.ArgumentParser()
        parser.add_argument("--video", type=str, default="scene_graph_data/" + example_vid + '/')
        parser.add_argument("--step", type=int, default=1)

        args = parser.parse_args()
        main(args)
