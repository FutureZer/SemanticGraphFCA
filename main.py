from graph_preprocessing import process_empty_nodes
from graph_visualize import visualize_graph_list
from parse import from_dict
from read_dataset import load_graphs

# Пример данных в формате JSON (можно взять один из ваших примеров)
ptg_data = {
    "id": "0",
    "input": "This is outrageous!",
    "nodes": [
        {
            "id": 0,
            "label": "this",
            "anchors": [
                {
                    "from": 0,
                    "to": 4
                }
            ],
            "properties": [
                "sempos"
            ],
            "values": [
                "n.pron.indef"
            ]
        },
        {
            "id": 1,
            "label": "be",
            "anchors": [
                {
                    "from": 5,
                    "to": 7
                },
                {
                    "from": 18,
                    "to": 19
                }
            ],
            "properties": [
                "frame",
                "sempos",
                "sentmod"
            ],
            "values": [
                "en-v#ev-w218f2",
                "v",
                "enunc"
            ]
        },
        {
            "id": 2,
            "label": "outrageous",
            "anchors": [
                {
                    "from": 8,
                    "to": 18
                }
            ],
            "properties": [
                "sempos"
            ],
            "values": [
                "adj.denot"
            ]
        },
        {
            "id": 3
        }
    ],
    "edges": [
        {
            "source": 3,
            "target": 1,
            "label": "pred"
        },
        {
            "source": 1,
            "target": 2,
            "label": "pat"
        },
        {
            "source": 1,
            "target": 0,
            "label": "act"
        }
    ],
    "tops": [
        3
    ],
    "framework": "ptg",
    "time": "2025-04-11"
}

ucca_data = {
    "id": "0",
    "input": "This is outrageous!",
    "nodes": [
        {
            "id": 0,
            "anchors": [
                {
                    "from": 0,
                    "to": 4
                }
            ]
        },
        {
            "id": 1,
            "anchors": [
                {
                    "from": 5,
                    "to": 7
                }
            ]
        },
        {
            "id": 2
        },
        {
            "id": 3,
            "anchors": [
                {
                    "from": 8,
                    "to": 18
                }
            ]
        },
        {
            "id": 4
        },
        {
            "id": 5,
            "anchors": [
                {
                    "from": 18,
                    "to": 19
                }
            ]
        }
    ],
    "edges": [
        {
            "source": 2,
            "target": 3,
            "label": "s"
        },
        {
            "source": 2,
            "target": 0,
            "label": "a"
        },
        {
            "source": 2,
            "target": 1,
            "label": "f"
        },
        {
            "source": 2,
            "target": 5,
            "label": "u"
        },
        {
            "source": 4,
            "target": 2,
            "label": "h"
        }
    ],
    "tops": [
        4
    ],
    "framework": "ucca",
    "time": "2025-04-11"
}

eds_data = {
    "id": "0",
    "input": "This is outrageous!",
    "nodes": [
        {
            "id": 0,
            "label": "generic_entity",
            "properties": [
                "num"
            ],
            "values": [
                "sg"
            ],
            "anchors": [
                {
                    "from": 0,
                    "to": 4
                }
            ]
        },
        {
            "id": 1,
            "label": "_this_q_dem",
            "anchors": [
                {
                    "from": 0,
                    "to": 4
                }
            ]
        },
        {
            "id": 2,
            "label": "_outrageous_a_1",
            "properties": [
                "tense"
            ],
            "values": [
                "pres"
            ],
            "anchors": [
                {
                    "from": 8,
                    "to": 19
                }
            ]
        }
    ],
    "edges": [
        {
            "source": 1,
            "target": 0,
            "label": "bv"
        },
        {
            "source": 2,
            "target": 0,
            "label": "arg1"
        }
    ],
    "tops": [
        2
    ],
    "framework": "eds",
    "time": "2025-04-11"
}

drg_data = {
    "id": "0",
    "input": "This is outrageous!",
    "nodes": [
        {
            "id": 0
        },
        {
            "id": 1,
            "label": "\"now\""
        },
        {
            "id": 2,
            "label": "time.n.08"
        },
        {
            "id": 3,
            "label": "outrageous.a.01"
        },
        {
            "id": 4
        },
        {
            "id": 5,
            "label": "entity.n.01"
        },
        {
            "id": 6,
            "label": "EQU"
        },
        {
            "id": 7,
            "label": "TIME"
        }
    ],
    "edges": [
        {
            "source": 0,
            "target": 2,
            "label": "in"
        },
        {
            "source": 0,
            "target": 3,
            "label": "in"
        },
        {
            "source": 2,
            "target": 6
        },
        {
            "source": 6,
            "target": 1
        },
        {
            "source": 3,
            "target": 7
        },
        {
            "source": 7,
            "target": 2
        },
        {
            "source": 4,
            "target": 0,
            "label": "presupposition"
        },
        {
            "source": 4,
            "target": 5,
            "label": "in"
        }
    ],
    "tops": [
        0
    ],
    "framework": "drg",
    "time": "2025-04-11"
}

amr_data = {
    "id": "0",
    "input": "This is outrageous!",
    "nodes": [
        {
            "id": 0,
            "label": "this"
        },
        {
            "id": 1,
            "label": "outrageous-02"
        }
    ],
    "edges": [
        {
            "source": 1,
            "target": 0,
            "label": "arg0"
        }
    ],
    "tops": [
        1
    ],
    "framework": "amr",
    "time": "2025-04-11"
}


if __name__ == "__main__":
    # Define the path to your dataset and the notation you want to load
    dataset_path = "data\\bbcsport"  # Replace with the actual path

    # Load the data
    #loaded_amr_graphs = load_graphs(dataset_path, "amr")
    #loaded_drg_graphs = load_graphs(dataset_path, "drg")
    #loaded_eds_graphs = load_graphs(dataset_path, "eds")
    #loaded_ptg_graphs = load_graphs(dataset_path, "ptg")
    loaded_ucca_graphs = load_graphs(dataset_path, "ucca")

    #visualize_graph_list(loaded_amr_graphs['athletics'][:100])
    #visualize_graph_list(loaded_drg_graphs['athletics'][:100])
    #visualize_graph_list(loaded_eds_graphs['athletics'][:100])
    #visualize_graph_list(loaded_ptg_graphs['athletics'][:100])
    visualize_graph_list(loaded_ucca_graphs['athletics'][:100])

    preprocessed_graphs = []
    for ucca_graph in loaded_ucca_graphs['athletics'][:100]:
        preprocessed_graphs.append(process_empty_nodes(ucca_graph))

    visualize_graph_list(preprocessed_graphs, "preprocessed")

