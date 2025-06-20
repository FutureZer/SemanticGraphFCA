from typing import List, Dict, Union, Optional
import pickle
from graph_model import SemanticGraph

class LabelEncoder:
    """
    Класс для инкапсуляции логики кодирования меток узлов и ребер.
    Каждый экземпляр этого класса будет иметь свои собственные словари кодирования.
    """
    def __init__(self):
        # Словари для хранения соответствия меток и их числовых кодов
        # {None: 0} для меток узлов обеспечивает, что отсутствие метки кодируется как 0.
        self.node_label_to_id: Dict[Optional[str], int] = {None: 0}
        self.edge_label_to_id: Dict[str, int] = {}

        # Следующие доступные ID для новых меток
        self.next_node_label_id: int = 1
        self.next_edge_label_id: int = 1

    def encode_node_label(self, label: Optional[str]) -> int:
        """Преобразует метку узла в целочисленный ID."""
        if label not in self.node_label_to_id:
            self.node_label_to_id[label] = self.next_node_label_id
            self.next_node_label_id += 1
        return self.node_label_to_id[label]

    def encode_edge_label_triple(self, source_label: str, edge_label: str, target_label: str) -> int:
        """
        Кодирует направленное ребро как тройку (метка источника, метка ребра, метка цели),
        однозначно сопоставляя ее целочисленному ID.
        Это сохраняет направленность без дублирования ребра.
        """
        # Используем полную тройку как ключ для кодирования направления ребра
        triple_key = f"{source_label}::{edge_label}::{target_label}"

        if triple_key not in self.edge_label_to_id:
            self.edge_label_to_id[triple_key] = self.next_edge_label_id
            self.next_edge_label_id += 1
        return self.edge_label_to_id[triple_key]

    def get_node_encodings(self) -> Dict[Optional[str], int]:
        """Возвращает текущие кодировки меток узлов."""
        return self.node_label_to_id

    def get_edge_encodings(self) -> Dict[str, int]:
        """Возвращает текущие кодировки меток ребер."""
        return self.edge_label_to_id

    def clear(self):
        """
        Очищает словари кодирования и сбрасывает счетчики ID.
        Полезно для повторного использования экземпляра для новой независимой кодировки.
        """
        self.node_label_to_id = {None: 0}
        self.edge_label_to_id = {}
        self.next_node_label_id = 1
        self.next_edge_label_id = 1


def semantic_graphs_to_gsofia_format(
    semantic_graphs: List[SemanticGraph],
    output_path: str,
    encoder: LabelEncoder # Теперь принимаем экземпляр LabelEncoder
) -> None:
    """
    Конвертирует список объектов SemanticGraph в формат, требуемый gSOFIA,
    используя тройное кодирование для сохранения направленности ребра.
    """
    with open(output_path, 'w') as f:
        for graph_index, graph in enumerate(semantic_graphs):
            f.write(f"t # {graph_index}\n")

            node_id_map: Dict[Union[int, str], int] = {}
            for local_node_index, node in enumerate(graph.nodes):
                node_id_map[node.id] = local_node_index
                # Используем метод encode_node_label из переданного encoder'а
                encoded_label = encoder.encode_node_label(node.label)
                f.write(f"v {local_node_index} {encoded_label}\n")

            for edge in graph.edges:
                source_id_mapped = node_id_map.get(edge.source)
                target_id_mapped = node_id_map.get(edge.target)

                if source_id_mapped is not None and target_id_mapped is not None:
                    source = graph.get_node_by_id(edge.source)
                    target = graph.get_node_by_id(edge.target)

                    # Убедитесь, что метки узлов не None, прежде чем передавать их в тройку
                    # По умолчанию "EMP" если метка узла None, как в оригинальном коде
                    source_label = "EMP" if source is None or source.label is None else source.label
                    target_label = "EMP" if target is None or target.label is None else target.label
                    edge_label = edge.label if edge.label is not None else ""

                    # Используем метод encode_edge_label_triple из переданного encoder'а
                    encoded_edge_label = encoder.encode_edge_label_triple(source_label, edge_label, target_label)
                    f.write(f"e {source_id_mapped} {target_id_mapped} {encoded_edge_label}\n")


def save_encoder(encoder: LabelEncoder, filepath: str):
    """
    Сохраняет объект LabelEncoder в файл pickle.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"LabelEncoder saved to {filepath} using pickle.")


def load_encoder(filepath: str) -> LabelEncoder:
    """
    Загружает объект LabelEncoder из файла pickle.
    """
    with open(filepath, 'rb') as f:
        encoder = pickle.load(f)
    print(f"LabelEncoder loaded from {filepath} using pickle.")
    return encoder