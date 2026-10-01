"""Testes da conversão XML → máscara do MonusegDataset.

Regras testadas: rasterização pelo centro do pixel (convenção de canto), descarte de regiões
degeneradas e "o menor vence" na sobreposição.

O ``monuseg_dataset`` importa ``cellpose.io`` só para ler imagens; o Cellpose não é necessário
aqui, então um módulo falso é instalado quando ele não existe.
"""

import os
import sys
import types

import numpy as np
import pytest

if "cellpose" not in sys.modules:
    try:
        import cellpose  # noqa: F401
    except ImportError:
        fake = types.ModuleType("cellpose")
        fake.io = types.SimpleNamespace(imread=None)
        sys.modules["cellpose"] = fake

from src.data.load.monuseg_dataset import (  # noqa: E402
    polygon_area,
    read_xml_regions,
    xml_to_instance_mask,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_xml(tmp_path, regions, name="amostra.xml"):
    """Escreve um XML no formato do MoNuSeg com uma região por lista de vértices ``(x, y)``."""
    body = []
    for k, vertices in enumerate(regions, start=1):
        verts = "".join(f'<Vertex X="{x}" Y="{y}" Z="0"/>' for x, y in vertices)
        body.append(f'<Region Id="{k}"><Vertices>{verts}</Vertices></Region>')
    xml = (
        '<Annotations MicronsPerPixel="0.25"><Annotation Id="1"><Regions>'
        + "".join(body)
        + "</Regions></Annotation></Annotations>"
    )
    path = tmp_path / name
    path.write_text(xml)
    return str(path)


def square(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_square_on_pixel_corners_has_exact_area(tmp_path):
    # Quadrado de canto (0,0) a (4,4): cobre exatamente os pixels 0..3 → 16 px = área do polígono.
    # Pintar também a borda, como faz o cv2.fillPoly, daria 5×5 = 25 px.
    path = write_xml(tmp_path, [square(0, 0, 4, 4)])

    instances = xml_to_instance_mask(path, (10, 10))

    expected = np.zeros((10, 10), dtype=np.int32)
    expected[0:4, 0:4] = 1
    np.testing.assert_array_equal(instances, expected)
    assert instances.dtype == np.int32


def test_degenerate_regions_are_discarded(tmp_path):
    # Regiões degeneradas como as que aparecem no MoNuSeg: 2 vértices e área 0; e 3 vértices colineares.
    path = write_xml(
        tmp_path,
        [[(3.3, 4.0), (3.6, 4.0)], [(1, 1), (2, 2), (3, 3)], square(5, 5, 8, 8)],
    )

    instances = xml_to_instance_mask(path, (10, 10))

    assert instances.max() == 1  # só o quadrado vira núcleo
    assert (instances == 1).sum() == 9


def test_smaller_nucleus_wins_on_overlap(tmp_path):
    # O grande vem primeiro no XML; com "o último vence" o pequeno sobreviveria por acaso da ordem.
    # Aqui o pequeno vem primeiro, e mesmo assim tem de manter todos os seus pixels.
    small = square(4, 4, 7, 7)   # 9 px
    big = square(0, 0, 6, 6)     # 36 px, cobre 4 px do pequeno
    path = write_xml(tmp_path, [small, big])

    instances = xml_to_instance_mask(path, (10, 10))

    assert (instances == 1).sum() == 9            # rótulo 1 = primeira região do XML (o pequeno)
    assert (instances == 2).sum() == 36 - 4       # o grande perde só a sobreposição
    assert set(np.unique(instances)) == {0, 1, 2}


def test_labels_are_consecutive_in_xml_order(tmp_path):
    regions = [square(0, 0, 2, 2), [(9, 9), (9.2, 9)], square(4, 4, 6, 6), square(7, 0, 9, 2)]
    path = write_xml(tmp_path, regions)

    instances = xml_to_instance_mask(path, (10, 10))

    # A região degenerada não consome rótulo: 3 núcleos → rótulos 1, 2, 3 na ordem do XML.
    assert instances[0, 0] == 1
    assert instances[4, 4] == 2
    assert instances[0, 7] == 3
    assert instances.max() == 3


def test_polygon_area_matches_shoelace():
    assert polygon_area(np.array(square(0, 0, 4, 3), dtype=float)) == pytest.approx(12.0)
    assert polygon_area(np.array([(0, 0), (4, 0), (0, 3)], dtype=float)) == pytest.approx(6.0)


def test_read_xml_regions_reads_all_annotation_blocks(tmp_path):
    xml = (
        "<Annotations>"
        '<Annotation Id="1"><Regions><Region Id="1"><Vertices>'
        '<Vertex X="0.5" Y="1.25"/><Vertex X="2" Y="1"/><Vertex X="2" Y="3"/>'
        "</Vertices></Region></Regions></Annotation>"
        '<Annotation Id="2"><Regions><Region Id="2"><Vertices>'
        '<Vertex X="5" Y="5"/><Vertex X="6" Y="5"/><Vertex X="6" Y="6"/>'
        "</Vertices></Region></Regions></Annotation>"
        "</Annotations>"
    )
    path = tmp_path / "dois_blocos.xml"
    path.write_text(xml)

    regions = read_xml_regions(str(path))

    assert len(regions) == 2
    np.testing.assert_allclose(regions[0][0], [0.5, 1.25])


REAL_XML = os.path.join(
    REPO_ROOT, "data_source", "MoNuSegTestData", "Annotations", "TCGA-2Z-A9J9-01A-01-TS1.xml"
)


@pytest.mark.skipif(not os.path.exists(REAL_XML), reason="dados do MoNuSeg ausentes")
def test_real_xml_area_matches_annotation():
    # Numa imagem real, a área rasterizada fica a ~1% da área anotada (pintando a borda, ficaria ~10% acima),
    # e todo núcleo válido ganha um rótulo.
    regions = read_xml_regions(REAL_XML)
    valid = [r for r in regions if len(r) >= 3 and polygon_area(r) > 0]

    instances = xml_to_instance_mask(REAL_XML, (1000, 1000))

    ratio = (instances > 0).sum() / sum(polygon_area(r) for r in valid)
    assert 0.97 < ratio < 1.03
    assert instances.max() == len(valid)
