#!/usr/bin/env python3
"""
Script simples para testar o pipeline completo Cellpose → MarkerNet → Watershed.

Uso:
    python pipeline_test.py [--sample-idx 0] [--visualize]

Este script:
1. Carrega uma amostra do MonusegDataset
2. Executa o pipeline completo em-memória
3. Mostra estatísticas dos resultados
4. (Opcional) Visualiza com matplotlib
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

# Adicionar project root ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.load.monuseg_dataset import MonusegDataset
from src.pipeline.model_pipeline import ModelPipeline
from src.pipeline.steps.cellpose_step import CellposeStep
from src.pipeline.steps.rgba_step import RGBAStep
from src.pipeline.steps.marker_step import MarkerStep
from src.pipeline.steps.segmentation_step import SegmentationStep
from src.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Teste do pipeline Cellpose → MarkerNet → Watershed"
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Índice da amostra no dataset (padrão: 0)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualizar resultados com matplotlib"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Número de amostras para processar (padrão: 1)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TESTE DO PIPELINE: Cellpose → MarkerNet → Watershed")
    print("=" * 70)

    # 1. Verificar GPU
    print(f"\n🔧 Verificação de Ambiente:")
    print(f"   PyTorch device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"   GPU disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 2. Carregar dataset
    print(f"\n📊 Carregando Dataset...")
    try:
        dataset = MonusegDataset(
            dataset_name="monuseg",
            config_key="monuseg_test",
            transform=None
        )
        print(f"   ✅ Dataset carregado: {len(dataset)} amostras")
    except Exception as e:
        print(f"   ❌ Erro ao carregar dataset: {e}")
        return 1

    # 3. Montar pipeline
    print(f"\n🔗 Montando Pipeline...")
    try:
        pipeline = ModelPipeline([
            CellposeStep(batch_size=1),
            RGBAStep(),
            MarkerStep(model=None),  # Sem MarkerNet por enquanto
            SegmentationStep(use_distance_map=True),
        ])
        print(f"   ✅ Pipeline criado com {len(pipeline.steps)} passos")
        for i, step in enumerate(pipeline.steps, 1):
            print(f"      {i}. {step.name}")
    except Exception as e:
        print(f"   ❌ Erro ao criar pipeline: {e}")
        return 1

    # 4. Processar amostras
    print(f"\n⚙️  Processando {args.num_samples} amostra(s)...")
    print("-" * 70)

    stats = {
        'successful': 0,
        'failed': 0,
        'cellpose_cells': [],
        'final_cells': [],
        'processing_time': [],
    }

    for sample_idx in range(args.sample_idx, min(args.sample_idx + args.num_samples, len(dataset))):
        try:
            sample = dataset[sample_idx]
            sample_id = sample['id']

            print(f"\n📌 Amostra {sample_idx}: {sample_id}")
            print(f"   Imagem shape: {sample['image'].shape}")

            # Preparar input
            pipeline_input = {
                'image': sample['image'],
                'id': sample_id
            }

            # Executar pipeline
            result = pipeline.forward(pipeline_input, verbose=False)

            # Extrair resultados
            cellpose_seg = result.get('segmentation', np.array([]))
            final_seg = result['segmentation']
            markers = result['markers']

            # Computar estatísticas
            n_cellpose = np.max(cellpose_seg) if cellpose_seg.size > 0 else 0
            n_final = np.max(final_seg) if final_seg.size > 0 else 0
            n_markers = np.sum(markers > 0.5)

            stats['cellpose_cells'].append(n_cellpose)
            stats['final_cells'].append(n_final)
            stats['successful'] += 1

            # Mostrar resultados
            print(f"   ✅ Sucesso!")
            print(f"      - Cellpose: {n_cellpose} células")
            print(f"      - Markers: {n_markers} regiões")
            print(f"      - Final: {n_final} células (refinado)")

            if 'ground_truth' in sample:
                n_gt = np.sum(sample['ground_truth'] > 0) if sample['ground_truth'].ndim == 2 else len(np.unique(sample['ground_truth'])) - 1
                print(f"      - Ground Truth: {n_gt} células (referência)")

        except Exception as e:
            print(f"\n   ❌ Erro: {e}")
            stats['failed'] += 1
            import traceback
            traceback.print_exc()

    # 5. Resumo
    print(f"\n" + "=" * 70)
    print("RESUMO DOS RESULTADOS")
    print("=" * 70)
    print(f"Processadas: {stats['successful']} amostras com sucesso")
    print(f"Falhadas: {stats['failed']} amostras")

    if stats['cellpose_cells']:
        print(f"\n📊 Estatísticas de Células:")
        print(f"   Cellpose (médio): {np.mean(stats['cellpose_cells']):.1f} ± {np.std(stats['cellpose_cells']):.1f}")
        print(f"   Final (médio): {np.mean(stats['final_cells']):.1f} ± {np.std(stats['final_cells']):.1f}")
        print(f"   Diferença (final - cellpose): {np.mean(np.array(stats['final_cells']) - np.array(stats['cellpose_cells'])):+.1f}")

    # 6. Visualizar (opcional)
    if args.visualize and stats['successful'] > 0:
        print(f"\n📈 Gerando visualização...")
        try:
            import matplotlib.pyplot as plt

            sample = dataset[args.sample_idx]
            result = pipeline.forward({
                'image': sample['image'],
                'id': sample['id']
            }, verbose=False)

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Normalizações
            def norm_display(img):
                if img.dtype == np.uint8:
                    return img.astype(np.float32) / 255.0
                elif img.max() > 0:
                    return (img - img.min()) / (img.max() - img.min())
                return img

            axes[0, 0].imshow(norm_display(sample['image']), cmap='gray')
            axes[0, 0].set_title("Imagem Original")
            axes[0, 0].axis('off')

            cellpose_seg = np.zeros_like(sample['image'], dtype=np.uint8)  # Placeholder
            axes[0, 1].imshow(norm_display(sample['image']), cmap='gray')
            axes[0, 1].set_title("Cellpose Segmentation")
            axes[0, 1].axis('off')

            rgba = result['rgba']
            axes[0, 2].imshow(rgba)
            axes[0, 2].set_title("RGBA")
            axes[0, 2].axis('off')

            markers = result['markers']
            axes[1, 0].imshow(markers, cmap='gray')
            axes[1, 0].set_title("Markers")
            axes[1, 0].axis('off')

            final_seg = result['segmentation']
            axes[1, 1].imshow(final_seg, cmap='nipy_spectral')
            axes[1, 1].set_title("Final Segmentation")
            axes[1, 1].axis('off')

            if 'ground_truth' in sample:
                axes[1, 2].imshow(sample['ground_truth'], cmap='gray')
                axes[1, 2].set_title("Ground Truth")
            else:
                axes[1, 2].text(0.5, 0.5, "Ground Truth\nNot Available", ha='center', va='center')
            axes[1, 2].axis('off')

            plt.tight_layout()
            output_path = Path(project_root) / "pipeline_output.png"
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"   ✅ Visualização salva: {output_path}")
            plt.show()

        except ImportError:
            print(f"   ⚠️  Matplotlib não disponível. Pulando visualização.")
        except Exception as e:
            print(f"   ❌ Erro na visualização: {e}")

    print(f"\n{'=' * 70}")
    return 0 if stats['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
