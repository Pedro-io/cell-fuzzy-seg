"""Execução de um pipeline em blocos sobrepostos, para imagens maiores que a entrada de treino.

Quando o treino usa recortes (``PreprocessedDataset(crop_size=...)``), a avaliação da imagem inteira
é feita por blocos do mesmo tamanho: a imagem é dividida em blocos que se sobrepõem, cada bloco
passa pelo pipeline sozinho, e as saídas são coladas de volta por média ponderada com uma janela de
Hann (peso menor perto da borda de cada bloco, onde a rede vê menos contexto).
"""

from typing import Any, Dict, List, Sequence

import numpy as np
import torch


def tile_starts(length: int, tile: int, stride: int) -> List[int]:
    """Posições iniciais dos blocos ao longo de um eixo, cobrindo até a borda.

    Os blocos andam ``stride`` pixels; o último é encostado na borda, mesmo que isso o deixe mais
    perto do anterior. Ex.: ``tile_starts(1000, 256, 192) == [0, 192, 384, 576, 744]``.
    """
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def hann_window(tile: int, floor: float) -> np.ndarray:
    """Janela de Hann 2D ``(tile, tile)``, com mínimo ``floor`` (a borda da imagem também pesa)."""
    w = np.hanning(tile)
    return np.maximum(np.outer(w, w), floor).astype(np.float32)


class TiledPipeline:
    """Envolve um pipeline e o executa em blocos quando a imagem é maior que ``tile_size``.

    Imagens do tamanho do bloco (ou menores) passam direto pelo pipeline: é o caso do treino com
    recortes. Imagens maiores são divididas em blocos ``tile_size × tile_size`` que andam
    ``tile_size × (1 − overlap)`` pixels; todos os tensores ``(B, C, H, W)`` do dicionário de
    entrada são recortados juntos, e as chaves ``stitch_keys`` da saída são coladas de volta.

    O objeto se comporta como o pipeline envolvido para quem o usa (``run``, ``steps``), então
    ``Trainer``, ``evaluate``, ``run_kfold`` e as figuras funcionam sem mudança.

    Args:
        pipeline: Pipeline com ``run(data, verbose)`` e ``steps``.
        tile_size: Lado do bloco, em pixels (o mesmo dos recortes do treino).
        overlap: Fração do lado compartilhada entre blocos vizinhos, em ``[0, 1)``.
        window_floor: Peso mínimo da janela de Hann.
        stitch_keys: Chaves da saída coladas de volta na imagem inteira.

    Raises:
        ValueError: Se ``tile_size`` ou ``overlap`` forem inválidos.
    """

    def __init__(self, pipeline: Any, tile_size: int, overlap: float = 0.25,
                 window_floor: float = 0.05,
                 stitch_keys: Sequence[str] = ("segmentation", "markers")) -> None:
        if tile_size <= 0:
            raise ValueError(f"tile_size deve ser positivo, recebido {tile_size}.")
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap deve estar em [0, 1), recebido {overlap}.")
        self.pipeline = pipeline
        self.tile_size = tile_size
        self.stride = max(int(round(tile_size * (1.0 - overlap))), 1)
        self.window_floor = window_floor
        self.stitch_keys = tuple(stitch_keys)

    @staticmethod
    def _is_image(value: Any, h: int, w: int) -> bool:
        return (isinstance(value, torch.Tensor) and value.ndim == 4
                and tuple(value.shape[-2:]) == (h, w))

    @property
    def steps(self):
        return self.pipeline.steps

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(data)

    def run(self, data: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
        """Executa o pipeline, em blocos se a imagem for maior que ``tile_size``.

        Args:
            data: Dicionário do batch; ``data["image"]`` é um tensor ``(B, C, H, W)``.
            verbose: Repassado ao pipeline.

        Returns:
            O dicionário de entrada acrescido das chaves de saída; em blocos, as ``stitch_keys`` têm
            o tamanho da imagem inteira.

        Raises:
            ValueError: Se a imagem for maior que o bloco num eixo e menor no outro.
        """
        h, w = data["image"].shape[-2:]
        if h <= self.tile_size and w <= self.tile_size:
            return self.pipeline.run(data, verbose=verbose)
        if min(h, w) < self.tile_size:
            raise ValueError(f"Imagem {h} × {w} menor que o bloco ({self.tile_size}) num dos eixos.")

        t = self.tile_size
        weight_np = hann_window(t, self.window_floor)
        sums: Dict[str, torch.Tensor] = {}
        total = None
        for y in tile_starts(h, t, self.stride):
            for x in tile_starts(w, t, self.stride):
                tile_data = {
                    key: (value[..., y:y + t, x:x + t] if self._is_image(value, h, w) else value)
                    for key, value in data.items()
                }
                out = self.pipeline.run(tile_data, verbose=verbose)
                for key in self.stitch_keys:
                    piece = out[key]
                    if key not in sums:
                        sums[key] = piece.new_zeros(*piece.shape[:-2], h, w)
                        if total is None:
                            total = piece.new_zeros(h, w)
                            weight = torch.from_numpy(weight_np).to(piece)
                    sums[key][..., y:y + t, x:x + t] += piece * weight
                total[y:y + t, x:x + t] += weight
        result = dict(data)
        for key, value in sums.items():
            result[key] = value / total
        return result
