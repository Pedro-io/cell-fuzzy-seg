import os
from typing import Any, Dict

import numpy as np
from cellpose import core, models
from scipy.special import expit

from src.utils.logger import logger

from ..base_step import PipelineStep


class CellposeStep(PipelineStep):
    """Passo do pipeline que aplica a segmentação do Cellpose a uma imagem de entrada.

    Este passo inicializa um único CellposeModel na GPU e reutiliza-o para todas
    as chamadas subsequentes a ``forward``.

    Atributos:
        model: Instância carregada de ``CellposeModel`` usando inferência na GPU.
        batch_size: Número de imagens processadas por batch de inferência.
        diam_mean: Diâmetro estimado dos objetos passado ao Cellpose.
        cellprob_threshold: Limiar de probabilidade celular para seleção da máscara.
        flow_threshold: Limiar de fluxo para rastreamento do Cellpose.
        min_size: Tamanho mínimo de objeto a ser mantido na máscara final.
    """

    def __init__(
        self, batch_size: int = 8,
        name: str = "CellposeStep",
        pretrained_model: str = "cpsam",
        diam_mean: float = 30.0,
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.2,
        min_size: int = 4
        ) -> None:
        """Cria um CellposeStep e verifica a disponibilidade da GPU.

        Args:
            batch_size: Número de imagens por batch de inferência.
            name: Identificador deste passo no pipeline.
            pretrained_model: Nome de um modelo do Cellpose (no Cellpose 4, ``"cpsam"`` ou um
                modelo registrado pelo usuário) ou caminho de um arquivo de modelo. Um nome
                desconhecido levanta ``ValueError``: sem a checagem, o Cellpose trocaria o
                modelo pelo ``cpsam`` só com um aviso, mudando a entrada da MarkerNet e a linha
                de base.
            diam_mean: Diâmetro médio das células para segmentação.
            cellprob_threshold: Limiar aplicado à probabilidade celular do Cellpose.
            flow_threshold: Limiar aplicado às saídas de fluxo do Cellpose.
            min_size: Tamanho mínimo de instância a ser mantido na máscara de segmentação.

        Raises:
            ValueError: Se ``pretrained_model`` não for um modelo conhecido nem um arquivo existente.
            RuntimeError: Se uma GPU compatível com CUDA não estiver disponível.
        """
        super().__init__(name=name)
        self._validate_model_name(pretrained_model)

        if not core.use_gpu():
            raise RuntimeError("GPU is required but not available.")

        self.model = models.CellposeModel(gpu=True, pretrained_model=pretrained_model, diam_mean=diam_mean)
        self.batch_size = batch_size
        self.diam_mean = diam_mean
        self.cellprob_threshold = cellprob_threshold
        self.flow_threshold = flow_threshold
        self.min_size = min_size
        logger.info(f"[CellposeStep] Running on GPU (model={pretrained_model})")

    @staticmethod
    def _validate_model_name(pretrained_model: str) -> None:
        """Levanta ``ValueError`` se o Cellpose não conhecer o modelo solicitado.

        Segue a mesma regra do ``CellposeModel`` (``cellpose/models.py``, v4.1.1): o nome é aceito
        se for um arquivo existente ou se estiver em ``MODEL_NAMES + get_user_models()``. Fora
        disso, a biblioteca usaria o ``cpsam`` e só registraria um aviso, que ainda mostra o
        caminho do modelo padrão em vez do nome pedido.

        Args:
            pretrained_model: Nome ou caminho do modelo solicitado.

        Raises:
            ValueError: Se o modelo não for conhecido nem existir como arquivo.
        """
        if os.path.exists(pretrained_model):
            return

        known = list(models.MODEL_NAMES) + list(models.get_user_models())
        if pretrained_model not in known:
            raise ValueError(
                f"[CellposeStep] Modelo do Cellpose desconhecido: {pretrained_model!r}. "
                f"Modelos disponíveis: {known}. Informe um desses nomes ou o caminho de um "
                f"arquivo de modelo."
            )

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Executa a segmentação do Cellpose e anexa os resultados aos dados de entrada.

        Args:
            data: Dicionário contendo pelo menos a chave ``"image"`` com um array
                NumPy de formato ``(H, W)`` ou ``(H, W, C)``.

        Returns:
            O mesmo dicionário ``data`` com chaves adicionadas:
                - ``"segmentation"``: array da máscara de instâncias com formato ``(H, W)``.
                - ``"cellpose_prob"``: probabilidade de cada pixel ser célula, ``float16``
                  ``(H, W)`` em ``[0, 1]``. É a sigmoide do mapa ``flows[2]`` do Cellpose, que é
                  um logit (a rede é treinada com ``BCEWithLogitsLoss``); por isso
                  ``cellprob_threshold=0`` corresponde a probabilidade 0,5. A máscara final não
                  sai só deste mapa: ela também depende dos fluxos, do ``flow_threshold`` e do
                  ``min_size``.

        Raises:
            KeyError: Se a chave ``"image"`` estiver ausente em ``data``.
            ValueError: Se o mapa de probabilidade não tiver o formato da máscara.
        """
        if "image" not in data:
            raise KeyError("Input data must contain 'image'")

        image = data["image"]

        masks, flows, _styles = self.model.eval(
            image,
            batch_size=self.batch_size,
            diameter=self.diam_mean,
            cellprob_threshold=self.cellprob_threshold,
            flow_threshold=self.flow_threshold,
            min_size=self.min_size,
        )

        cellprob_logit = np.asarray(flows[2])
        if cellprob_logit.shape != masks.shape:
            raise ValueError(
                f"[CellposeStep] Mapa de probabilidade com formato {cellprob_logit.shape}, "
                f"diferente da máscara {masks.shape}."
            )

        data["segmentation"] = masks
        data["cellpose_prob"] = expit(cellprob_logit).astype(np.float16)

        return data
