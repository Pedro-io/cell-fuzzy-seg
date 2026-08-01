from typing import Any, Dict

from cellpose import core, models

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
        pretreined_model: str = "cpsam_v2",
        diam_mean: float = 30.0,
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.2,
        min_size: int = 4
        ) -> None:
        """Cria um CellposeStep e verifica a disponibilidade da GPU.

        Args:
            batch_size: Número de imagens por batch de inferência.
            name: Identificador deste passo no pipeline.
            pretreined_model: Nome do modelo pré-treinado do Cellpose a carregar.
            diam_mean: Diâmetro médio das células para segmentação.
            cellprob_threshold: Limiar aplicado à probabilidade celular do Cellpose.
            flow_threshold: Limiar aplicado às saídas de fluxo do Cellpose.
            min_size: Tamanho mínimo de instância a ser mantido na máscara de segmentação.

        Raises:
            RuntimeError: Se uma GPU compatível com CUDA não estiver disponível.
        """
        super().__init__(name=name)
        if not core.use_gpu():
            raise RuntimeError("GPU is required but not available.")

        self.model = models.CellposeModel(gpu=True, pretrained_model=pretreined_model, diam_mean=diam_mean)
        self.batch_size = batch_size
        self.diam_mean = diam_mean
        self.cellprob_threshold = cellprob_threshold
        self.flow_threshold = flow_threshold
        self.min_size = min_size
        logger.info("[CellposeStep] Running on GPU")

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Executa a segmentação do Cellpose e anexa os resultados aos dados de entrada.

        Args:
            data: Dicionário contendo pelo menos a chave ``"image"`` com um array
                NumPy de formato ``(H, W)`` ou ``(H, W, C)``.

        Returns:
            O mesmo dicionário ``data`` com chaves adicionadas:
                - ``"segmentation"``: array da máscara de instâncias com formato ``(H, W)``.
                - ``"flows"``: saídas do campo de fluxo do Cellpose.
                - ``"styles"``: vetores de estilo do Cellpose.

        Raises:
            KeyError: Se a chave ``"image"`` estiver ausente em ``data``.
        """
        if "image" not in data:
            raise KeyError("Input data must contain 'image'")

        image = data["image"]

        masks, flows, styles = self.model.eval(
            image,
            batch_size=self.batch_size,
            diameter=self.diam_mean,
            cellprob_threshold=self.cellprob_threshold,
            flow_threshold=self.flow_threshold,
            min_size=self.min_size,
        )

        data["segmentation"] = masks
        data["flows"] = flows
        data["styles"] = styles

        return data
