from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.logger import logger

from ..base_step import PipelineStep


class MarkerStep(PipelineStep):
    """Passo do pipeline que gera o marcador difuso com a MarkerUNet.

    Recebe a entrada ``rgba`` (imagem RGB e máscara do Cellpose, 4 canais em ``[0, 1]``), executa a
    MarkerUNet, aplica a sigmoide e grava o marcador em ``markers``: um tensor ``(N, 1, H, W)`` com
    valores em ``[0, 1]``, do tamanho da entrada.

    O marcador nunca é binarizado e o grafo computacional é preservado (sem ``torch.no_grad()``),
    para que o gradiente da perda, calculada sobre a segmentação da rede final congelada, chegue
    aos pesos da MarkerUNet. Na avaliação, quem desliga o gradiente é o chamador
    (:func:`src.evaluation.metrics.evaluate`).

    Se a entrada não tiver ``target_size × target_size``, ela é reduzida para esse tamanho por
    interpolação bilinear antes da rede, e o marcador é ampliado de volta da mesma forma. No treino
    por recortes, o recorte já tem o tamanho alvo e passa sem redução.

    O modo da rede (``train()``/``eval()``, que muda o comportamento da BatchNorm) começa em
    ``train()`` e é alternado por :meth:`set_training`, chamado pelo ``Trainer`` e pelo ``evaluate``.

    Atributos:
        model: MarkerUNet (objeto com o atributo ``model``, o ``nn.Module`` executado).
        target_size: Lado da entrada da rede, em pixels.
        device: Dispositivo PyTorch (cpu ou cuda).
        name: Identificador deste passo no pipeline.
    """

    def __init__(
        self,
        model: Any,
        target_size: int = 256,
        device: Optional[str] = None,
        name: str = "MarkerStep",
        differentiable: bool = True,
    ):
        """Inicializa o MarkerStep.

        Args:
            model: MarkerUNet (objeto com o atributo ``model``, o ``nn.Module`` executado).
            target_size: Lado da entrada da rede, em pixels.
            device: Dispositivo PyTorch. Se ``None``, usa a GPU se disponível.
            name: Identificador deste passo no pipeline.
            differentiable: Mantido para compatibilidade com os notebooks; só ``True`` é aceito,
                porque o passo tem um único modo, o diferenciável.

        Raises:
            ValueError: Se ``model`` for ``None`` ou ``differentiable`` for ``False``.
        """
        super().__init__(name=name)
        if model is None:
            raise ValueError(f"[{name}] model é obrigatório: passe a MarkerUNet.")
        if not differentiable:
            raise ValueError(
                f"[{name}] differentiable=False não é suportado: o passo só tem o modo diferenciável. "
                "Para avaliar sem gradiente, use src.evaluation.metrics.evaluate."
            )
        self.model = model
        self.target_size = target_size
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.model.to(self.device)
        self.model.model.train()

        logger.debug(f"[{self.name}] Initialized on {self.device}")

    def set_training(self, training: bool) -> None:
        """Alterna a MarkerUNet entre treino (BatchNorm com as estatísticas do lote) e avaliação
        (BatchNorm com as médias acumuladas)."""
        self.model.model.train(training)

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Gera o marcador a partir da entrada RGBA.

        Args:
            data: Dicionário contendo ``"rgba"``: tensor ``(N, 4, H, W)`` (ou ``(N, H, W, 4)``,
                ``(4, H, W)``, ``(H, W, 4)``) ou array NumPy ``(H, W, 4)``, com valores em ``[0, 1]``
                ou em ``[0, 255]`` (ver :meth:`_to_tensor_bchw`).

        Returns:
            O mesmo dicionário, acrescido de ``"markers"``: tensor ``(N, 1, H, W)`` com valores em
            ``[0, 1]`` e grafo preservado.

        Raises:
            KeyError: Se ``"rgba"`` estiver ausente.
            RuntimeError: Se a execução da rede falhar.
        """
        if "rgba" not in data:
            raise KeyError("Missing 'rgba' in data. Ensure RGBAStep runs before MarkerStep.")

        try:
            data["markers"] = self._forward_differentiable(data["rgba"])
            logger.debug(f"[{self.name}] Generated markers with shape {tuple(data['markers'].shape)}")
        except Exception as e:
            logger.error(f"[{self.name}] Inference failed: {e}")
            raise RuntimeError(f"MarkerNet inference failed: {e}") from e

        return data

    def _forward_differentiable(self, rgba: Any) -> torch.Tensor:
        """Executa a rede preservando o grafo e devolve o marcador.

        Chama o ``nn.Module`` direto (e não um ``predict`` com ``torch.no_grad()``) e usa
        ``F.interpolate``, que é diferenciável, para reduzir e ampliar.

        Args:
            rgba: Entrada RGBA (formatos aceitos em :meth:`_to_tensor_bchw`).

        Returns:
            Tensor ``(N, 1, H, W)`` com valores em ``[0, 1]``.
        """
        rgba_tensor = self._to_tensor_bchw(rgba)  # (N, 4, H, W) em [0, 1], no device
        original_size = tuple(rgba_tensor.shape[-2:])

        if original_size != (self.target_size, self.target_size):
            rgba_resized = F.interpolate(
                rgba_tensor,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            rgba_resized = rgba_tensor

        logits = self.model.model(rgba_resized)  # (N, 1, target_size, target_size)
        probs = torch.sigmoid(logits)

        if tuple(probs.shape[-2:]) != original_size:
            probs = F.interpolate(
                probs,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )

        return probs  # (N, 1, H, W)

    def _to_tensor_bchw(self, rgba: Any) -> torch.Tensor:
        """Converte a entrada RGBA para um tensor ``(N, C, H, W)`` float32 em ``[0, 1]``.

        Aceita arrays NumPy ``(H, W, C)`` e tensores ``(N, C, H, W)``, ``(N, H, W, C)``,
        ``(C, H, W)`` ou ``(H, W, C)``, com ``C`` igual a 3 ou 4. Se algum valor do lote passar de 1,
        o lote inteiro é tratado como estando em ``[0, 255]`` e dividido por 255. Os dados
        pré-processados já chegam em ``[0, 1]``, então essa divisão não ocorre neles.

        Args:
            rgba: Entrada RGBA (numpy ou tensor).

        Returns:
            Tensor float32 ``(N, C, H, W)`` no device configurado.

        Raises:
            ValueError: Se o número de canais não for 3 nem 4.
        """
        if isinstance(rgba, torch.Tensor):
            t = rgba.float()
            if t.ndim == 4 and t.shape[-1] in (3, 4):
                t = t.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            elif t.ndim == 3:
                # Com H ou W iguais a 3 ou 4 a forma é ambígua: assume-se canais primeiro quando o
                # primeiro eixo tem 3 ou 4 elementos e o último não.
                if t.shape[0] in (3, 4) and t.shape[-1] not in (3, 4):
                    t = t.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
                else:
                    t = t.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
        else:
            arr = np.asarray(rgba)
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                arr = np.transpose(arr, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            t = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0)

        if t.ndim != 4 or t.shape[1] not in (3, 4):
            raise ValueError(f"Esperado rgba com 3 ou 4 canais, recebido formato {tuple(t.shape)}.")

        if t.max() > 1.0:
            t = t / 255.0

        return t.to(self.device)
