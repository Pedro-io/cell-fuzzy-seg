from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.logger import logger

from ..base_step import PipelineStep


class MarkerStep(PipelineStep):
    """Passo do pipeline que gera marcadores celulares usando uma rede neural.

    Usa um modelo MarkerNet pré-treinado para refinar máscaras de segmentação em
    marcadores celulares precisos. Espera entrada RGBA de 4 canais (imagem RGB + segmentação).

    Possui **dois modos**:

    - ``differentiable=False`` (padrão, inferência): executa a MarkerNet sob
      ``torch.no_grad()``, redimensiona a saída para o tamanho original e
      binariza com um limiar, retornando um array NumPy ``(H, W)``.
    - ``differentiable=True`` (treinamento): executa a MarkerNet mantendo o grafo
      computacional (sem ``no_grad``), usando interpolação diferenciável
      (``F.interpolate``) e **sem binarizar**, retornando um tensor
      ``(1, 1, H, W)`` com probabilidades em ``[0, 1]``. Isso permite que o
      gradiente da loss (via rede final congelada) flua até a MarkerNet.

    Atributos:
        model: Modelo MarkerNet carregado para inferência.
        target_size: Tamanho para redimensionar a entrada para inferência do modelo (padrão: 256).
        threshold: Limiar de probabilidade para binarizar a saída dos marcadores (padrão: 0.5).
        device: Dispositivo PyTorch para inferência (cpu ou cuda).
        differentiable: Se ``True``, o forward preserva o gradiente e retorna um tensor.
        name: Identificador deste passo no pipeline.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        target_size: int = 256,
        threshold: float = 0.5,
        device: Optional[str] = None,
        name: str = "MarkerStep",
        differentiable: bool = False,
    ):
        """Inicializa MarkerStep.

        Args:
            model: Modelo MarkerNet pré-treinado. Se None, a inferência é pulada.
            target_size: Altura/largura para redimensionar a entrada para inferência.
            threshold: Limiar de probabilidade para binarização (0.0 a 1.0).
            device: Dispositivo PyTorch. Se None, usa GPU se disponível.
            name: Identificador deste passo no pipeline.
            differentiable: Se ``True``, o forward retorna um tensor diferenciável
                ``(1, 1, H, W)`` (modo treinamento). Padrão ``False`` (inferência).
        """
        super().__init__(name=name)
        self.model = model
        self.target_size = target_size
        self.threshold = threshold
        self.differentiable = differentiable

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.model is not None:
            self.model.model.to(self.device)
            # Mantém eval() em ambos os modos: a diferenciabilidade não depende de
            # train/eval, e eval() evita instabilidade de BatchNorm com batches
            # pequenos (típicos de teste com 1 imagem).
            self.model.model.eval()

        logger.info(
            f"[{self.name}] Initialized on {self.device} (differentiable={differentiable})"
        )

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Gera marcadores a partir de uma imagem RGBA usando MarkerNet.

        Args:
            data: Dicionário contendo:
                - ``"rgba"``: array uint8 com formato ``(H, W, 4)`` (RGB + alpha)
                  ou tensor ``(N, 4, H, W)``.

        Returns:
            O mesmo dicionário ``data``, agora estendido com:
                - ``"markers"``: no modo inferência, array float ``(H, W)``
                  binarizado; no modo diferenciável, tensor ``(1, 1, H, W)``
                  com probabilidades em ``[0, 1]`` e grafo preservado.

        Raises:
            KeyError: Se as chaves obrigatórias estiverem ausentes e o modelo estiver carregado.
            RuntimeError: Se a inferência do modelo falhar.
        """
        if "rgba" not in data:
            raise KeyError("Missing 'rgba' in data. Ensure RGBAStep runs before MarkerStep.")

        # Skip if model is not provided
        if self.model is None:
            logger.warning(f"[{self.name}] Model not provided. Markers will be computed from segmentation.")
            # Fallback: use segmentation as binary markers
            if "segmentation" not in data:
                raise KeyError("Fallback requires 'segmentation' key")
            data["markers"] = (data["segmentation"] > 0).astype(np.float32)
            return data

        try:
            if self.differentiable:
                data["markers"] = self._forward_differentiable(data["rgba"])
            else:
                data["markers"] = self._forward_inference(data["rgba"])

            logger.debug(
                f"[{self.name}] Generated markers with shape {tuple(data['markers'].shape)}"
            )

        except Exception as e:
            logger.error(f"[{self.name}] Inference failed: {e}")
            raise RuntimeError(f"MarkerNet inference failed: {e}") from e

        return data

    def _forward_inference(self, rgba: np.ndarray) -> np.ndarray:
        """Caminho de inferência: no_grad, resize não diferenciável e binarização.

        Args:
            rgba: array uint8 com formato ``(H, W, 4)``.

        Returns:
            array float32 com formato ``(H, W)`` binarizado com ``threshold``.
        """
        original_shape = rgba.shape[:2]

        # Preprocess: resize and normalize
        rgba_resized = self._preprocess(rgba)

        # Convert to tensor and add batch dimension
        rgba_tensor = torch.from_numpy(rgba_resized).float().to(self.device)
        rgba_tensor = rgba_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 4, 256, 256)

        # Inference
        with torch.no_grad():
            markers_pred = self.model.predict(rgba_tensor)  # (1, 1, 256, 256)

        # Post-process: resize back and threshold
        markers = self._postprocess(markers_pred.squeeze().cpu().numpy(), original_shape)

        return markers

    def _forward_differentiable(self, rgba: Any) -> torch.Tensor:
        """Caminho de treinamento: preserva o grafo e retorna probabilidades.

        Usa ``F.interpolate`` (diferenciável) em vez de ``cv2.resize`` e não
        aplica ``no_grad`` nem limiar, mantendo o fluxo de gradiente da loss
        até os parâmetros da MarkerNet.

        Args:
            rgba: array uint8 ``(H, W, 4)`` ou tensor ``(N, 4, H, W)``.

        Returns:
            Tensor ``(1, 1, H, W)`` com probabilidades em ``[0, 1]``.
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

        # Forward direto (sem predict() que usa no_grad) — grafo preservado.
        logits = self.model.model(rgba_resized)  # (1, 1, 256, 256)
        probs = torch.sigmoid(logits)

        if tuple(probs.shape[-2:]) != original_size:
            probs = F.interpolate(
                probs,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )

        return probs  # (1, 1, H, W)

    def _to_tensor_bchw(self, rgba: Any) -> torch.Tensor:
        """Converte a entrada RGBA para um tensor ``(N, 4, H, W)`` em ``[0, 1]``.

        Aceita arrays NumPy ``(H, W, 4)`` e tensores ``(4, H, W)`` ou
        ``(N, 4, H, W)``. Valores em ``[0, 255]`` são normalizados.

        Args:
            rgba: Entrada RGBA (numpy ou tensor).

        Returns:
            Tensor float32 ``(N, 4, H, W)`` no device configurado.
        """
        if isinstance(rgba, torch.Tensor):
            t = rgba.float()
            if t.ndim == 4 and t.shape[-1] in (3, 4):
                t = t.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            elif t.ndim == 4:
                pass  # já (N, C, H, W)
            elif t.ndim == 3:
                # Heurística: imagens com H ou W em (3, 4) são ambíguas; assume-se
                # canais-primeiro (C, H, W) quando o primeiro eixo é o canal.
                if t.shape[0] in (3, 4) and t.shape[-1] not in (3, 4):
                    t = t.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
                else:
                    t = t.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
            elif t.ndim == 2:
                t = t.unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1)  # (H, W) -> (1, 4, H, W)
        else:
            arr = rgba
            if arr.ndim == 2:
                arr = np.stack([arr] * 4, axis=-1)  # (H, W, 4)
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
            t = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0)

        if t.ndim == 3:
            t = t.unsqueeze(0)

        if t.shape[1] not in (3, 4):
            raise ValueError(f"Esperado 3 ou 4 canais em rgba, obtido {t.shape[1]}")

        if t.max() > 1.0:
            t = t / 255.0

        return t.to(self.device)

    def _preprocess(self, rgba: np.ndarray) -> np.ndarray:
        """Redimensiona a imagem RGBA para o tamanho alvo e normaliza para [0, 1].

        Args:
            rgba: array uint8 com formato ``(H, W, 4)``.

        Returns:
            array float32 com formato ``(256, 256, 4)`` normalizado para [0, 1].
        """
        # Resize to target size
        rgba_resized = cv2.resize(
            rgba,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalize to [0, 1] if the input is still in [0, 255]
        rgba_norm = rgba_resized.astype(np.float32)
        if rgba_norm.max() > 1.0:
            rgba_norm = rgba_norm / 255.0

        return rgba_norm

    def _postprocess(self, markers_pred: np.ndarray, original_shape: tuple) -> np.ndarray:
        """Redimensiona as previsões para o formato original e aplica um limiar.

        Args:
            markers_pred: array float32 com formato ``(256, 256)`` e valores em [0, 1].
            original_shape: Formato alvo de saída ``(H, W)``.

        Returns:
            array float32 com formato ``original_shape`` e probabilidades limiarizadas.
        """
        # Resize back to original shape
        markers_resized = cv2.resize(
            markers_pred,
            (original_shape[1], original_shape[0]),  # cv2.resize uses (W, H)
            interpolation=cv2.INTER_LINEAR
        )

        # Apply threshold
        markers_binary = (markers_resized > self.threshold).astype(np.float32)

        return markers_binary
