"""Rede final de segmentação congelada baseada no ScribblePrompt.

Este módulo implementa :class:`ScribblePromptingNetwork`, a implementação da
interface :class:`~src.models.networks.final_segmentation.base_final_segmentation.BaseFinalSegmentation`
baseada no modelo **ScribblePrompt** (https://github.com/halleewong/ScribblePrompt).

Diferente de uma rede treinável, esta rede é **congelada**: os pesos do
ScribblePrompt-UNet não são atualizados durante o treinamento. O objetivo é
apenas usar a saída dela (a segmentação final) para calcular as perdas que
otimizam a MarkerNet. Para que isso seja possível, o grafo computacional deve
permanecer diferenciável em relação aos *scribbles* de entrada (que são os
marcadores produzidos pela MarkerNet).

**Por que não usamos ``ScribblePromptUNet.predict``?**
O método ``predict`` do pacote é decorado com ``@torch.no_grad()``, o que
interromperia o fluxo de gradiente até a MarkerNet. Por isso, este wrapper
chama diretamente a UNet congelada (``self.unet``), preparando as entradas com
as funções ``prepare_inputs``/``rescale_inputs`` do próprio pacote.

**Checkpoint:**
O pacote ``scribbleprompt`` (instalado via ``requirements.txt`` a partir do
git) espera o checkpoint ``ScribblePrompt_unet_v1_nf192_res128.pt`` no
diretório padrão ou em um caminho informado via ``checkpoint``. Use o método
:meth:`download_checkpoint` para baixá-lo (links do Dropbox do repositório
original) ou informe o caminho do arquivo baixado.
"""

import pathlib
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.logger import logger

from .base_final_segmentation import BaseFinalSegmentation


class ScribblePromptingNetwork(BaseFinalSegmentation):
    """Rede final congelada baseada no ScribblePrompt-UNet.

    Recebe um dicionário com as chaves ``"image"`` e ``"scribbles"`` e devolve
    a máscara de segmentação ``(N, 1, H, W)`` em ``[0, 1]``. A imagem é
    convertida para tons de cinza (como no tutorial do ScribblePrompt) e os
    scribbles (marcadores da MarkerNet, 1 canal) são expandidos para 2 canais
    (positivo/negativo) antes de alimentar a rede.

    Atributos:
        version: Versão do modelo ScribblePrompt (apenas ``"v1"``).
        device: Dispositivo PyTorch usado na inferência.
        resize_output: Se ``True`` (padrão), redimensiona a máscara de volta
            para o tamanho espacial da imagem de entrada, alinhando com o
            ground truth para o cálculo da loss.
        unet: UNet interna congelada (registrada como submódulo).
        input_size: Tamanho espacial esperado pelo modelo (``(128, 128)``).
    """

    CHECKPOINT_URLS: Dict[str, str] = {
        "v1": (
            "https://www.dropbox.com/scl/fi/pnw88n05irnv5z1snlklr/"
            "ScribblePrompt_unet_v1_nf192_res128.pt?rlkey=dr8xvkf0wj2r082h1zzpcmz5o&dl=1"
        ),
    }

    def __init__(
        self,
        version: Literal["v1"] = "v1",
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        resize_output: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Inicializa a rede congelada do ScribblePrompt.

        Args:
            version: Versão do modelo (apenas ``"v1"`` está disponível).
            checkpoint: Caminho para o arquivo de checkpoint. Se ``None``,
                usa o caminho padrão esperado pelo pacote.
            device: Dispositivo PyTorch. Se ``None``, usa GPU se disponível.
            resize_output: Redimensiona a saída para o tamanho da imagem de
                entrada (padrão ``True``).
            config: Configuração extra armazenada na rede.

        Raises:
            ImportError: Se o pacote ``scribbleprompt`` não estiver instalado.
            RuntimeError: Se o checkpoint não for encontrado.
        """
        if config is None:
            config = {
                "version": version,
                "checkpoint": checkpoint,
                "resize_output": resize_output,
            }
        super().__init__(config=config)

        self.version = version
        self.resize_output = resize_output
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            from scribbleprompt.models.unet import (
                ScribblePromptUNet,
                prepare_inputs,
                rescale_inputs,
            )
        except ImportError as e:
            raise ImportError(
                "O pacote 'scribbleprompt' não está instalado. Adicione "
                "'scribbleprompt' ao requirements.txt (instalado a partir do git)."
            ) from e

        self._prepare_inputs = prepare_inputs
        self._rescale_inputs = rescale_inputs

        if checkpoint is not None:
            # O pacote lê ``weights`` de um dict de CLASSE; sobrescrever o
            # caminho aqui é um efeito colateral global (afeta futuras
            # instâncias). Aceitável porque o projeto usa uma única instância
            # desta rede congelada.
            ScribblePromptUNet.weights[version] = str(pathlib.Path(checkpoint))

        try:
            self._sp = ScribblePromptUNet(version=version, device=self.device)
        except (AssertionError, FileNotFoundError) as e:
            raise RuntimeError(
                "Checkpoint do ScribblePrompt não encontrado. Baixe-o com "
                f"ScribblePromptingNetwork.download_checkpoint('caminho') ou "
                f"informe o parâmetro 'checkpoint' no construtor. ({e})"
            ) from e

        # Registra a UNet interna como submódulo (permite .to(), .train(), etc.)
        self.unet = self._sp.model
        # Congela os pesos: esta rede nunca é treinada.
        self.unet.requires_grad_(False)
        self.unet.eval()

        self.input_size = tuple(self._sp.input_size)
        logger.info(
            f"[ScribblePromptingNetwork] Inicializada (version={version}, "
            f"device={self.device}, input_size={self.input_size})"
        )

    def train(self, mode: bool = True):
        """Mantém a rede congelada sempre em modo ``eval``.

        O :class:`~src.pipeline.steps.inference.frozen_segmentation_step.FrozenSegmentationStep`
        chama ``train()`` para preservar o grafo diferenciável, mas esta rede é
        congelada e não possui camadas sensíveis a ``train/eval`` (Conv2d +
        PReLU). Forçamos ``eval()`` para manter o contrato de rede congelada.
        """
        return super().train(False)

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        """Executa o forward congelado do ScribblePrompt sobre um dicionário.

        Args:
            data: Dicionário com as chaves:
                - ``"image"``: imagem de entrada (numpy ``(H, W)``/``(H, W, 3)``
                  ou tensor ``(N, 1, H, W)``/``(N, 3, H, W)``).
                - ``"scribbles"``: marcadores da MarkerNet (numpy ``(H, W)`` ou
                  tensor ``(N, 1, H, W)``/``(N, 2, H, W)``).

        Returns:
            Máscara de segmentação ``(N, 1, H, W)`` em ``[0, 1]`` (sigmoid).
            Se ``resize_output``, a máscara tem o tamanho espacial da imagem
            de entrada; caso contrário, ``(128, 128)``.

        Raises:
            KeyError: Se ``"image"`` ou ``"scribbles"`` estiverem ausentes.
        """
        if not isinstance(data, dict) or "image" not in data or "scribbles" not in data:
            raise KeyError("ScribblePromptingNetwork.forward espera um dict com 'image' e 'scribbles'.")

        device = next(self.unet.parameters()).device

        img = self._prepare_image(data["image"], device)  # (N, 1, H, W) [0, 1]
        scribbles = self._prepare_scribbles(data["scribbles"], device)  # (N, 2, H, W) [0, 1]
        original_size = tuple(img.shape[-2:])

        inputs = self._rescale_inputs({"img": img, "scribbles": scribbles}, self.input_size)
        x = self._prepare_inputs(inputs).float().to(device)  # (N, 5, 128, 128)

        # Chamada direta na UNet congelada (sem no_grad): gradiente flui até os
        # scribbles, permitindo otimizar a MarkerNet através da loss.
        logits = self.unet(x)

        if self.resize_output and tuple(logits.shape[-2:]) != original_size:
            logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)

        return torch.sigmoid(logits)

    def predict(self, data: Dict[str, Any]) -> torch.Tensor:
        """Versão de inferência (sem gradiente) do :meth:`forward`.

        Args:
            data: Mesmo dicionário aceito por :meth:`forward`.

        Returns:
            Máscara de segmentação ``(N, 1, H, W)`` em ``[0, 1]``.
        """
        with torch.no_grad():
            return self.forward(data)

    def _prepare_image(self, image: Any, device: torch.device) -> torch.Tensor:
        """Converte a imagem de entrada para ``(N, 1, H, W)`` em tons de cinza.

        Segue o tutorial do ScribblePrompt: converte para grayscale e normaliza
        para ``[0, 1]``. Tensores já existentes são desanexados do grafo (a
        imagem é dado de entrada, não parte da otimização).

        Args:
            image: Imagem numpy ou tensor.
            device: Dispositivo de destino.

        Returns:
            Tensor ``(N, 1, H, W)`` em ``[0, 1]``.
        """
        if isinstance(image, torch.Tensor):
            img = image.detach().float()
            if img.ndim == 2:
                img = img[None, None]
            elif img.ndim == 3:
                img = img[None]  # assume (C, H, W)
            elif img.ndim == 4:
                pass
            else:
                raise ValueError(f"Dimensões de imagem não suportadas: {img.shape}")
            if img.shape[1] == 3:
                img = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
            elif img.shape[1] != 1:
                raise ValueError(f"Esperado 1 ou 3 canais, obtido {img.shape[1]}")
        else:
            arr = np.asarray(image, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[None, None]  # (1, 1, H, W)
            elif arr.ndim == 3 and arr.shape[-1] == 3:
                arr = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
                arr = arr[None, None]  # (1, 1, H, W)
            elif arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0][None, None]
            elif arr.ndim == 4:
                if arr.shape[-1] == 3:
                    arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
                elif arr.shape[-1] == 1:
                    arr = arr[..., 0]
                else:
                    raise ValueError(f"Número de canais não suportado: {arr.shape}")
                arr = arr[:, None]  # (N, 1, H, W)
            else:
                raise ValueError(f"Dimensões de imagem não suportadas: {arr.shape}")
            img = torch.from_numpy(np.ascontiguousarray(arr))

        if img.max() > 1.0:
            img = img / 255.0

        return img.to(device)

    def _prepare_scribbles(self, scribbles: Any, device: torch.device) -> torch.Tensor:
        """Converte os marcadores em scribbles ``(N, 2, H, W)``.

        O ScribblePrompt espera 2 canais de scribbles (positivo/negativo). Se a
        entrada tiver 1 canal (marcadores da MarkerNet), expande para
        ``[s, 1 - s]``. Tensores preservam o grafo computacional (sem detach)
        para permitir a backpropagation até a MarkerNet.

        Args:
            scribbles: Marcadores numpy ou tensor.
            device: Dispositivo de destino.

        Returns:
            Tensor ``(N, 2, H, W)`` em ``[0, 1]``.
        """
        if isinstance(scribbles, torch.Tensor):
            s = scribbles.float()
            if s.ndim == 2:
                s = s[None, None]
            elif s.ndim == 3:
                s = s[None]  # assume (C, H, W)
            elif s.ndim == 4:
                pass
            else:
                raise ValueError(f"Dimensões de scribbles não suportadas: {s.shape}")
            if s.shape[1] == 1:
                s = torch.cat([s, 1.0 - s], dim=1)
            elif s.shape[1] != 2:
                raise ValueError(f"Esperado 1 ou 2 canais de scribbles, obtido {s.shape[1]}")
        else:
            arr = np.asarray(scribbles, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[None, None]  # (1, 1, H, W)
            elif arr.ndim == 3 and arr.shape[-1] in (1, 2):
                arr = arr.transpose(2, 0, 1)[None]  # (1, C, H, W)
            elif arr.ndim == 3:
                arr = arr[None]  # assume (C, H, W)
            elif arr.ndim == 4 and arr.shape[-1] in (1, 2):
                arr = arr.transpose(0, 3, 1, 2)  # (N, C, H, W)
            else:
                raise ValueError(f"Dimensões de scribbles não suportadas: {arr.shape}")
            s = torch.from_numpy(np.ascontiguousarray(arr))
            if s.shape[1] == 1:
                s = torch.cat([s, 1.0 - s], dim=1)
            elif s.shape[1] != 2:
                raise ValueError(f"Esperado 1 ou 2 canais de scribbles, obtido {s.shape[1]}")

        return s.to(device)

    @classmethod
    def download_checkpoint(cls, dest_dir: str, version: Literal["v1"] = "v1") -> str:
        """Baixa o checkpoint do ScribblePrompt para ``dest_dir``.

        Útil no Colab, onde os pesos precisam ser baixados antes de instanciar
        a rede. Se o arquivo já existir, não baixa novamente.

        Args:
            dest_dir: Diretório de destino do checkpoint.
            version: Versão do modelo (apenas ``"v1"``).

        Returns:
            Caminho completo do checkpoint baixado.
        """
        import urllib.request

        name = f"ScribblePrompt_unet_{version}_nf192_res128.pt"
        dest = pathlib.Path(dest_dir) / name
        if dest.exists():
            logger.info(f"[ScribblePromptingNetwork] Checkpoint já existe: {dest}")
            return str(dest)

        url = cls.CHECKPOINT_URLS[version]
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"[ScribblePromptingNetwork] Baixando checkpoint de {url}")
        urllib.request.urlretrieve(url, dest)
        logger.info(f"[ScribblePromptingNetwork] Checkpoint salvo em {dest}")
        return str(dest)
