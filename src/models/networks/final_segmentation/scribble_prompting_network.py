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

import hashlib
import pathlib
from typing import Any, Dict, Literal, Optional, Tuple

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
        scribble_mode: Como converter os marcadores (1 canal) da MarkerNet nos
            2 canais de scribbles esperados pela rede. ``"sharpened"`` (padrão)
            aplica ``sigmoid(T * (s - 0.5))`` / ``sigmoid(T * (0.5 - s))``,
            produzindo canais complementares quase binários — mais próximos da
            distribuição de treino do ScribblePrompt (traços 0/1 exclusivos) do
            que ``[s, 1-s]`` com s suave em toda a imagem.
            ``"dense_soft"`` usa ``[s, 1-s]``. ``"positive"`` usa só o canal
            positivo, ``relu(s − τ) / (1 − τ)``, e zera o negativo: abaixo do
            limiar τ o positivo é exatamente zero, porque qualquer valor positivo
            espalhado pela imagem (mesmo ~0,001) faz a rede segmentar demais.
        scribble_temperature: Temperatura ``T`` do sharpening (padrão ``10.0``).
            Valores maiores aproximam os scribbles de 0/1; valores menores
            preservam mais a suavidade (e o gradiente) dos marcadores.
        positive_threshold: Limiar τ do modo ``"positive"`` (padrão ``0.5``).
        unet: UNet interna congelada (registrada como submódulo).
        input_size: Tamanho espacial da entrada da rede (padrão do pacote: ``(128, 128)``).
        n_interactions: Número de passadas da rede por forward (padrão ``1``). A partir da
            segunda, a rede recebe os *logits* da passada anterior como ``mask_input``, como na
            interação em passos do ScribblePrompt.
    """

    CHECKPOINT_URLS: Dict[str, str] = {
        "v1": (
            "https://www.dropbox.com/scl/fi/pnw88n05irnv5z1snlklr/"
            "ScribblePrompt_unet_v1_nf192_res128.pt?rlkey=dr8xvkf0wj2r082h1zzpcmz5o&dl=1"
        ),
    }
    # SHA-256 do checkpoint esperado em cada versão: download_checkpoint recusa qualquer outro arquivo.
    CHECKPOINT_SHA256: Dict[str, str] = {
        "v1": "43f57ee8fa8ec529c31be281e06749f9e629b30157bbbcc9baf200cddec1acbe",
    }

    def __init__(
        self,
        version: Literal["v1"] = "v1",
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        resize_output: bool = True,
        scribble_mode: Literal["sharpened", "dense_soft", "positive"] = "sharpened",
        scribble_temperature: float = 10.0,
        positive_threshold: float = 0.5,
        input_size: Optional[Tuple[int, int]] = None,
        n_interactions: int = 1,
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
            scribble_mode: Conversão dos marcadores em scribbles — veja a
                descrição da classe (padrão ``"sharpened"``).
            scribble_temperature: Temperatura do sharpening (padrão ``10.0``).
            positive_threshold: Limiar τ do modo ``"positive"``, em ``[0, 1)``.
            input_size: ``(altura, largura)`` da entrada da rede; ambas divisíveis por 16 (a U-Net
                tem 4 *poolings*). ``None``: o tamanho do pacote, ``(128, 128)``. A rede é totalmente
                convolucional, então outros tamanhos usam os mesmos pesos.
            n_interactions: Passadas da rede por forward, ``>= 1``. Com mais de uma, cada passada
                recebe os *logits* da anterior como ``mask_input``; o gradiente atravessa todas.
            config: Configuração extra armazenada na rede.

        Raises:
            ImportError: Se o pacote ``scribbleprompt`` não estiver instalado.
            RuntimeError: Se o checkpoint não for encontrado.
            ValueError: Se ``scribble_mode``, ``positive_threshold``, ``input_size`` ou
                ``n_interactions`` forem inválidos.
        """
        if scribble_mode not in ("sharpened", "dense_soft", "positive"):
            raise ValueError(
                f"scribble_mode desconhecido: {scribble_mode!r}. Use 'sharpened', 'dense_soft' ou 'positive'."
            )
        if not 0.0 <= positive_threshold < 1.0:
            raise ValueError(f"positive_threshold deve estar em [0, 1), recebido {positive_threshold}.")
        if int(n_interactions) != n_interactions or n_interactions < 1:
            raise ValueError(f"n_interactions deve ser um inteiro >= 1, recebido {n_interactions}.")
        if input_size is not None:
            input_size = tuple(int(v) for v in input_size)
            if len(input_size) != 2 or any(v <= 0 or v % 16 for v in input_size):
                raise ValueError(f"input_size deve ter 2 dimensões positivas divisíveis por 16, recebido {input_size}.")

        if config is None:
            config = {
                "version": version,
                "checkpoint": checkpoint,
                "resize_output": resize_output,
                "scribble_mode": scribble_mode,
                "scribble_temperature": scribble_temperature,
                "positive_threshold": positive_threshold,
                "input_size": input_size,
                "n_interactions": int(n_interactions),
            }
        super().__init__(config=config)

        self.version = version
        self.resize_output = resize_output
        self.scribble_mode = scribble_mode
        self.scribble_temperature = scribble_temperature
        self.positive_threshold = positive_threshold
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

        self.input_size = input_size if input_size is not None else tuple(self._sp.input_size)
        self.n_interactions = int(n_interactions)
        logger.debug(
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
                - ``"negative"`` (opcional): canal negativo fixo (numpy ``(H, W)``
                  ou tensor ``(N, 1, H, W)``), em ``[0, 1]``. Substitui o canal
                  negativo produzido por ``scribble_mode``; o positivo não muda.

        Returns:
            Máscara de segmentação ``(N, 1, H, W)`` em ``[0, 1]`` (sigmoid).
            Se ``resize_output``, a máscara tem o tamanho espacial da imagem
            de entrada; caso contrário, ``(128, 128)``.

        Raises:
            KeyError: Se ``"image"`` ou ``"scribbles"`` estiverem ausentes.
            ValueError: Se ``"negative"`` não tiver o formato do canal positivo.
        """
        if not isinstance(data, dict) or "image" not in data or "scribbles" not in data:
            raise KeyError("ScribblePromptingNetwork.forward espera um dict com 'image' e 'scribbles'.")

        device = next(self.unet.parameters()).device

        img = self._prepare_image(data["image"], device)  # (N, 1, H, W) [0, 1]
        scribbles = self._prepare_scribbles(data["scribbles"], device)  # (N, 2, H, W) [0, 1]
        if data.get("negative") is not None:
            negative = self._prepare_negative(data["negative"], device)
            if negative.shape != scribbles[:, :1].shape:
                raise ValueError(
                    f"O canal negativo {tuple(negative.shape)} não tem o formato do positivo "
                    f"{tuple(scribbles[:, :1].shape)}."
                )
            scribbles = torch.cat([scribbles[:, :1], negative], dim=1)
        original_size = tuple(img.shape[-2:])

        inputs = self._rescale_inputs({"img": img, "scribbles": scribbles}, self.input_size)

        # Chamada direta na UNet congelada (sem no_grad): gradiente flui até os
        # scribbles, permitindo otimizar a MarkerNet através da loss. A partir da
        # segunda passada, os logits da anterior entram como mask_input.
        logits = None
        for _ in range(self.n_interactions):
            if logits is not None:
                inputs["mask_input"] = logits
            x = self._prepare_inputs(inputs).float().to(device)  # (N, 5, h, w)
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
        entrada tiver 1 canal (marcadores da MarkerNet), converte conforme
        ``scribble_mode``:

        - ``"sharpened"`` (padrão): aplica um sharpening sigmoide com
          temperatura configurável, produzindo canais complementares quase
          binários — ``pos = sigmoid(T*(s-0.5))`` e ``neg = sigmoid(T*(0.5-s))``.
          Isso aproxima a distribuição de treino do ScribblePrompt (traços 0/1
          mutuamente exclusivos) e evita a entrada degenerada ``[s, 1-s]`` com
          s suave em todos os pixels.
        - ``"dense_soft"``: ``[s, 1 - s]``.
        - ``"positive"``: ``[relu(s − τ) / (1 − τ), 0]``.

        Tensores preservam o grafo computacional (sem detach) para permitir a
        backpropagation até a MarkerNet.

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
            s = self._expand_scribble_channels(s)
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
            s = self._expand_scribble_channels(s)

        return s.to(device)

    def scribble_channels(self, markers: Any) -> torch.Tensor:
        """Canais ``(N, 2, H, W)`` [positivo, negativo] que ``scribble_mode`` produz para os marcadores.

        Útil para visualizar o que chega à rede. Não inclui o canal negativo fixo (``"negative"``).

        Args:
            markers: Marcadores (numpy ou tensor), como em ``data["scribbles"]``.

        Returns:
            Tensor ``(N, 2, H, W)`` no dispositivo da rede.
        """
        return self._prepare_scribbles(markers, next(self.unet.parameters()).device)

    @staticmethod
    def _prepare_negative(negative: Any, device: torch.device) -> torch.Tensor:
        """Converte o canal negativo fixo em tensor ``(N, 1, H, W)`` float, sem gradiente.

        Args:
            negative: Numpy ``(H, W)`` ou tensor ``(H, W)``/``(1, H, W)``/``(N, 1, H, W)``.
            device: Dispositivo de destino.

        Returns:
            Tensor ``(N, 1, H, W)``.
        """
        if isinstance(negative, torch.Tensor):
            n = negative.detach().float()
        else:
            n = torch.from_numpy(np.ascontiguousarray(np.asarray(negative, dtype=np.float32)))
        if n.ndim == 2:
            n = n[None, None]
        elif n.ndim == 3:
            n = n[None]
        elif n.ndim != 4:
            raise ValueError(f"Dimensões do canal negativo não suportadas: {tuple(n.shape)}")
        return n.to(device)

    def _expand_scribble_channels(self, s: torch.Tensor) -> torch.Tensor:
        """Expande 1 canal de scribbles para 2 canais conforme ``scribble_mode``.

        Args:
            s: Tensor ``(N, 1, H, W)`` ou ``(N, 2, H, W)``.

        Returns:
            Tensor ``(N, 2, H, W)`` em ``[0, 1]``.
        """
        if s.shape[1] == 2:
            return s
        if s.shape[1] != 1:
            raise ValueError(f"Esperado 1 ou 2 canais de scribbles, obtido {s.shape[1]}")

        if self.scribble_mode == "dense_soft":
            return torch.cat([s, 1.0 - s], dim=1)

        if self.scribble_mode == "sharpened":
            t = self.scribble_temperature
            pos = torch.sigmoid(t * (s - 0.5))
            neg = torch.sigmoid(t * (0.5 - s))
            return torch.cat([pos, neg], dim=1)

        if self.scribble_mode == "positive":
            tau = self.positive_threshold
            pos = torch.relu(s - tau) / (1.0 - tau)
            return torch.cat([pos, torch.zeros_like(pos)], dim=1)

        raise ValueError(
            f"scribble_mode desconhecido: {self.scribble_mode!r}. "
            "Use 'sharpened', 'dense_soft' ou 'positive'."
        )

    @staticmethod
    def _sha256(path: pathlib.Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def download_checkpoint(cls, dest_dir: str, version: Literal["v1"] = "v1") -> str:
        """Baixa o checkpoint do ScribblePrompt para ``dest_dir`` e confere o SHA-256.

        Útil no Colab, onde os pesos precisam ser baixados antes de instanciar a rede. Se o
        arquivo já existir, não baixa de novo, mas confere o SHA-256 antes de devolvê-lo. O download
        vai para um arquivo temporário (``<nome>.part``) e só ganha o nome final depois de
        conferido: um download interrompido ou um arquivo diferente na origem nunca vira o
        checkpoint.

        Args:
            dest_dir: Diretório de destino do checkpoint.
            version: Versão do modelo (apenas ``"v1"``).

        Returns:
            Caminho completo do checkpoint conferido.

        Raises:
            RuntimeError: Se o arquivo existente ou o baixado não tiver o SHA-256 esperado.
        """
        import urllib.request

        expected = cls.CHECKPOINT_SHA256[version]
        name = f"ScribblePrompt_unet_{version}_nf192_res128.pt"
        dest = pathlib.Path(dest_dir) / name
        if dest.exists():
            found = cls._sha256(dest)
            if found != expected:
                raise RuntimeError(
                    f"O checkpoint em {dest} não confere com o SHA-256 esperado (esperado "
                    f"{expected[:12]}…, encontrado {found[:12]}…). "
                    "Apague o arquivo para baixar de novo."
                )
            logger.debug(f"[ScribblePromptingNetwork] Checkpoint já existe e confere: {dest}")
            return str(dest)

        url = cls.CHECKPOINT_URLS[version]
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".part")
        logger.info(f"[ScribblePromptingNetwork] Baixando checkpoint de {url}")
        try:
            urllib.request.urlretrieve(url, partial)
            found = cls._sha256(partial)
            if found != expected:
                raise RuntimeError(
                    f"O checkpoint baixado de {url} não confere com o SHA-256 esperado (esperado "
                    f"{expected[:12]}…, recebido {found[:12]}…): o arquivo na origem mudou ou o "
                    "download foi corrompido."
                )
            partial.replace(dest)
        finally:
            if partial.exists():
                partial.unlink()
        logger.info(f"[ScribblePromptingNetwork] Checkpoint conferido e salvo em {dest}")
        return str(dest)
