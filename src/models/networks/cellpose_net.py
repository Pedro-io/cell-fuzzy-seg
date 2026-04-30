from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from base_networks import BaseNetwork
from cellpose import core, io, models, plot, transforms
from natsort import natsorted
from PIL import Image
from tqdm import trange


class CellposeSAMNetwork(BaseNetwork):

  def __init__(self, config):
    super().__init__(config)
    #Check if colab notebook instance has GPU access
    if not core.use_gpu():
      raise ImportError("No GPU access, change your runtime")

    model = models.CellposeModel(gpu=True)

    # %% [markdown]
    # Input directory with your images:

    # %%
  # *** change to your google drive folder path ***
  dir = "/content/cell-fuzzy-seg/data/MoNuSegTrainingData/Tissue_Images"
  output_dir = "/content/cell-fuzzy-seg/data/outputs"
  dir = Path(dir)
  output_dir = Path(output_dir)
  if not dir.exists():
    raise FileNotFoundError("directory does not exist")
  if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)


  # %%
  # *** change to your image extension ***
  image_ext = ".tif"

  # list all files
  files = natsorted([f for f in dir.glob("*"+image_ext) if "_masks" not in f.name and "_flows" not in f.name])

  if(len(files)==0):
    raise FileNotFoundError("no image files found, did you specify the correct folder and extension?")
  else:
    print(f"{len(files)} images in folder:")

  for f in files:
    print(f.name)

  # %% [markdown]
  # ## Run Cellpose-SAM on one image in folder
  #
  # Here are some of the parameters you can change:
  #
  # * ***flow_threshold*** is  the  maximum  allowed  error  of  the  flows  for  each  mask.   The  default  is 0.4.
  #     *  **Increase** this threshold if cellpose is not returning as many masks as you’d expect (or turn off completely with 0.0)
  #     *   **Decrease** this threshold if cellpose is returning too many ill-shaped masks.
  #
  # * ***cellprob_threshold*** determines proability that a detected object is a cell.   The  default  is 0.0.
  #     *   **Decrease** this threshold if cellpose is not returning as many masks as you’d expect or if masks are too small
  #     *   **Increase** this threshold if cellpose is returning too many masks esp from dull/dim areas.
  #
  # * ***tile_norm_blocksize*** determines the size of blocks used for normalizing the image. The default is 0, which means the entire image is normalized together.
  #   You may want to change this to 100-200 pixels if you have very inhomogeneous brightness across your image.
  #
  #

  # %%
  img = io.imread(files[0])

  print(f'your image has shape: {img.shape}. Assuming channel dimension is last with {img.shape[-1]} channels')

  # %% [markdown]
  # ### Channel Selection:
  #
  # - Use the dropdowns below to select the _zero-indexed_ channels of your image to segment. The order does not matter. Remember to rerun the cell after you edit the dropdowns.
  #
  # - If you have a histological image taken in brightfield, you don't need to adjust the channels.
  #
  # - If you have a fluroescent image with multiple stains, you should choose one channel with a cytoplasm/membrane stain, one channel with a nuclear stain, and set the third channel to `None`. Choosing multiple channels may produce segmentaiton of all the structures in the image. If you have retrained the model on your data with a thrid stain (described below), you can run segmentation with all channels.

  # %%
  first_channel = '0' # @param ['None', 0, 1, 2, 3, 4, 5]
  second_channel = '0' # @param ['None', 0, 1, 2, 3, 4, 5]
  third_channel = '0' # @param ['None', 0, 1, 2, 3, 4, 5]

  # %%
  selected_channels = []
  for i, c in enumerate([first_channel, second_channel, third_channel]):
    if c == 'None':
      continue
    if int(c) > img.shape[-1]:
      assert False, 'invalid channel index, must have index greater or equal to the number of channels'
    if c != 'None':
      selected_channels.append(int(c))



  img_selected_channels = np.zeros_like(img)
  img_selected_channels[:, :, :len(selected_channels)] = img[:, :, selected_channels]


  flow_threshold = 0.4
  cellprob_threshold = -4.0
  tile_norm_blocksize = 0

  transforms.normalize_img(img, normalize=True, norm3D=True, invert=True, lowhigh=None, percentile=(1.0, 99.0), sharpen_radius=0, smooth_radius=0, tile_norm_blocksize=0, tile_norm_smooth3D=1, axis=-1)

  masks, flows, styles = model.eval(img_selected_channels, batch_size=10, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                    normalize={"tile_norm_blocksize": tile_norm_blocksize}, diameter=100)

  fig = plt.figure(figsize=(12,5))
  plot.show_segmentation(fig, img_selected_channels, masks, flows[0])
  plt.tight_layout()
  plt.show()


  # %% [markdown]
  # ## Run Cellpose-SAM on folder of images

  # %% [markdown]
  # if you have small images, you may want to load all of them first and then run, so that they can be batched together on the GPU

  # %%
  print("loading images")
  imgs = [io.imread(files[i]) for i in trange(len(files))]

  print("running cellpose-SAM")
  masks, flows, styles = model.eval(imgs, batch_size=10, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                    normalize={"tile_norm_blocksize": tile_norm_blocksize})
  # tentar diminuir o threshold para ver se consegue pegar mais células, mas cuidado para não pegar muito ruído

  masks_ext = ".png" if image_ext == ".png" else ".tif"
  print("saving masks")
  for i in trange(len(files)):
      f = files[i]
      io.imsave(output_dir / (f.stem + "_masks" + masks_ext), masks[i])

  # %% [
  def load(self):
    for idx in trange(len(files)):
        img_original = imgs[idx]
        mask = masks[idx]

        print(f"Processando imagem {files[idx].name} para salvar com 4 canais (RGBA)...")
        print(f"Forma da imagem original: {img_original.shape}")
        print(f"Forma da máscara: {mask.shape}")

        # 1. Normalizar imagem original e manter RGB PURO
        img_normalized = (img_original / img_original.max() * 255).astype(np.uint8)
        if len(img_normalized.shape) == 2:
            img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_normalized[:, :, :3]

        # 2. Criar o canal Alpha como uma MÁSCARA BINÁRIA (0 ou 255)
        # Isso serve como o "guia espacial" para a rede
        alpha_channel = (mask > 0).astype(np.uint8) * 255

        # 3. Concatenar sem fazer "blending" de cores
        # Resultado: Canal 0=R, 1=G, 2=B (puros) e 3=Máscara
        input_rgba = np.dstack([img_rgb, alpha_channel])

        print(f"Forma do overlay RGBA: {input_rgba.shape} (Canais: R,G,B,Alpha)")

        # Salvar
        img_pil = Image.fromarray(input_rgba)
        img_pil.save(str(output_dir / f"{files[idx].stem}_input_4ch.png"))
