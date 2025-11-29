import numpy as np
from PIL import Image
from skimage.morphology import reconstruction

def load_backbone_mask(backbone_path, threshold=0):
    """
    Load backbone_<id>.png and return a binary mask in {0,1}.
    """
    bb = np.array(Image.open(backbone_path))

    # In the repo: they save (0 or 1)*255 as uint8
    # So threshold at >0 is enough
    mask = bb > threshold
    return mask.astype(np.uint8)

def load_skeleton_mask(heatmap_path, heatmap_threshold=0.0):
    """
    Load heatmap_<id>.npy and convert to a binary skeleton mask.

    seg_map / heatmap in the repo is already 'skeletal' and only
    contains structures with p̄(e) >= 0.5, but values are floats.
    """
    hm = np.load(heatmap_path)     # shape (H, W) or (1, H, W)
    if hm.ndim == 3:
        hm = np.squeeze(hm, axis=0)

    skel = hm > heatmap_threshold  # e.g. 0 or tiny eps
    return skel.astype(np.uint8)

def build_final_segmentation(backbone_path, heatmap_path,
                             heatmap_threshold=0.0):
    """
    Implements the paper's 'overlay' step:

      - skeleton map from Mϕ (structures with p̄(e) ≥ 0.5)
      - backbone mask from Fθ

    Final mask = morphological reconstruction of the backbone
    using the skeleton as marker (keeps full thickness for
    structures that have a skeleton).
    """
    backbone = load_backbone_mask(backbone_path)        # mask
    skeleton = load_skeleton_mask(heatmap_path, heatmap_threshold)

    # The skeleton should lie within the backbone
    marker = (skeleton & backbone).astype(np.uint8)
    mask   = backbone.astype(np.uint8)

    # Morphological reconstruction by dilation:
    # - Only those connected components of 'mask' that contain at least
    #   one 'marker' pixel are kept, with their original thickness.
    final = reconstruction(marker, mask, method='dilation')

    # reconstruction() returns floats in [0,1]; binarize
    final_bin = final > 0

    return final_bin.astype(np.uint8)

# Example usage
if __name__ == "__main__":
    backbone_path = "results/backbone_case123.png"
    heatmap_path  = "results/heatmap_case123.npy"

    final_seg = build_final_segmentation(backbone_path, heatmap_path)

    # Save as PNG
    final_img = Image.fromarray((final_seg * 255).astype(np.uint8))
    final_img.save("results/final_seg_case123.png")
