import torch.nn.functional as F
import torch

def fast_dynamic_preprocess(stitched_tensor, image_size=448, use_thumbnail=False):
    """
    Fast tensor slicing into blocks without any Python for-loop.
    """
    # stitched_tensor: (C, H, W)
    C, H, W = stitched_tensor.shape
    num_blocks_w, num_blocks_h = 6, 2  # target aspect ratio width:height

    block_width = image_size
    block_height = image_size

    assert H >= block_height * num_blocks_h, f"Height {H} too small!"
    assert W >= block_width * num_blocks_w, f"Width {W} too small!"

    # First, reshape into blocks
    stitched_tensor = stitched_tensor[:, :block_height * num_blocks_h, :block_width * num_blocks_w]
    blocks = stitched_tensor.view(
        C,
        num_blocks_h, block_height,
        num_blocks_w, block_width
    )  # Shape: (C, num_blocks_h, block_height, num_blocks_w, block_width)

    # Now permute to group the blocks nicely: (num_blocks_h, num_blocks_w, C, block_height, block_width)
    blocks = blocks.permute(1, 3, 0, 2, 4)

    # Merge first two dimensions (flatten block grid)
    blocks = blocks.reshape(-1, C, block_height, block_width)  # (12, 3, 448, 448)

    # Optional thumbnail
    if use_thumbnail:
        thumbnail_img = F.interpolate(
            stitched_tensor.unsqueeze(0),  # (1, 3, H, W)
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)  # (3, 448, 448)
        blocks = torch.cat([blocks, thumbnail_img.unsqueeze(0)], dim=0)  # (13, 3, 448, 448)

    return blocks  # shape (12, 3, 448, 448) or (13, 3, 448, 448)

stitched = torch.randn(3, 4, 12)  # 3x(448*2)x(448*6)

blocks = fast_dynamic_preprocess(stitched, image_size=2, use_thumbnail=False)
print(blocks)
print(stitched)
print(blocks.shape)