
from PIL import Image
from pathlib import Path

# Original and processed folders
mask_folder = Path("data/masks")
output_folder = Path("data/processed_masks")

output_folder.mkdir(parents=True, exist_ok=True)

for i in range(100):
    mask_path = mask_folder / f"{i}.png"

    # Open original mask
    mask = Image.open(mask_path).convert("RGBA")

    # Extract transparency channel
    alpha = mask.getchannel("A")

    # Convert to binary mask
    binary_mask = alpha.point(
        lambda pixel: 255 if pixel >= 128 else 0
    )

    # Save processed mask
    binary_mask.save(output_folder / f"{i}.png")

print("All masks processed successfully!")
print("Total processed masks:", len(list(output_folder.glob("*.png"))))
print("Original masks preserved.")
