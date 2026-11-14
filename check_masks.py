
from PIL import Image
from pathlib import Path

mask_folder = Path("data/masks")

for i in range(100):
    path = mask_folder / f"{i}.png"

    if not path.exists():
        print("Missing:", path)
        continue

    mask = Image.open(path).convert("RGBA")
    alpha = mask.getchannel("A")

    print(
        f"{i}.png | Size: {mask.size} | "
        f"Alpha range: {min(alpha.getdata())}-"
        f"{max(alpha.getdata())}"
    )

print("\nAll mask checks completed!")
