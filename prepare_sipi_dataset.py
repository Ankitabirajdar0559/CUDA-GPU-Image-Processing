from pathlib import Path
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "sipi_dataset" / "misc"
OUTPUT_DIR = PROJECT_DIR / "input"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tiff_files = sorted(SOURCE_DIR.glob("*.tiff"))

if not tiff_files:
    raise FileNotFoundError(f"No TIFF files found in: {SOURCE_DIR}")

converted = 0

for tiff_file in tiff_files:
    output_file = OUTPUT_DIR / f"{tiff_file.stem}.pgm"

    with Image.open(tiff_file) as image:
        grayscale = image.convert("L")
        grayscale.save(output_file, format="PPM")

    print(
        f"Converted: {tiff_file.name} -> "
        f"{output_file.name} "
        f"({grayscale.width}x{grayscale.height})"
    )

    converted += 1

print()
print(f"Total TIFF files found : {len(tiff_files)}")
print(f"Total PGM files created: {converted}")
print(f"Output directory       : {OUTPUT_DIR}")
