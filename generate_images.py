import os

WIDTH = 1024
HEIGHT = 1024
IMAGE_COUNT = 20

os.makedirs("input", exist_ok=True)

for image_id in range(IMAGE_COUNT):
    filename = f"input/image_{image_id:02d}.pgm"

    with open(filename, "wb") as file:
        file.write(b"P5\n")
        file.write(f"{WIDTH} {HEIGHT}\n".encode())
        file.write(b"255\n")

        data = bytearray(WIDTH * HEIGHT)

        for y in range(HEIGHT):
            for x in range(WIDTH):
                value = (
                    x
                    + y
                    + image_id * 10
                    + (x // 32) * 20
                ) % 256

                data[y * WIDTH + x] = value

        file.write(data)

print(
    f"Generated {IMAGE_COUNT} images "
    f"of {WIDTH}x{HEIGHT} pixels."
)