from PIL import Image
import sys
import io

def flip_image(image_path):
    # Open the image
    with Image.open(image_path) as img:
        # Flip the image horizontally
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert the flipped image to binary data
        with io.BytesIO() as output:
            flipped_img.save(output, format='PNG')
            flipped_img_data = output.getvalue()

    return flipped_img_data

if __name__ == "__main__":
    # Read the image file path from standard input
    image_path = sys.stdin.readline().strip()

    # Flip the image
    flipped_image_data = flip_image(image_path)

    # Write the flipped image data to standard output
    sys.stdout.buffer.write(flipped_image_data)
