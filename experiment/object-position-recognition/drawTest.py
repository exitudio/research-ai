from PIL import Image
img = Image.new('RGB', (800,1280), (255, 255, 255))
img.save("data/image.png", "PNG")