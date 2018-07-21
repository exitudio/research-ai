from PIL import Image, ImageDraw
from random import randint
import pathlib
from math import ceil


class Gen_image:
    def __init__(self, size=(800, 800)):
        self.size = size

    def circles(self, num_image):
        path = "data/circles"
        for id in range(num_image):
            x = randint(0, self.size[0])
            y = randint(0, self.size[1])
            r = randint( ceil(self.size[0]/10), self.size[0]/2)

            [img, draw] = self.gen_image()
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(self.randColor(),
                                                     self.randColor(), self.randColor(), 255))

            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            img.save(f"{path}/image{id}.png", "PNG")

    def squares(self, num_image):
        path = "data/squares"
        for id in range(num_image):
            x = randint(0, self.size[0])
            y = randint(0, self.size[1])
            r = randint( ceil(self.size[0]/10), self.size[0]/2)

            [img, draw] = self.gen_image()
            draw.rectangle(((x-r, y-r), (x+r, y+r)), fill=(self.randColor(),
                                                           self.randColor(), self.randColor(), 255))

            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            img.save(f"{path}/image{id}.png", "PNG")

    def randColor(self):
        return randint(0, 255)

    def gen_image(self):
        img = Image.new('RGB', self.size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        return [img, draw]


gen_image = Gen_image((28, 28))
gen_image.circles(1000)
gen_image.squares(1000)
