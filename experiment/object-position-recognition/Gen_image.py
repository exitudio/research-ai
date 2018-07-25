from PIL import Image, ImageDraw
from random import randint
import pathlib
from math import ceil, floor
from Csv_manager import save_csv


class Gen_image:
    def __init__(self, size=(800, 800)):
        self.size = size

    def circles(self, num_image, path="data/circles"):
        def draw_circle(draw, x, y, r):
            draw.ellipse((x-r, y-r, x+r, y+r), fill=self._get_random_color())
        return self._base_geo(num_image, path=path, geo_function=draw_circle)

    def squares(self, num_image, path="data/squares"):
        def draw_square(draw, x, y, r):
            draw.rectangle(((x-r, y-r), (x+r, y+r)),
                           fill=self._get_random_color())
        return self._base_geo(num_image, path=path, geo_function=draw_square)

    def _base_geo(self, num_image, path, geo_function):
        locations = []
        low = ceil(self.size[0]/3)
        high = floor(self.size[0]/1.5)
        for id in range(num_image):
            x = randint(low, high) # randint(0, self.size[0])
            y = randint(low, high) # randint(0, self.size[1])
            r = randint(ceil(self.size[0]/10), self.size[0]/4)

            img = Image.new('RGB', self.size, (255, 255, 255))
            draw = ImageDraw.Draw(img)
            geo_function(draw, x, y, r)

            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            img.save(f"{path}/image{id}.png", "PNG")
            locations.append([x, y, 2*r, 2*r])
        return locations

    def _get_random_color(self):
        return (randint(0, 255), randint(0, 255), randint(0, 255), 255)


gen_image = Gen_image((128, 128))
circle_locations_train = gen_image.circles(512, "data/circles_train")
save_csv(circle_locations_train, './data/circles_train/locations.csv')

circle_locations_test = gen_image.circles(64, "data/circles_test")
save_csv(circle_locations_test, './data/circles_test/locations.csv')

circle_locations_val = gen_image.circles(64, "data/circles_val")
save_csv(circle_locations_val, './data/circles_val/locations.csv')

# square_locations = gen_image.squares(64, "data/squares")
# save_csv(square_locations, './data/squares/locations.csv')

