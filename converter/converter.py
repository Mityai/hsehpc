import sys
import argparse

from PIL import Image


def image_to_rgb(in_file, out_file):
  img = Image.open(in_file)
  rgb = img.convert('RGB')

  out = open(out_file, 'w')
  print('{} {}'.format(rgb.height, rgb.width), file=out)
  for channel in range(3):
    for y in range(rgb.height):
      for x in range(rgb.width):
        print('{} '.format(rgb.getpixel((x, y))[channel]), end='', file=out)
      print(file=out)


def rgb_to_image(in_file, out_file):
  f = open(in_file, 'r')
  n, m = map(int, f.readline().split())
  rgb = []
  for channel in range(3):
    rgb.append([])
    for idx in range(n):
      rgb[channel].append(list(map(int, f.readline().split())))

  img = Image.new("RGB", (m, n))
  for y in range(n):
    for x in range(m):
      img.putpixel((x, y), (rgb[0][y][x], rgb[1][y][x], rgb[2][y][x]))
  img.save(open(out_file, 'w'))


def parse_args(argv):
  parsers = argparse.ArgumentParser()
  subs = parsers.add_subparsers()
  image_to_rgb_parser = subs.add_parser('image_to_rgb')
  image_to_rgb_parser.add_argument('-i', '--input', required=True, help='input image')
  image_to_rgb_parser.add_argument('-o', '--output', required=True, help='output file')
  image_to_rgb_parser.set_defaults(func=image_to_rgb)
  rgb_to_image_parser = subs.add_parser('rgb_to_image')
  rgb_to_image_parser.add_argument('-i', '--input', required=True, help='input file with rgb values')
  rgb_to_image_parser.add_argument('-o', '--output', required=True, help='output image')
  rgb_to_image_parser.set_defaults(func=rgb_to_image)
  return parsers.parse_args(argv)


def main(argv):
  args = parse_args(argv)
  args.func(args.input, args.output)
  

if __name__ == '__main__':
  main(sys.argv[1:])
