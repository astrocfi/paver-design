import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

np.set_printoptions(linewidth=np.inf)

cur_x = 0
cur_y = -2

cur_pat = 0

DRIVEWAY_SIZE = 990 # sq ft
SLOP = 1.0
TOTAL_DRIVEWAY_SIZE = DRIVEWAY_SIZE * SLOP
TILE_GRANULARITY = 3

TILES_PER_PAL = {
    (2,2): 448,
    (2,3): 280,
    (2,4): 224,
    (3,4): 160,
    (4,4): 128
}

COLORS = {
    (2,2): (255, 255, 128),
    (2,3): (255, 160, 0),
    (2,4): (128, 255, 128),
    (3,4): (255, 192, 200),
    (4,4): (128, 255, 255)
}

patterns = []

sys.argv.pop(0)
png_filename = sys.argv.pop(0)
width = int(sys.argv.pop(0))
height = int(sys.argv.pop(0))
step1_x = int(sys.argv.pop(0))
step1_y = int(sys.argv.pop(0))
step2_x = int(sys.argv.pop(0))
step2_y = int(sys.argv.pop(0))
do_random = False

# Read in the patterns from the source text files
for pat_filename in sys.argv:
    if pat_filename == 'random':
        do_random = True
        continue
    with open(pat_filename, 'r') as fp:
        lines = [x.strip() for x in fp.readlines()]
    rows = len(lines)
    columns = len(lines[0])
    pat = np.zeros((rows, columns), dtype=int)
    pat_width = np.zeros((rows, columns), dtype=int)
    pat_height = np.zeros((rows, columns), dtype=int)
    for row in range(rows):
        for column in range(columns):
            c = lines[row][column]
            if c != '.':
                pat[row,column] = ord(lines[row][column])
    patterns.append((pat, pat_width, pat_height))

# Calculate the brick sizes for pallet cost calculation and also
# augment the pattern list with sizes for each element
brick_sizes = {}
for pattern_num, pattern in enumerate(patterns):
    pat_data = pattern[0].copy() # We're going to mutate it
    while True:
        wh = np.where(pat_data != 0)
        if len(wh[0]) == 0:
            break
        y, x = wh[0][0], wh[1][0] # Top left of tile
        for tile_width in range(1, pat_data.shape[1]-x):
            if pat_data[y][x] != pat_data[y][x+tile_width]:
                break
        else:
            tile_width = pat_data.shape[1]-x
        for tile_height in range(1, pat_data.shape[0]-y):
            if pat_data[y][x] != pat_data[y+tile_height][x]:
                break
        else:
            tile_height = pat_data.shape[0]-y
        pat_data[y:y+tile_height, x:x+tile_width] = 0
        tile_width2, tile_height2 = tile_width, tile_height
        if tile_width2 > tile_height2:
            tile_width2, tile_height2 = tile_height2, tile_width2
        pattern[1][y:y+tile_height, x:x+tile_width] = tile_width2
        pattern[2][y:y+tile_height, x:x+tile_width] = tile_height2
        # print('Pattern', pattern_num, 'Size', tile_width2, tile_height2)
        if (tile_width2, tile_height2) not in brick_sizes:
            brick_sizes[(tile_width2, tile_height2)] = 1
        else:
            brick_sizes[(tile_width2, tile_height2)] += 1

print('Total bricks for all patterns:')
total_sq_ft = 0.
for brick_size in sorted(brick_sizes):
    num_bricks = brick_sizes[brick_size]
    print(f'  {brick_size[0]*TILE_GRANULARITY:2d} x '
          f'{brick_size[1]*TILE_GRANULARITY:2d}  = {num_bricks}')
    total_sq_ft += (num_bricks *
                    brick_size[0] * TILE_GRANULARITY *
                    brick_size[1] * TILE_GRANULARITY / 144)
print(f'Total sq ft per grouping {total_sq_ft:.3f}')
num_copies = np.ceil(DRIVEWAY_SIZE / total_sq_ft)
print(f'For {DRIVEWAY_SIZE} sq ft need {num_copies:.0f} copies')

print(f'Pallet count (wastage {SLOP:.4f}):')
for brick_size in sorted(brick_sizes):
    num_bricks = brick_sizes[brick_size]
    total_bricks = num_bricks * num_copies * SLOP
    num_pallets = total_bricks / TILES_PER_PAL[brick_size]
    print(f'  {brick_size[0]*TILE_GRANULARITY:2d} x '
          f'{brick_size[1]*TILE_GRANULARITY:2d} = {num_pallets:.3f} pallets')

#######

final_rows = height
final_cols = width
tile_rows = patterns[0][0].shape[0]
tile_cols = patterns[0][0].shape[1]
print(f'Final rows {final_rows} final cols {final_cols} '
      f'tile rows {tile_rows} tile cols {tile_cols}')

result = np.zeros((final_rows, final_cols), dtype=int)
result_color = np.zeros((final_rows, final_cols, 3), dtype=int)

unique_id = 1000

while cur_y < final_rows+tile_rows*int(final_cols/tile_cols):
    last_start_x = cur_x
    last_start_y = cur_y
    last_cur_pat = cur_pat
    while cur_x < final_cols+tile_cols:
        if (cur_x+tile_cols >= 0 and cur_x < final_cols and
            cur_y+tile_rows >= 0 and cur_y < final_rows):
            start_y = max(-cur_y, 0)
            end_y = min(cur_y+tile_rows, final_rows)-cur_y
            start_x = max(-cur_x, 0)
            end_x = min(cur_x+tile_cols, final_cols)-cur_x
            if do_random:
                cur_pat = random.randrange(len(patterns))
            # print(f'CURX {cur_x:5d} CURY {cur_y:5d} PAT {cur_pat:5d}')
            result[max(cur_y, 0):max(cur_y, 0)+end_y-start_y,
                   max(cur_x, 0):max(cur_x, 0)+end_x-start_x] = \
                patterns[cur_pat][0][start_y:end_y, start_x:end_x] + unique_id
            tile_widths = patterns[cur_pat][1][start_y:end_y, start_x:end_x]
            tile_heights = patterns[cur_pat][2][start_y:end_y, start_x:end_x]
            for tw in range(tile_widths.shape[1]):
                for th in range(tile_heights.shape[0]):
                    if tile_widths[th,tw] != 0:
                        color = COLORS[(tile_widths[th,tw], tile_heights[th,tw])]
                        result_color[max(cur_y, 0)+th, max(cur_x, 0)+tw] = color
        cur_x += step1_x
        cur_y += step1_y
        unique_id += 1000
        cur_pat = (cur_pat+1)%len(patterns)
        # if cur_pat == 0: break
    # if cur_pat == 0: break
    cur_x = last_start_x + step2_x
    cur_y = last_start_y + step2_y
    cur_pat = (last_cur_pat+1)%len(patterns)

####

longest_h = 0
for y in range(final_rows-1):
    trans = (result[y,:-1] != result[y+1,:-1])
    longest_h = max(longest_h, max([sum(g) for b, g in itertools.groupby(trans)]))
longest_v = 0
for x in range(final_cols-1):
    trans = (result[:-1,x] != result[:-1,x+1])
    longest_v = max(longest_v, max([sum(g) for b, g in itertools.groupby(trans)]))

print('LONGEST H RUN', longest_h)
print('LONGEST V RUN', longest_v)

####

img_scale = 10
img = np.zeros((final_rows*img_scale+1, final_cols*img_scale+1, 3), dtype=int) + 255

# First color the blocks
for y in range(final_rows):
    for x in range(final_cols):
        img[y*img_scale:(y+1)*img_scale, x*img_scale:(x+1)*img_scale] = \
            result_color[y,x]
# Then draw the black lines
for y in range(final_rows):
    for x in range(final_cols):
        if x != final_cols-1 and result[y,x] != result[y,x+1]:
            img[y*img_scale:(y+1)*img_scale, (x+1)*img_scale, :] = 0
        if y != final_rows-1 and result[y,x] != result[y+1,x]:
            img[(y+1)*img_scale, x*img_scale:(x+1)*img_scale, :] = 0

img[ 0,  :, :] = 0
img[-1,  :, :] = 0
img[ :,  0, :] = 0
img[ :, -1, :] = 0

plt.figure(figsize=(10,7.5))
pltimg = plt.imshow(img)
plt.axis('off')
plt.tight_layout()
if png_filename == 'screen':
    plt.show()
else:
    plt.savefig(png_filename)
