import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import hashlib

# Define 8 pastel colors
PASTEL_COLORS = np.array(
    [
        [1.0, 0.8, 0.8],  # Light pink
        [0.8, 1.0, 0.8],  # Light green
        [0.8, 0.8, 1.0],  # Light blue
        [1.0, 1.0, 0.8],  # Light yellow
        [1.0, 0.8, 1.0],  # Light magenta
        [0.8, 1.0, 1.0],  # Light cyan
        [1.0, 0.9, 0.8],  # Light peach
        [0.9, 0.8, 1.0],  # Light lavender
    ]
)


def get_random_pastel_color():
    """Get a random pastel color from the predefined set"""
    return PASTEL_COLORS[np.random.randint(0, len(PASTEL_COLORS))]


def generate_voronoi_squares(width, height, num_seeds=50):
    """Voronoi-based squares using Manhattan distance"""
    seeds = np.random.randint(0, min(width, height), (num_seeds, 2))
    colors = [get_random_pastel_color() for _ in range(num_seeds)]

    image = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            distances = np.abs(seeds[:, 0] - x) + np.abs(seeds[:, 1] - y)
            closest_seed = np.argmin(distances)
            image[y, x] = colors[closest_seed]

    return image


def generate_grid_subdivision(
    width, height, initial_grid_size=64, subdivision_prob=0.7
):
    """Fixed grid subdivision algorithm"""
    image = np.zeros((height, width, 3))

    def subdivide_region(x, y, w, h, depth=0):
        if w <= 8 or h <= 8 or (depth > 0 and np.random.random() > subdivision_prob):
            # Don't subdivide, fill with single color
            color = get_random_pastel_color()
            end_x = min(x + w, width)
            end_y = min(y + h, height)
            image[y:end_y, x:end_x] = color
        else:
            # Subdivide into 4 smaller rectangles
            half_w = w // 2
            half_h = h // 2
            subdivide_region(x, y, half_w, half_h, depth + 1)
            subdivide_region(x + half_w, y, w - half_w, half_h, depth + 1)
            subdivide_region(x, y + half_h, half_w, h - half_h, depth + 1)
            subdivide_region(x + half_w, y + half_h, w - half_w, h - half_h, depth + 1)

    # Start with the full canvas
    subdivide_region(0, 0, width, height)
    return image


def generate_noise_squares(width, height):
    """Noise-based squares"""

    def simple_noise(x, y, scale):
        return (np.sin(x * scale) * np.cos(y * scale) + 1) / 2

    image = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            # Use noise to determine square size
            size_noise = simple_noise(x, y, 0.02)
            square_size = int(4 + size_noise * 20)

            # Snap to grid based on square size
            grid_x = (x // square_size) * square_size
            grid_y = (y // square_size) * square_size

            # Use grid position to select pastel color
            color_index = (grid_x // square_size + grid_y // square_size) % len(
                PASTEL_COLORS
            )
            image[y, x] = PASTEL_COLORS[color_index]

    return image


def generate_hash_squares(width, height):
    """Hash-based approach"""
    image = np.zeros((height, width, 3))

    def hash_to_color_index(text):
        """Convert hash to pastel color index"""
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest()[0:2], 16)
        return hash_int % len(PASTEL_COLORS)

    def hash_to_size(text, min_size=4, max_size=24):
        """Convert hash to square size"""
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest()[2:4], 16)
        return min_size + (hash_int % (max_size - min_size + 1))

    for y in range(height):
        for x in range(width):
            # Determine square size based on region
            region_key = f"size_{x//16}_{y//16}"
            square_size = hash_to_size(region_key)

            # Snap to grid
            grid_x = (x // square_size) * square_size
            grid_y = (y // square_size) * square_size

            # Generate color based on grid position
            color_key = f"color_{grid_x}_{grid_y}"
            color_index = hash_to_color_index(color_key)

            image[y, x] = PASTEL_COLORS[color_index]

    return image


# Generate test images
width, height = 256, 256
np.random.seed(42)  # For reproducible results

print("Generating Voronoi squares...")
voronoi_img = generate_voronoi_squares(width, height)

print("Generating grid subdivision...")
grid_img = generate_grid_subdivision(width, height)

print("Generating noise-based squares...")
noise_img = generate_noise_squares(width, height)

print("Generating hash-based squares...")
hash_img = generate_hash_squares(width, height)

# Display results
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(voronoi_img)
axes[0, 0].set_title("Voronoi Squares (Manhattan Distance)")
axes[0, 0].axis("off")

axes[0, 1].imshow(grid_img)
axes[0, 1].set_title("Grid Subdivision")
axes[0, 1].axis("off")

axes[1, 0].imshow(noise_img)
axes[1, 0].set_title("Noise-based Squares")
axes[1, 0].axis("off")

axes[1, 1].imshow(hash_img)
axes[1, 1].set_title("Hash-based Squares")
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()
