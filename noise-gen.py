# Generate Simplex Noise
import numpy as np
from noise import snoise2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def generate_noise(size, scale=1.0, octaves=1, persistence=0.5, lacunarity=2.0, seed=0) -> np.ndarray:
    """
    Generate a 4D Simplex noise array.

    :param size: Tuple of (width, height) for the noise array.
    :param scale: Scale of the noise.
    :param octaves: Number of octaves for the noise.
    :param persistence: Amplitude persistence for each octave.
    :param lacunarity: Frequency increase for each octave.
    :return: 2D numpy array of noise values.
    """
    noise_array = np.zeros((size, size), dtype=np.uint8)
    for x in range(size):
        for y in range(size):
            noise_value = snoise2(
                x / scale,
                y / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=size,
                repeaty=size,
                base=seed
            )
            noise_array[x, y] = int((noise_value + 1.0) * 127.5)  # Scale to [0, 255]
    return noise_array

def generate_stacked_rgba_noise(size):
    """
    Generate a 3D array of stacked RGBA noise arrays.

    :param size: Size of the noise array.
    :return: 4D numpy array of noise values.
    """
    scale = 1.0
    octaves = 4
    persistence = 0.5
    lacunarity = 2.0

    depth = 5000
    stacked_noise = np.zeros((4, depth, size, size), dtype=np.uint8)
    status = ["\\", "|", "/", "-"]
    
    for d in range(depth):
        for channel in range(4):
            stacked_noise[channel, d] = generate_noise(size, scale, octaves, persistence, lacunarity, seed=d * 1000 + channel * 250)
        print(f"\r{status[(d // 120) % len(status)]} Generating...", end="", flush=True)

    # Convert to (height, width, depth, 4) as expected by RGBA
    stacked_transposed = np.transpose(stacked_noise, (2, 3, 1, 0))

    for (i, channel) in enumerate(['R', 'G', 'B', 'A']):
        print(f"Channel {channel} min: {np.min(stacked_transposed[:, :, :, i])}, max: {np.max(stacked_transposed[:, :, :, i])}")

    # Printr the first 10 RGBA values

    return stacked_transposed


def export_stacked_rgba_noise(stacked_rgba_noise):
    # Normalize from [-1,1] to [0,1] then to [0,255]
    with open("stacked_rgba_noise.tex.bin", 'wb') as f:
        stacked_rgba_noise.tofile(f)
    print("Stacked RGBA noise array saved to stacked_rgba_noise.bin")


def load_stacked_rgba_noise(filepath):
    """
    Load a stacked RGBA noise array from a binary file.

    :param filepath: Path to the binary file containing the noise data.
    :return: 4D numpy array of noise values.
    """
    size = 64  # Adjust this based on your noise generation parameters
    num_channels = 4
    depth = 5000
    
    # Calculate expected file size
    expected_size = size * size * depth * num_channels
    print(f"Expected file size: {expected_size} bytes")
    
    # Load the data
    noise_array = np.fromfile(filepath, dtype=np.uint8)
    print(f"Actual data size: {len(noise_array)} bytes")
    
    # Check if sizes match
    if len(noise_array) != expected_size:
        raise ValueError(f"File size mismatch. Expected {expected_size}, got {len(noise_array)}")
    
    # Reshape to (height, width, depth, channels)
    stacked_noise = noise_array.reshape((size, size, depth, num_channels))
    
    return stacked_noise

def export_stacked_rgba_noise_to_png(stacked_rgba_noise, prefix="stacked_noise"):
    """
    Export a stacked RGBA noise array to PNG files.

    :param stacked_rgba_noise: 4D numpy array of noise values.
    :param prefix: Prefix for the output PNG files.
    """
    # Print pre-noramlized values
    for i in range(10):
        print(f"RGBA value at (0, 0, {i}): {stacked_rgba_noise[0, 0, i, :]}")
    print("")

    for i in range(0, 10, 1):  # Iterate over depth
        rgba_image = stacked_rgba_noise[:, :, i, :]

        
        plt.imsave(f"rgba_noise_channels/{prefix}_depth_{i}.png", rgba_image, format='png')
        print(f"Saved {prefix}_depth_{i}.png")



if __name__ == "__main__":
    stacked = generate_stacked_rgba_noise(64)
    print("Generated stacked RGBA noise shape:", stacked.shape)

    print("Stacked mins and maxes:", 
          [np.min(stacked[:, :, :, i]) for i in range(4)],
          [np.max(stacked[:, :, :, i]) for i in range(4)])
    export_stacked_rgba_noise_to_png(stacked, "stacked_noise")
    export_stacked_rgba_noise(stacked)
    loaded_stacked = load_stacked_rgba_noise("stacked_rgba_noise.bin")
    print("Loaded stacked RGBA noise shape:", loaded_stacked.shape)
    export_stacked_rgba_noise_to_png(loaded_stacked, "loaded_stacked_noise")
    #export_stacked_rgba_noise_to_png(loaded_stacked, "stacked_noise")