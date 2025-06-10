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
    noise_array = np.zeros((size, size), dtype=np.float32)
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
            noise_array[x, y] = noise_value
    return noise_array

def generate_stacked_noise(size, depth) -> np.ndarray:
    """
    Generate a 3D array of stacked noise arrays

    :param height: Height of the noise array.
    :param width: Width of the noise array.
    :param depth: Depth of the noise array.
    :return: 3D numpy array of noise values.
    """
    scale = 1.0
    octaves = 4
    persistence = 0.5
    lacunarity = 2.0

    depth = 5000
    stacked_noise = np.zeros((depth, size, size), dtype=np.float32)
    status = ["\\", "|", "/", "-"]
    for d in range(depth):
        stacked_noise[d] = generate_noise(size, scale, octaves, persistence, lacunarity, seed=d)
        print(f"\r{status[(d // 120) % len(status)]} Generating...", end="", flush=True)
    print("")

    # Transpose to get the shape (width, height, depth)
    return np.transpose(stacked_noise, (2, 1, 0))

def save_texture_info(noise_array, filename="tex_info.txt"):
    """
    Save the noise array to a file.

    :param noise_array: 3D numpy array of noise values.
    :param filename: Name of the file to save the noise array.
    """
    np.save(filename, noise_array)
    print(f"Noise array saved to {filename}")

def save_texture_txt(noise_array, filename="tex_info.txt"):
    """
    Save the noise array to a text file.

    :param noise_array: 3D numpy array of noise values.
    :param filename: Name of the file to save the noise array.
    """
    with open(filename, 'w') as f:
        for d in range(noise_array.shape[2]):
            f.write(f"Depth {d}:\n")
            for y in range(noise_array.shape[1]):
                row = " ".join(f"{noise_array[x, y, d]:.4f}" for x in range(noise_array.shape[0]))
                f.write(row + "\n")
            f.write("\n")
    print(f"Noise array saved to {filename}")

def save_texture_png_greyscale(noise_array, slice):
    """
    Save a slice of the noise array as a PNG image.

    :param noise_array: 3D numpy array of noise values.
    :param slice: Index of the slice to save.
    :param filename: Name of the file to save the PNG image.
    """
    # Get just the slice you want to display
    slice_data = noise_array[:, :, slice]
    
    # Normalize ONLY this slice
    norm = Normalize(vmin=np.min(slice_data), vmax=np.max(slice_data))
    plt.imshow(slice_data, cmap='gray', norm=norm)
    plt.axis('off')
    plt.savefig(f"noise_slice_{slice}_greyscale.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Slice {slice} saved as noise_slice_{slice}.png")

def save_texture_red_image(noise_array, filename="noise_red_channel.png"):
    """
    Save a 2D noise array as a red channel image.
    """

    red_channel = noise_array
    red_image = np.zeros((red_channel.shape[0], red_channel.shape[1], 3), dtype=np.uint8)
    red_image[:, :, 0] = (red_channel * 255).astype(np.uint8)  # Red channel
    plt.imsave(filename, red_image, format='png')
    print("Red channel saved as noise_red_channel.png")

def save_texture_blue_image(noise_array, filename="noise_blue_channel.png"):
    """
    Save a 2D noise array as a blue channel image.
    """

    blue_channel = noise_array
    blue_image = np.zeros((blue_channel.shape[0], blue_channel.shape[1], 3), dtype=np.uint8)
    blue_image[:, :, 2] = (blue_channel * 255).astype(np.uint8)  # Blue channel
    plt.imsave(filename, blue_image, format='png')
    print("Blue channel saved as noise_blue_channel.png")

def save_texture_green_image(noise_array, filename="noise_green_channel.png"):
    """
    Save a 2D noise array as a green channel image.
    """
    green_channel = noise_array
    green_image = np.zeros((green_channel.shape[0], green_channel.shape[1], 3), dtype=np.uint8)
    green_image[:, :, 1] = (green_channel * 255).astype(np.uint8)  # Green channel
    plt.imsave(filename, green_image, format='png')
    print("Green channel saved as noise_green_channel.png")

def save_texture_alpha_image(noise_array, filename="noise_alpha_channel.png"):
    """
    Save a texture alpha image (with base color black)

    """
    alpha_channel = noise_array
    alpha_image = np.zeros((alpha_channel.shape[0], alpha_channel.shape[1], 4), dtype=np.uint8)
    alpha_image[:, :, 3] = (alpha_channel * 255).astype(np.uint8)  # Alpha channel
    plt.imsave(filename, alpha_image, format='png')
    print("Alpha channel saved as noise_alpha_channel.png")

def save_rexture_rgba_image(noise_array4, filename="noise_rgba_image.png"):
    """
    Save a 4D noise array as a PNG image with RGBA channels.

    :param noise_array4: 4D numpy array of noise values.
    :param filename: Name of the file to save the PNG image.
    """
    rgba_image = np.zeros((noise_array4[0].shape[0], noise_array4[0].shape[1], 4), dtype=np.uint8)
    rgba_image[:, :, 0] = (noise_array4[0] * 255).astype(np.uint8)  # Red channel
    rgba_image[:, :, 1] = (noise_array4[1] * 255).astype(np.uint8)  # Green channel
    rgba_image[:, :, 2] = (noise_array4[2] * 255).astype(np.uint8)  # Blue channel
    rgba_image[:, :, 3] = (noise_array4[3] * 255).astype(np.uint8)  # Alpha channel

    plt.imsave(filename, rgba_image, format='png')
    print(f"RGBA image saved as {filename}")

def save_texture_png_rgba(noise_array_4, prefix="noise") -> None:
    """
    Save a 4D noise array as a PNG image with RGBA channels.

    :param noise_array_4: 4D numpy array of noise values.
    """
    # Save red only
    red_channel = noise_array_4[0]
    blue_channel = noise_array_4[1]
    green_channel = noise_array_4[2]
    alpha_channel = noise_array_4[3]

    save_texture_red_image(red_channel, "{}_red_channel.png".format(prefix))
    save_texture_blue_image(blue_channel, "{}_blue_channel.png".format(prefix))
    save_texture_green_image(green_channel, "{}_green_channel.png".format(prefix))
    save_texture_alpha_image(alpha_channel, "{}_alpha_channel.png".format(prefix))
    save_rexture_rgba_image(noise_array_4, "{}_rgba_image.png".format(prefix))


def save_rgba_binary(noise_array4, filename="noise_rgba_binary.txt"):
    """
    Save a 4D noise array as a binary file.

    :param noise_array4: 4D numpy array of noise values.
    :param filename: Name of the file to save the binary data.
    """
    with open(filename, 'wb') as f:
        for channel in noise_array4:
            channel.tofile(f)
    print(f"RGBA binary data saved to {filename}")

def rgba_binary_to_noise_array4(filepath):
    """
    Load a 4D noise array from a binary file.

    :param filepath: Path to the binary file containing the noise data.
    :return: 4D numpy array of noise values.
    """
    with open(filepath, 'rb') as f:
        data = f.read()
    
    # Assuming each channel is of size (size, size)
    size = 64  # Adjust this based on your noise generation parameters
    num_channels = 4
    noise_array4 = np.frombuffer(data, dtype=np.float32).reshape((num_channels, size, size))
    
    return noise_array4

def save_texture_png_bw(noise_array, slice):
    # Save texture in black and white

    """
    Save a slice of the noise array as a black and white PNG image (no grey)
    :param noise_array: 3D numpy array of noise values.
    :param slice: Index of the slice to save.
    """
    slice_data = noise_array[:, :, slice]
    # Normalize the slice data to 0-255
    norm = Normalize(vmin=np.min(slice_data), vmax=np.max(slice_data))
    bw_data = (norm(slice_data) * 255).astype(np.uint8)
    # Convert to binary (black and white)
    bw_data = np.where(bw_data > 127, 255, 0)
    plt.imsave(f"noise_slice_{slice}_bw.png", bw_data, cmap='gray', format='png')
    print(f"Slice {slice} saved as noise_slice_{slice}_bw.png")

def generate_rgba_noise(size):
    return [
        generate_noise(size, scale=1.0, octaves=4, persistence=0.5, lacunarity=2.0, seed=0),
        generate_noise(size, scale=1.0, octaves=4, persistence=0.5, lacunarity=2.0, seed=1),
        generate_noise(size, scale=1.0, octaves=4, persistence=0.5, lacunarity=2.0, seed=2),
        generate_noise(size, scale=1.0, octaves=4, persistence=0.5, lacunarity=2.0, seed=3)
    ]

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
    stacked_noise = np.zeros((4, depth, size, size), dtype=np.float32)
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
    normalized = np.clip((stacked_rgba_noise + 1.0) * 0.5, 0, 1)
    uint8_data = (normalized * 255).astype(np.uint8)
    
    with open("stacked_rgba_noise.bin", 'wb') as f:
        uint8_data.tofile(f)
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
    noise_array4 = noise_array.reshape((size, size, depth, num_channels))
    
    return noise_array4

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
    
    normalized_rgba_noise = np.clip((stacked_rgba_noise + 1.0) * 0.5, 0, 1)
    # Print first 10 normalized rgba values
    for i in range(10):
        print(f"Normalized RGBA value at (0, 0, {i}): {normalized_rgba_noise[0, 0, i, :]}")

    for i in range(0, 10, 1):  # Iterate over depth
        # rgba_image = np.zeros((stacked_rgba_noise.shape[0], stacked_rgba_noise.shape[1], 4), dtype=np.uint8)
        # rgba_image[:, :, 0] = (stacked_rgba_noise[:, :, i, 0] * 255).astype(np.uint8)  # Red channel
        # rgba_image[:, :, 1] = (stacked_rgba_noise[:, :, i, 1] * 255).astype(np.uint8)  # Green channel
        # rgba_image[:, :, 2] = (stacked_rgba_noise[:, :, i, 2] * 255).astype(np.uint8)  # Blue channel
        # rgba_image[:, :, 3] = (stacked_rgba_noise[:, :, i, 3] * 255).astype(np.uint8)  # Alpha channel

        rgba_image = (normalized_rgba_noise[:, :, i, :] * 255).astype(np.uint8)  # Convert to uint8
        rgba_image[:, :, 0] = (normalized_rgba_noise[:, :, i, 0] * 255).astype(np.uint8)  # Red channel
        rgba_image[:, :, 1] = (normalized_rgba_noise[:, :, i, 1] * 255).astype(np.uint8)  # Green channel
        rgba_image[:, :, 2] = (normalized_rgba_noise[:, :, i, 2] * 255).astype(np.uint8)  # Blue channel
        rgba_image[:, :, 3] = (normalized_rgba_noise[:, :, i, 3] * 255).astype(np.uint8)  # Alpha channel
        
        plt.imsave(f"rgba_noise_channels/{prefix}_depth_{i}.png", rgba_image, format='png')
        print(f"Saved {prefix}_depth_{i}.png")



if __name__ == "__main__":
    # noise = generate_stacked_noise(64, 500)
    # print("Generated noise shape:", noise.shape)
    # print("Noise sample at (0, 0, 0):", noise[0, 0, 0])
    # save_texture_info(noise)
    # save_texture_txt(noise)
    # save_texture_png_greyscale(noise, 0)  # Save the first slice as an example
    # save_texture_png_bw(noise, 0)  # Save the first slice as an example

    # noise = generate_rgba_noise(64)
    # print("Generated noise shape:", [n.shape for n in noise])
    # save_texture_png_rgba(noise, "noise")
    # save_rgba_binary(noise)
    # noise_loaded = rgba_binary_to_noise_array4("noise_rgba_binary.txt")
    # print("Loaded noise shape:", [n.shape for n in noise_loaded])
    # save_texture_png_rgba(noise_loaded, "loaded_noise")

    stacked = generate_stacked_rgba_noise(64)
    print("Generated stacked RGBA noise shape:", stacked.shape)

    print("Stacked mins and maxes:", 
          [np.min(stacked[:, :, :, i]) for i in range(4)],
          [np.max(stacked[:, :, :, i]) for i in range(4)])
    export_stacked_rgba_noise_to_png(stacked, "stacked_noise")
    export_stacked_rgba_noise(stacked)
    loaded_stacked = load_stacked_rgba_noise("stacked_rgba_noise.bin")
    print("Loaded stacked RGBA noise shape:", loaded_stacked.shape)
    #export_stacked_rgba_noise_to_png(loaded_stacked, "stacked_noise")