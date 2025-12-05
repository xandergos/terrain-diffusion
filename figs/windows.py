from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def load_terrain_grayscale(target_size: int, project_root: Path) -> Image.Image:
    src = project_root / "example_terrain.png"
    img = Image.open(src).convert("L")
    return img.resize((target_size, target_size), resample=Image.BICUBIC)


def gaussian_noise_image(size: int, mean: float = 127.5, std: float = 30.0) -> Image.Image:
    noise = np.random.normal(loc=mean, scale=std, size=(size, size))
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noise, mode="L")


def blend_images(noise_img: Image.Image, base_img: Image.Image, alpha: float = 0.5) -> Image.Image:
    # Linear interpolation: out = alpha * base + (1 - alpha) * noise
    base = np.array(base_img, dtype=np.float32)
    noise = np.array(noise_img, dtype=np.float32)
    out = alpha * base + (1.0 - alpha) * noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def draw_interior_square_border(draw: ImageDraw.ImageDraw, x: int, y: int, size: int = 256, width: int = 5, color: int = 0) -> None:
    # Draw four filled bars fully inside the size x size region.
    left = x
    top = y
    right = x + size - 1
    bottom = y + size - 1
    # Top
    draw.rectangle((left, top, right, top + width - 1), fill=color)
    # Bottom
    draw.rectangle((left, bottom - width + 1, right, bottom), fill=color)
    # Left
    draw.rectangle((left, top, left + width - 1, bottom), fill=color)
    # Right
    draw.rectangle((right - width + 1, top, right, bottom), fill=color)


def overlay_grid(img: Image.Image, grid_rows: int = 6, grid_cols: int = 6, box_size: int = 256, stride: int = 128, border: int = 5) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    # Compute positions with given stride, but ensure coverage by including the last start at size - box_size
    xs = list(range(0, max(1, w - box_size + 1), stride))
    ys = list(range(0, max(1, h - box_size + 1), stride))
    last_x = w - box_size
    last_y = h - box_size
    if xs[-1] != last_x:
        xs.append(last_x)
    if ys[-1] != last_y:
        ys.append(last_y)
    for y in ys:
        for x in xs:
            draw_interior_square_border(draw, x, y, size=box_size, width=border, color=0)
    return out


def main() -> None:
    base_size = 256
    noise_size = base_size + 64
    terrain_crop_size = base_size - 64

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    out_path = script_path.parent / "windows.png"

    # Load terrain at base resolution, then center-crop to 768x768 for output
    terrain_base = load_terrain_grayscale(base_size, project_root)
    offset = (base_size - terrain_crop_size) // 2
    crop_box = (offset, offset, offset + terrain_crop_size, offset + terrain_crop_size)
    terrain_cropped = terrain_base.crop(crop_box)

    noise = gaussian_noise_image(noise_size)
    # Resize inputs to keep mixed at 1024x1024 (use 1024 terrain, resize noise)
    terrain_for_mix = terrain_base
    noise_for_mix = noise.resize((base_size, base_size), resample=Image.NEAREST)
    mixed = blend_images(noise_for_mix, terrain_for_mix, alpha=0.5)

    # Save separately
    noise.save(script_path.parent / "windows_noise.png")
    mixed.save(script_path.parent / "windows_mixed.png")
    terrain_cropped.save(script_path.parent / "windows_terrain.png")


if __name__ == "__main__":
    main()

