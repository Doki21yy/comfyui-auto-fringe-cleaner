import torch
import torch.nn.functional as F


def _match_batch(tensor, batch_size):
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.repeat(batch_size, *([1] * (tensor.dim() - 1)))
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]

    pad_count = batch_size - tensor.shape[0]
    last = tensor[-1:].repeat(pad_count, *([1] * (tensor.dim() - 1)))
    return torch.cat([tensor, last], dim=0)


def _resize_mask(mask, height, width):
    if mask.shape[-2] == height and mask.shape[-1] == width:
        return mask
    resized = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False)
    return resized[:, 0, :, :]


def _resize_image(image, height, width):
    if image.shape[1] == height and image.shape[2] == width:
        return image
    resized = F.interpolate(image.movedim(-1, 1), size=(height, width), mode="bilinear", align_corners=False)
    return resized.movedim(1, -1)


def _max_filter(mask, pixels):
    pixels = int(pixels)
    if pixels <= 0:
        return mask
    kernel = pixels * 2 + 1
    return F.max_pool2d(mask.unsqueeze(1), kernel_size=kernel, stride=1, padding=pixels)[:, 0]


def _min_filter(mask, pixels):
    return -_max_filter(-mask, pixels)


def _box_blur_nhwc(image, pixels):
    pixels = int(pixels)
    if pixels <= 0:
        return image
    kernel = pixels * 2 + 1
    nchw = image.movedim(-1, 1)
    blurred = F.avg_pool2d(nchw, kernel_size=kernel, stride=1, padding=pixels)
    return blurred.movedim(1, -1)


def _masked_local_average(image, source_mask, radius):
    radius = max(int(radius), 1)
    kernel_area = float((radius * 2 + 1) ** 2)
    source = source_mask.unsqueeze(-1)

    numerator = F.avg_pool2d(
        (image * source).movedim(-1, 1),
        kernel_size=radius * 2 + 1,
        stride=1,
        padding=radius,
    ).movedim(1, -1) * kernel_area
    denominator = F.avg_pool2d(
        source_mask.unsqueeze(1),
        kernel_size=radius * 2 + 1,
        stride=1,
        padding=radius,
    ).movedim(1, -1) * kernel_area

    return numerator / denominator.clamp_min(1e-6), denominator


def _inside_edge_weight(alpha, radius, threshold):
    radius = int(radius)
    foreground = (alpha > threshold).to(alpha.dtype)
    if radius <= 0:
        return torch.zeros_like(alpha)

    weight = torch.zeros_like(alpha)
    current = foreground
    for layer in range(radius):
        eroded = _min_filter(current, 1)
        rim = (current - eroded).clamp(0.0, 1.0)
        layer_weight = 1.0 - (layer / max(radius, 1))
        weight = torch.maximum(weight, rim * layer_weight)
        current = eroded
    return weight


def _border_mean(mask):
    height, width = mask.shape[-2], mask.shape[-1]
    border = max(1, min(height, width) // 32)
    pieces = [
        mask[:, :border, :],
        mask[:, -border:, :],
        mask[:, :, :border],
        mask[:, :, -border:],
    ]
    return torch.cat([p.reshape(p.shape[0], -1) for p in pieces], dim=1).mean(dim=1)


def _center_mean(mask):
    height, width = mask.shape[-2], mask.shape[-1]
    y0, y1 = height // 4, height - height // 4
    x0, x1 = width // 4, width - width // 4
    return mask[:, y0:y1, x0:x1].reshape(mask.shape[0], -1).mean(dim=1)


def _mask_to_alpha(mask, mask_mode, height, width, batch_size):
    if mask is None:
        return None

    if mask.dim() == 4:
        mask = mask[..., 0] if mask.shape[-1] == 1 else mask.mean(dim=-1)
    mask = mask.to(dtype=torch.float32).clamp(0.0, 1.0)
    mask = _resize_mask(mask, height, width)
    mask = _match_batch(mask, batch_size)

    if mask_mode == "Load Image MASK (透明=白)":
        return 1.0 - mask
    if mask_mode == "Alpha MASK (物体=白)":
        return mask

    invert = (_border_mean(mask) > _center_mean(mask)).view(batch_size, 1, 1)
    return torch.where(invert, 1.0 - mask, mask)


def _infer_alpha_from_corners(image):
    height, width = image.shape[1], image.shape[2]
    patch = max(2, min(height, width) // 64)
    corners = torch.cat(
        [
            image[:, :patch, :patch].reshape(image.shape[0], -1, 3),
            image[:, :patch, -patch:].reshape(image.shape[0], -1, 3),
            image[:, -patch:, :patch].reshape(image.shape[0], -1, 3),
            image[:, -patch:, -patch:].reshape(image.shape[0], -1, 3),
        ],
        dim=1,
    )
    bg = corners.median(dim=1).values.view(image.shape[0], 1, 1, 3)
    distance = (image - bg).abs().amax(dim=-1)
    alpha = ((distance - 0.03) / 0.12).clamp(0.0, 1.0)
    return _box_blur_nhwc(alpha.unsqueeze(-1), 1)[..., 0].clamp(0.0, 1.0)


def _clean_rgb_with_alpha(
    image,
    alpha,
    clean_strength,
    edge_width,
    sample_radius,
    alpha_threshold,
    solid_threshold,
    protect_texture,
    alpha_contract,
):
    image = image[..., :3].to(dtype=torch.float32).clamp(0.0, 1.0)
    alpha = alpha.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)

    foreground = (alpha > alpha_threshold).to(image.dtype)
    safe_interior = _min_filter(foreground, max(1, int(edge_width)))
    solid = safe_interior * (alpha >= solid_threshold).to(image.dtype)
    fallback_source = (alpha > max(alpha_threshold, 0.12)).to(image.dtype)

    local_clean, clean_den = _masked_local_average(image, solid, sample_radius)
    fallback_clean, _ = _masked_local_average(image, fallback_source, sample_radius)
    local_clean = torch.where(clean_den > 0.5, local_clean, fallback_clean)

    edge_weight = _inside_edge_weight(alpha, edge_width, alpha_threshold)
    semi_weight = ((1.0 - alpha).clamp(0.0, 1.0) ** 0.7) * (alpha > alpha_threshold).to(image.dtype)
    edge_weight = torch.maximum(edge_weight, semi_weight)
    edge_weight = _box_blur_nhwc(edge_weight.unsqueeze(-1), 1)[..., 0]

    texture_delta = (image - local_clean).abs().amax(dim=-1)
    texture_protection = (1.0 - protect_texture * (texture_delta / 0.35).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    blend = (edge_weight * clean_strength * texture_protection).unsqueeze(-1).clamp(0.0, 1.0)
    cleaned = image * (1.0 - blend) + local_clean * blend

    if alpha_contract > 0.0:
        whole = int(alpha_contract)
        frac = alpha_contract - whole
        contracted = _min_filter(alpha, whole) if whole > 0 else alpha
        if frac > 0.0:
            contracted_next = _min_filter(contracted, 1)
            contracted = contracted * (1.0 - frac) + contracted_next * frac
        alpha = contracted.clamp(0.0, 1.0)

    # Fully transparent pixels still carry RGB in many cutout PNGs. ComfyUI IMAGE
    # previews ignore alpha, so clear those pixels too instead of leaving a blue/black halo.
    invisible = (alpha <= alpha_threshold).unsqueeze(-1)
    cleaned = torch.where(invisible, torch.zeros_like(cleaned), cleaned)

    return cleaned.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0)


def _input_images():
    import os

    import folder_paths

    input_dir = folder_paths.get_input_directory()
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    return sorted(folder_paths.filter_files_content_types(files, ["image"]))


def _load_rgba_from_input(image_name):
    import numpy as np
    from PIL import Image, ImageOps

    import folder_paths

    image_path = folder_paths.get_annotated_filepath(image_name)
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGBA")
    rgba = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    rgb = rgba[..., :3][None,]
    alpha = rgba[..., 3][None,]
    return rgb, alpha


class TransparentPNGData:
    def __init__(self, rgb, alpha, filename=None):
        self.rgb = rgb
        self.alpha = alpha
        self.filename = filename


def _save_rgba(images, alpha, filename_prefix, output_dir, output_type, prompt=None, extra_pnginfo=None):
    import json
    import os
    import re

    import numpy as np
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    import folder_paths
    from comfy.cli_args import args

    images = images[..., :3].clamp(0.0, 1.0)
    if alpha.dim() == 4:
        alpha = alpha[..., 0] if alpha.shape[-1] == 1 else alpha.mean(dim=-1)
    alpha = _resize_mask(alpha.clamp(0.0, 1.0), images.shape[1], images.shape[2])
    alpha = _match_batch(alpha, images.shape[0])

    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix,
        output_dir,
        images[0].shape[1],
        images[0].shape[0],
    )
    results = []

    def next_counter():
        max_counter = counter
        if not os.path.isdir(full_output_folder):
            return max_counter
        for existing_file in sorted(os.listdir(full_output_folder)):
            match = re.fullmatch(fr"{re.escape(filename)}_(\d+)_?\.[a-zA-Z0-9]+", existing_file)
            if match:
                max_counter = max(max_counter, int(match.group(1)))
        return max_counter + 1

    for image, alpha_image in zip(images, alpha):
        rgb = (255.0 * image.cpu().numpy()).round().clip(0, 255).astype(np.uint8)
        a = (255.0 * alpha_image.cpu().numpy()).round().clip(0, 255).astype(np.uint8)
        img = Image.fromarray(rgb, mode="RGB")
        if a.shape[::-1] != img.size:
            a = np.array(Image.fromarray(a, mode="L").resize(img.size, Image.LANCZOS))
        img.putalpha(Image.fromarray(a, mode="L"))

        metadata = None
        if not args.disable_metadata:
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    metadata.add_text(key, json.dumps(value))

        file = f"{filename}_{next_counter():05}.png"
        img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
        results.append({"filename": file, "subfolder": subfolder, "type": output_type})

    return results


class AutoCleanTransparentPNG:
    def __init__(self):
        import folder_paths

        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        files = _input_images()
        return {
            "required": {
                "transparent_png": (files, {"image_upload": True}),
                "filename_prefix": ("STRING", {"default": "ComfyUI_fringe_cleaned"}),
                "clean_strength": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "edge_width": (
                    "INT",
                    {"default": 14, "min": 1, "max": 64, "step": 1},
                ),
                "sample_radius": (
                    "INT",
                    {"default": 24, "min": 2, "max": 128, "step": 1},
                ),
                "alpha_threshold": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005},
                ),
                "solid_threshold": (
                    "FLOAT",
                    {"default": 0.92, "min": 0.2, "max": 1.0, "step": 0.005},
                ),
                "protect_texture": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "alpha_contract": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.1},
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "clean_and_save"
    OUTPUT_NODE = True
    CATEGORY = "image/cleanup"
    DESCRIPTION = "Loads a transparent PNG, automatically removes colored fringes using its alpha channel, and saves a transparent PNG."

    def clean_and_save(
        self,
        transparent_png,
        filename_prefix,
        clean_strength,
        edge_width,
        sample_radius,
        alpha_threshold,
        solid_threshold,
        protect_texture,
        alpha_contract,
        prompt=None,
        extra_pnginfo=None,
    ):
        image, alpha = _load_rgba_from_input(transparent_png)
        cleaned, cleaned_alpha = _clean_rgb_with_alpha(
            image,
            alpha,
            clean_strength,
            edge_width,
            sample_radius,
            alpha_threshold,
            solid_threshold,
            protect_texture,
            alpha_contract,
        )
        results = _save_rgba(
            cleaned,
            cleaned_alpha,
            filename_prefix + self.prefix_append,
            self.output_dir,
            self.type,
            prompt,
            extra_pnginfo,
        )
        preview = cleaned * cleaned_alpha.unsqueeze(-1)
        return {"ui": {"images": results}, "result": (preview.clamp(0.0, 1.0),)}


class AutoCleanCompositeOnBackground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transparent_png": ("IMAGE",),
                "background_image": ("IMAGE",),
                "clean_strength": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "edge_width": (
                    "INT",
                    {"default": 14, "min": 1, "max": 64, "step": 1},
                ),
                "sample_radius": (
                    "INT",
                    {"default": 24, "min": 2, "max": 128, "step": 1},
                ),
                "alpha_threshold": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005},
                ),
                "solid_threshold": (
                    "FLOAT",
                    {"default": 0.92, "min": 0.2, "max": 1.0, "step": 0.005},
                ),
                "protect_texture": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "alpha_contract": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.1},
                ),
            },
            "optional": {
                "alpha_from_load_image": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composited_image",)
    FUNCTION = "clean_and_composite"
    CATEGORY = "image/cleanup"
    DESCRIPTION = "Cleans a transparent cutout and composites it directly on a background image."

    def clean_and_composite(
        self,
        transparent_png,
        background_image,
        clean_strength,
        edge_width,
        sample_radius,
        alpha_threshold,
        solid_threshold,
        protect_texture,
        alpha_contract,
        alpha_from_load_image=None,
    ):
        foreground = transparent_png[..., :3].to(dtype=torch.float32).clamp(0.0, 1.0)
        background = background_image[..., :3].to(dtype=torch.float32).clamp(0.0, 1.0)
        batch_size = max(foreground.shape[0], background.shape[0])
        foreground = _match_batch(foreground, batch_size)
        background = _match_batch(background, batch_size)

        height, width = foreground.shape[1], foreground.shape[2]
        background = _resize_image(background, height, width)

        if alpha_from_load_image is not None:
            alpha = _mask_to_alpha(
                alpha_from_load_image,
                "Load Image MASK (透明=白)",
                height,
                width,
                batch_size,
            )
        else:
            alpha = _infer_alpha_from_corners(foreground)

        cleaned, alpha = _clean_rgb_with_alpha(
            foreground,
            alpha,
            clean_strength,
            edge_width,
            sample_radius,
            alpha_threshold,
            solid_threshold,
            protect_texture,
            alpha_contract,
        )
        composited = cleaned * alpha.unsqueeze(-1) + background * (1.0 - alpha.unsqueeze(-1))
        return (composited.clamp(0.0, 1.0),)


class CleanPNGEdgeCompositeBackground:
    @classmethod
    def INPUT_TYPES(cls):
        files = _input_images()
        return {
            "required": {
                "transparent_png": (files, {"image_upload": True}),
                "background_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "image/cleanup"
    DESCRIPTION = "Loads a transparent PNG file, automatically cleans black/blue/green edge contamination, and composites it on the background image."

    def run(self, transparent_png, background_image):
        foreground, alpha = _load_rgba_from_input(transparent_png)
        background = background_image[..., :3].to(dtype=torch.float32).clamp(0.0, 1.0)

        batch_size = max(foreground.shape[0], background.shape[0])
        foreground = _match_batch(foreground, batch_size)
        alpha = _match_batch(alpha, batch_size)
        background = _match_batch(background, batch_size)
        background = _resize_image(background, foreground.shape[1], foreground.shape[2])

        cleaned, alpha = _clean_rgb_with_alpha(
            foreground,
            alpha,
            0.95,
            14,
            24,
            0.02,
            0.92,
            0.15,
            0.0,
        )
        composited = cleaned * alpha.unsqueeze(-1) + background * (1.0 - alpha.unsqueeze(-1))
        return (composited.clamp(0.0, 1.0),)


class LoadTransparentPNG:
    @classmethod
    def INPUT_TYPES(cls):
        files = _input_images()
        return {
            "required": {
                "png": (files, {"image_upload": True}),
            },
        }

    RETURN_TYPES = ("TRANSPARENT_PNG_RGBA", "IMAGE")
    RETURN_NAMES = ("transparent_png", "preview")
    FUNCTION = "load"
    CATEGORY = "image/cleanup"
    DESCRIPTION = "Loads a PNG as RGBA and keeps the real alpha channel inside a custom data pipe."

    def load(self, png):
        rgb, alpha = _load_rgba_from_input(png)
        data = TransparentPNGData(rgb, alpha, png)
        preview = rgb * alpha.unsqueeze(-1)
        return (data, preview.clamp(0.0, 1.0))


class CleanLoadedPNGCompositeBackground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transparent_png": ("TRANSPARENT_PNG_RGBA",),
                "background_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "image/cleanup"
    DESCRIPTION = "Cleans loaded transparent PNG edges and composites on the background image."

    def run(self, transparent_png, background_image):
        foreground = transparent_png.rgb
        alpha = transparent_png.alpha
        background = background_image[..., :3].to(dtype=torch.float32).clamp(0.0, 1.0)

        batch_size = max(foreground.shape[0], background.shape[0])
        foreground = _match_batch(foreground, batch_size)
        alpha = _match_batch(alpha, batch_size)
        background = _match_batch(background, batch_size)
        background = _resize_image(background, foreground.shape[1], foreground.shape[2])

        cleaned, alpha = _clean_rgb_with_alpha(
            foreground,
            alpha,
            0.95,
            14,
            24,
            0.02,
            0.92,
            0.15,
            0.0,
        )
        composited = cleaned * alpha.unsqueeze(-1) + background * (1.0 - alpha.unsqueeze(-1))
        return (composited.clamp(0.0, 1.0),)


class AutoFringeCleaner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_mode": (
                    ["auto", "Load Image MASK (透明=白)", "Alpha MASK (物体=白)"],
                    {"default": "auto"},
                ),
                "clean_strength": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "edge_width": (
                    "INT",
                    {"default": 14, "min": 1, "max": 64, "step": 1},
                ),
                "sample_radius": (
                    "INT",
                    {"default": 24, "min": 2, "max": 128, "step": 1},
                ),
                "alpha_threshold": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005},
                ),
                "solid_threshold": (
                    "FLOAT",
                    {"default": 0.92, "min": 0.2, "max": 1.0, "step": 0.005},
                ),
                "protect_texture": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "alpha_contract": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.1},
                ),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("cleaned_image", "alpha_mask", "transparent_mask")
    FUNCTION = "clean"
    CATEGORY = "image/cleanup"
    DESCRIPTION = "Automatically removes black, blue, or other color fringes from transparent cutouts by sampling clean nearby interior color."

    def clean(
        self,
        image,
        mask_mode,
        clean_strength,
        edge_width,
        sample_radius,
        alpha_threshold,
        solid_threshold,
        protect_texture,
        alpha_contract,
        mask=None,
    ):
        image = image[..., :3].to(dtype=torch.float32).clamp(0.0, 1.0)
        batch_size, height, width, _ = image.shape
        alpha = _mask_to_alpha(mask, mask_mode, height, width, batch_size)
        if alpha is None:
            alpha = _infer_alpha_from_corners(image)
        cleaned, alpha = _clean_rgb_with_alpha(
            image,
            alpha,
            clean_strength,
            edge_width,
            sample_radius,
            alpha_threshold,
            solid_threshold,
            protect_texture,
            alpha_contract,
        )

        alpha_mask = alpha.clamp(0.0, 1.0)
        transparent_mask = (1.0 - alpha_mask).clamp(0.0, 1.0)
        return (cleaned.clamp(0.0, 1.0), alpha_mask, transparent_mask)


class SaveTransparentPNG:
    def __init__(self):
        import folder_paths

        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "alpha_mask": ("MASK",),
                "filename_prefix": ("STRING", {"default": "ComfyUI_clean_alpha"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "image/cleanup"
    DESCRIPTION = "Saves RGB image plus alpha mask as a transparent PNG."

    def save(self, images, alpha_mask, filename_prefix="ComfyUI_clean_alpha", prompt=None, extra_pnginfo=None):
        results = _save_rgba(
            images,
            alpha_mask,
            filename_prefix + self.prefix_append,
            self.output_dir,
            self.type,
            prompt,
            extra_pnginfo,
        )
        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "LoadTransparentPNGWithAlpha": LoadTransparentPNG,
    "CleanLoadedPNGCompositeBackground": CleanLoadedPNGCompositeBackground,
    "CleanPNGEdgeCompositeBackground": CleanPNGEdgeCompositeBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTransparentPNGWithAlpha": "Load Transparent PNG (Keep Alpha)",
    "CleanLoadedPNGCompositeBackground": "Clean Loaded PNG + Composite Background",
    "CleanPNGEdgeCompositeBackground": "Clean PNG Edge + Composite Background",
}
