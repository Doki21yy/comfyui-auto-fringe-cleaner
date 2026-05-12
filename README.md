# ComfyUI Auto Fringe Cleaner

A small ComfyUI custom node package for cleaning colored fringes from transparent PNG cutouts before compositing them onto a background.

It is designed for PNGs that have already been cut out but still show black, blue, green, or mixed-color edge contamination.

## Nodes

- `Load Transparent PNG (Keep Alpha)`
- `Image Cutout To Transparent PNG Pipe`
- `Clean Loaded PNG + Composite Background`
- `Clean PNG Edge + Composite Background`

All nodes appear under `image/cleanup`.

## Recommended Workflow

1. Use `Load Transparent PNG (Keep Alpha)` to load the transparent cutout.
2. Use regular `Load Image` to load the background.
3. Connect `transparent_png` to `Clean Loaded PNG + Composite Background`.
4. Connect background `图像` to `background_image`.
5. Save the output image with regular `Save Image`.

The special loader outputs a custom `TRANSPARENT_PNG_RGBA` pipe, so RGB and real alpha stay together. This avoids ComfyUI's normal `IMAGE` alpha loss.

`Clean PNG Edge + Composite Background` is still available as a one-node shortcut: it picks/uploads the transparent PNG inside the compositing node.

## Upstream Cutout Workflow

If another node already outputs a cutout `IMAGE`, use this bridge:

1. Connect the cutout node `image` output to `Image Cutout To Transparent PNG Pipe`.
2. Connect its `transparent_png` output to `Clean Loaded PNG + Composite Background`.
3. Connect background `图像` from regular `Load Image` to `background_image`.
4. Save the final `image` output.

`Image Cutout To Transparent PNG Pipe` keeps alpha if the upstream image has it. If the upstream image is RGB on a solid blue, green, black, or red background, it estimates alpha from the background automatically.

## Installation

Clone this repository into `ComfyUI/custom_nodes`, then restart ComfyUI.

```bash
cd ComfyUI/custom_nodes
git clone <repo-url> comfyui-auto-fringe-cleaner
```

No extra Python dependencies are required beyond ComfyUI's normal `torch`, `PIL`, and `numpy` environment.

## Why A Special PNG Loader?

ComfyUI's regular `Load Image` node separates PNG alpha into a mask. The `IMAGE` output alone does not keep the original alpha channel.

`Load Transparent PNG (Keep Alpha)` reads PNG files as RGBA and passes RGB plus alpha through a custom `TRANSPARENT_PNG_RGBA` pipe, so edge cleanup and compositing can stay accurate without manual mask wiring.
