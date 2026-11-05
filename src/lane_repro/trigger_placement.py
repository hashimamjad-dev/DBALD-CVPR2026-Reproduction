"""One-off sanity check for ultraedit_trigger.UltraEditTrigger's mask compositing.

Run this once on your GPU machine, on ONE real image, before trusting any batch poisoning run.
It checks the thing the mask-compositing fix is supposed to guarantee: every pixel outside the
40x40 trigger footprint is bit-for-bit identical between the clean and poisoned image. If this
fails, the compositing fix did not get applied correctly -- stop and fix it before poisoning a
real dataset.

Usage (from the project's .venv, on the GPU-enabled machine):
    python check_trigger_compositing.py --image path/to/one/tusimple/frame.jpg --y0 400 --x0 300

Writes a diff heatmap PNG next to the script so you can eyeball exactly where (if anywhere) pixels
changed outside the trigger box.
"""
import argparse
import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to a real road-scene image (RGB).")
    ap.add_argument("--y0", type=int, required=True, help="Trigger top-left y (pick a spot on the road).")
    ap.add_argument("--x0", type=int, required=True, help="Trigger top-left x.")
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument("--prompt", default=None, help="Defaults to MUD_PROMPT.")
    ap.add_argument("--out", default="trigger_diff.png")
    args = ap.parse_args()

    from ultraedit_trigger import UltraEditTrigger, MUD_PROMPT, feasibility_check

    print("Running feasibility_check() first (loads model, times one tiny edit)...")
    info = feasibility_check()
    print(info)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    trigger = UltraEditTrigger()
    prompt = args.prompt or MUD_PROMPT
    poisoned = trigger(image, args.y0, args.x0, args.size, prompt, seed=0)

    diff = np.abs(image.astype(int) - poisoned.astype(int)).sum(axis=-1)  # (H, W)

    mask = np.zeros(diff.shape, np.uint8)
    mask[args.y0 : args.y0 + args.size, args.x0 : args.x0 + args.size] = 255

    outside_diff = diff[mask == 0]
    inside_diff = diff[mask == 255]

    print(f"Max pixel diff OUTSIDE trigger box: {outside_diff.max()} (must be 0, or you have a bug)")
    print(f"Mean pixel diff INSIDE trigger box:  {inside_diff.mean():.2f} (should be > 0, trigger should visibly differ)")
    print(f"Nonzero-diff pixel count outside box: {(outside_diff > 0).sum()} / {outside_diff.size}")

    heat = cv2.applyColorMap((np.clip(diff, 0, 255)).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.rectangle(heat, (args.x0, args.y0), (args.x0 + args.size, args.y0 + args.size), (255, 255, 255), 1)
    cv2.imwrite(args.out, heat)
    print(f"Diff heatmap written to {args.out} (white box = trigger footprint)")

    if outside_diff.max() > 0:
        raise SystemExit(
            "FAIL: pixels outside the trigger footprint changed. "
            "Compositing fix is not working as expected -- do not run a full poisoning batch yet."
        )
    print("PASS: everything outside the trigger footprint is pixel-identical.")


if __name__ == "__main__":
    main()