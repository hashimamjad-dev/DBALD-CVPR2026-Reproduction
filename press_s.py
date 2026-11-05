"""Repeatedly press S in the focused window. Press Esc to stop (Windows)."""

import argparse
import ctypes
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=0.1,
                        help="Seconds between presses (default: 0.1)")
    parser.add_argument("--delay", type=float, default=5.0,
                        help="Startup delay in seconds (default: 5)")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("This script requires Windows.")
    if not (0 < args.interval < float("inf")) or not (0 <= args.delay < float("inf")):
        parser.error("Interval must be positive and delay nonnegative; both must be finite.")

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = ctypes.c_short
    user32.keybd_event.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte,
                                  ctypes.c_ulong, ctypes.c_size_t]
    user32.keybd_event.restype = None

    def wait(seconds):
        deadline = time.monotonic() + seconds
        while True:
            if user32.GetAsyncKeyState(0x1B) & 0x8000:
                raise KeyboardInterrupt
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.02, remaining))

    print(f"Starting in {args.delay:g} seconds. Focus the target window.")
    print("Press Esc anywhere, or Ctrl+C in this terminal, to stop.")
    try:
        wait(args.delay)
        while True:
            try:
                user32.keybd_event(0x53, 0, 0, 0)  # S down
            finally:
                user32.keybd_event(0x53, 0, 0x0002, 0)  # S up
            wait(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
