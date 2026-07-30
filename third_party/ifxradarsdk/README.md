Optional location for a prebuilt Infineon `ifxradarsdk` wheel.

Expected usage on Jetson:

```bash
chmod +x scripts/install_ifxradarsdk.sh
./scripts/install_ifxradarsdk.sh
```

Notes:
- Use the Linux/aarch64 wheel for Jetson Orin.
- The wheel must provide the packaged shared libraries used by `ifxradarsdk`.
- If no wheel is present here, the installer falls back to building one from
  `third_party/radar_sdk`.
- After install, the script runs `DeviceFmcw.get_list()` as a smoke test.
