#!/usr/bin/env python3
"""Print identification and capability info for connected Infineon FMCW radar boards."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ifxradarsdk import get_version, get_version_full
from ifxradarsdk.common.common_types import RadarSensor
from ifxradarsdk.common.exceptions import ErrorDeviceBusy, ErrorNoDevice
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import create_dict_from_sequence


def _sensor_name(sensor: RadarSensor) -> str:
    try:
        return RadarSensor(sensor).name
    except ValueError:
        return str(int(sensor))


def _hz_to_ghz(hz: float) -> float:
    return hz / 1e9


def list_devices(sensor_filter: RadarSensor | None) -> list[str]:
    return DeviceFmcw.get_list(sensor_filter)


def collect_device_info(device: DeviceFmcw, include_sequence: bool) -> dict[str, Any]:
    sensor_type = device.get_sensor_type()
    sensor_info = device.get_sensor_information()
    firmware = device.get_firmware_information()

    info: dict[str, Any] = {
        "sdk_version": get_version(),
        "sdk_version_full": get_version_full(),
        "board_uuid": device.get_board_uuid(),
        "sensor_type": _sensor_name(sensor_type),
        "sensor_type_id": int(sensor_type),
        "firmware": firmware,
        "sensor": {
            **sensor_info,
            "min_rf_frequency_GHz": _hz_to_ghz(sensor_info["min_rf_frequency_Hz"]),
            "max_rf_frequency_GHz": _hz_to_ghz(sensor_info["max_rf_frequency_Hz"]),
        },
    }

    try:
        info["temperature_C"] = device.get_temperature()
    except Exception as exc:
        info["temperature_C"] = None
        info["temperature_note"] = str(exc)

    if include_sequence:
        sequence = device.get_acquisition_sequence()
        info["default_acquisition_sequence"] = create_dict_from_sequence(sequence)
        try:
            info["default_sequence_duration_s"] = device.get_sequence_duration(sequence)
        except RuntimeError as exc:
            info["default_sequence_duration_s"] = None
            info["default_sequence_duration_note"] = str(exc)

    return info


def print_human(info: dict[str, Any]) -> None:
    print(f"Radar SDK:        {info['sdk_version']} ({info['sdk_version_full']})")
    print(f"Board UUID:       {info['board_uuid']}")
    print(f"Sensor:           {info['sensor_type']} (id={info['sensor_type_id']})")

    fw = info["firmware"]
    print(f"Firmware:         {fw.get('description', '?')}")
    print(
        f"                  v{fw.get('version_major', '?')}."
        f"{fw.get('version_minor', '?')}."
        f"{fw.get('version_build', '?')} "
        f"({fw.get('extended_version', '')})"
    )

    s = info["sensor"]
    print(f"RF band:          {s['min_rf_frequency_GHz']:.3f} - {s['max_rf_frequency_GHz']:.3f} GHz")
    print(f"Antennas:         {s['num_tx_antennas']} TX, {s['num_rx_antennas']} RX")
    print(f"Max TX power:     {s['max_tx_power']}")
    print(f"Max samples/chirp:{s['max_num_samples_per_chirp']}")
    print(
        f"ADC:              {s['adc_resolution_bits']} bit, "
        f"{s['min_adc_sampling_rate']/1e6:.3f}-{s['max_adc_sampling_rate']/1e6:.3f} MHz"
    )
    print(f"Device ID:        0x{s['device_id']:X}" if s["device_id"] else "Device ID:        0")
    print(f"IF gain options:  {s['if_gain_list']} dB")
    print(f"HP cutoff options:{s['hp_cutoff_list']} Hz")
    print(f"LP cutoff options:{s['lp_cutoff_list']} Hz")

    if info.get("temperature_C") is not None:
        print(f"Temperature:      {info['temperature_C']:.1f} C")
    elif info.get("temperature_note"):
        print(f"Temperature:      n/a ({info['temperature_note']})")

    if "default_acquisition_sequence" in info:
        dur = info.get("default_sequence_duration_s")
        if dur is not None:
            print(f"Default frame:    {dur * 1e3:.2f} ms")
        print("Default sequence: (use --json for full structure)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uuid",
        help="Connect to board with this UUID (default: first device found)",
    )
    parser.add_argument("--port", help="Connect via COM port, e.g. COM7")
    parser.add_argument(
        "--sensor",
        choices=[s.name for s in RadarSensor if s.name not in ("Unknown_sensor", "Unknown_Avian")],
        help="Filter device list by sensor type",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list connected UUIDs, do not open a device",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )
    parser.add_argument(
        "--no-sequence",
        action="store_true",
        help="Skip reading default acquisition sequence",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sensor_filter = RadarSensor[args.sensor] if args.sensor else None

    try:
        uuids = list_devices(sensor_filter)
    except Exception as exc:
        print(f"Failed to enumerate devices: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"connected_uuids": uuids}, indent=2))
    else:
        print("Connected boards:")
        if uuids:
            for u in uuids:
                print(f"  {u}")
        else:
            print("  (none)")

    if args.list_only:
        return 0 if uuids else 1

    if not uuids:
        print(
            "\nNo device found. Connect the board, close Radar Fusion GUI, check USB.",
            file=sys.stderr,
        )
        return 1

    connect_kw: dict[str, str] = {}
    if args.uuid:
        connect_kw["uuid"] = args.uuid
    elif args.port:
        connect_kw["port"] = args.port

    try:
        with DeviceFmcw(**connect_kw) as device:
            info = collect_device_info(device, include_sequence=not args.no_sequence)
    except ErrorNoDevice:
        print("Error: no compatible device.", file=sys.stderr)
        return 1
    except ErrorDeviceBusy:
        print(
            "Error: device busy (close Radar Fusion GUI / other apps using the board).",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(info, indent=2, default=str))
    else:
        print()
        print_human(info)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
