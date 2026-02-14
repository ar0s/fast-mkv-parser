# fast-mkv-parser

Sparse MKV track extraction library and CLI. Reads only block headers and desired track payloads from Matroska containers, skipping unwanted data (e.g. video) via `lseek`. Designed for large files on network storage where reading every byte is prohibitively slow.

## The Problem

MKV (Matroska) containers interleave all tracks within Clusters — video, audio, and subtitle blocks are stored sequentially in the same data stream. Every existing tool (ffmpeg, mkvextract, all known Python/Rust/Go libraries) reads **every block payload** before checking the track number and discarding unwanted data.

For an 80 GB UHD Blu-ray rip where you only need the audio track (~4.5 GB) and subtitles (~40 MB), conventional tools transfer the entire 80 GB from storage — even though 95% of it is video data you don't need.

## How It Works

`fast-mkv-parser` exploits the EBML block header structure: each SimpleBlock begins with a track number (1–4 bytes) followed by the payload. By reading just the header, the parser determines whether the block belongs to a wanted track. If not, it calls `lseek()` to skip the payload — **no I/O occurs for skipped data**, even on NFS.

Additionally, `posix_fadvise(FADV_RANDOM)` is used to disable kernel read-ahead, preventing the OS from speculatively fetching video data that will never be read.

```
Source MKV Cluster layout:

[Cluster Header] [Video 2MB] [Audio 2KB] [Audio 2KB] [Video 1.5MB] [Audio 2KB] [Sub 500B] ...
                  ^^^^^^^^^^                          ^^^^^^^^^^^^^
                  lseek (skip)                        lseek (skip)
```

## Performance

Benchmarks on an 80 GB UHD MKV over NFS on a 1 Gbps LAN:

| Method | Time | Data Transferred | Notes |
|--------|------|-----------------|-------|
| **ffmpeg** (full demux) | ~7 min | ~80 GB (entire file) | Reads all blocks, discards video in userspace |
| **fast-mkv-parser** v0.1 (no fadvise) | 3m 36s | ~27 GB (~34% of file) | Sparse seeking, but NFS read-ahead wastes bandwidth |
| **fast-mkv-parser** v0.1 (with fadvise) | **57s** | ~6 GB (~8% of file) | FADV_RANDOM disables read-ahead; only wanted data is transferred |
| Theoretical minimum | ~48s | ~4.5 GB (audio only) | Limited by 1 Gbps wire speed |

**Subtitle-only extraction** (40 MB PGS track from the same 80 GB file): **16 seconds**.

### Summary

- **~7x faster** than ffmpeg for audio extraction from large MKV files over NFS
- **~13x less I/O** than conventional tools
- Approaches wire-speed theoretical minimum on 1 Gbps networks

On local storage (SSD/NVMe), the improvement is smaller since local I/O is fast regardless, but sparse reading still reduces unnecessary wear and memory pressure.

## Installation

```bash
pip install -e /path/to/fast-mkv-parser
```

No external dependencies — pure Python stdlib (`struct`, `os`, `io`, `argparse`).

Requires Python 3.10+.

## CLI Usage

### List tracks

```bash
fast-mkv-parser info media.mkv
```

```
File: media.mkv
Duration: 2:55:28 (10528.1s)
TimecodeScale: 1000000 ns

#    Type       Codec                     Lang   Name
----------------------------------------------------------------------
1    video      V_MPEGH/ISO/HEVC          und
2    audio      A_TRUEHD                  eng    Surround 5.1
3    audio      A_AC3                     eng    Surround 5.1
6    subtitle   S_HDMV/PGS                eng
```

### Extract tracks by type

```bash
# Audio + subtitles to a minimal MKV
fast-mkv-parser extract media.mkv -t audio,subtitle -o stripped.mkv --progress

# Audio only to MKA container
fast-mkv-parser extract media.mkv -t audio -o audio.mka --progress

# Subtitles only
fast-mkv-parser extract media.mkv -t subtitle -o subs.mkv
```

### Extract specific track numbers

```bash
fast-mkv-parser extract media.mkv --track-numbers 2,6 -o selected.mkv
```

### Extract PGS subtitles as Blu-ray SUP format

```bash
fast-mkv-parser extract media.mkv -t subtitle -f sup -o subs.sup
```

The SUP output is compatible with any tool that reads Blu-ray PGS subtitle files (e.g. BDSup2Sub, Tesseract-based OCR pipelines).

### Default behavior

If no `-t` or `--track-numbers` is specified, all non-video tracks are extracted.

## Library API

```python
from fast_mkv_parser import MkvParser

parser = MkvParser("/path/to/media.mkv")

# Inspect tracks
for track in parser.tracks:
    print(f"Track {track.number}: {track.type} ({track.codec_id})")

# Extract audio + subtitles to minimal MKV
parser.extract(track_types=["audio", "subtitle"], output="stripped.mkv")

# Extract with progress reporting
def on_progress(bytes_done, total_bytes):
    print(f"{bytes_done / total_bytes * 100:.1f}%")

parser.extract(track_types=["audio"], output="audio.mka", progress_callback=on_progress)

# Convenience methods
parser.extract_audio("audio.mka")
parser.extract_subtitles("subs.sup", format="sup")
```

### Track filtering options

| Parameter | Example | Description |
|-----------|---------|-------------|
| `track_types` | `["audio", "subtitle"]` | Extract all tracks of these types |
| `track_numbers` | `[2, 6]` | Extract specific track numbers |
| Neither | — | Extracts all non-video tracks (default) |

### Output formats

| Format | Extension | Description |
|--------|-----------|-------------|
| `"mkv"` | `.mkv`, `.mka`, `.mks` | Valid Matroska container with selected tracks only |
| `"sup"` | `.sup` | Blu-ray PGS subtitle format (reconstructed from MKV timestamps) |

## How the Output Compares to ffmpeg

The extracted MKA/MKV output is a valid Matroska container that ffmpeg, mkvtoolnix, and any standards-compliant player can read. The audio codec data is copied byte-for-byte — no transcoding or quality loss.

```bash
# Verify output with ffprobe
ffprobe stripped.mkv

# Decode extracted audio to WAV (if needed for processing)
ffmpeg -i audio.mka -acodec pcm_s16le -ac 1 -ar 16000 audio.wav
```

## Limitations

- **Read-only**: Extracts tracks from existing MKV files. Does not mux or remux.
- **Sequential Cluster walking**: Processes Clusters in file order. MKV files without Cues (seek index) still require walking through the entire file's block headers, though video payloads are skipped.
- **No video codec awareness**: Does not parse codec-specific data (NAL units, frame types). Treats all block payloads as opaque byte sequences.
- **Python performance**: The EBML header parsing is pure Python. For files with millions of blocks, the CPU overhead is ~3-4 seconds. The I/O savings far outweigh this cost for network storage.
