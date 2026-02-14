"""Unit tests for EBML VINT encoding/decoding."""

import io
import pytest

from fast_mkv_parser.ebml import (
    UNKNOWN_SIZE,
    read_vint_raw,
    read_element_id,
    read_element_size,
    read_vint_value,
    encode_element_id,
    encode_element_size,
    encode_vint_value,
    peek_element_id,
)


class TestReadVintRaw:
    """Test raw VINT reading (marker bit kept)."""

    def test_width_1(self):
        # 0x81 = 1_0000001 → width 1, raw value 0x81
        f = io.BytesIO(b"\x81")
        val, width = read_vint_raw(f)
        assert width == 1
        assert val == 0x81

    def test_width_1_max(self):
        # 0xFF = 1_1111111 → width 1, raw value 0xFF
        f = io.BytesIO(b"\xFF")
        val, width = read_vint_raw(f)
        assert width == 1
        assert val == 0xFF

    def test_width_2(self):
        # 0x4000 = 01_000000 00000000 → width 2
        f = io.BytesIO(b"\x40\x00")
        val, width = read_vint_raw(f)
        assert width == 2
        assert val == 0x4000

    def test_width_2_example(self):
        # 0x4286 = real EBML element ID for EBMLVersion
        f = io.BytesIO(b"\x42\x86")
        val, width = read_vint_raw(f)
        assert width == 2
        assert val == 0x4286

    def test_width_4(self):
        # 0x1A45DFA3 = EBML Header element ID
        f = io.BytesIO(b"\x1A\x45\xDF\xA3")
        val, width = read_vint_raw(f)
        assert width == 4
        assert val == 0x1A45DFA3

    def test_eof_raises(self):
        f = io.BytesIO(b"")
        with pytest.raises(EOFError):
            read_vint_raw(f)

    def test_zero_byte_raises(self):
        f = io.BytesIO(b"\x00")
        with pytest.raises(ValueError):
            read_vint_raw(f)

    def test_truncated_raises(self):
        # Width-2 VINT but only 1 byte available.
        f = io.BytesIO(b"\x40")
        with pytest.raises(EOFError):
            read_vint_raw(f)


class TestReadElementId:
    """Element IDs keep the marker bit."""

    def test_ebml_header_id(self):
        f = io.BytesIO(b"\x1A\x45\xDF\xA3")
        eid, width = read_element_id(f)
        assert eid == 0x1A45DFA3
        assert width == 4

    def test_segment_id(self):
        f = io.BytesIO(b"\x18\x53\x80\x67")
        eid, width = read_element_id(f)
        assert eid == 0x18538067
        assert width == 4

    def test_simple_block_id(self):
        f = io.BytesIO(b"\xA3")
        eid, width = read_element_id(f)
        assert eid == 0xA3
        assert width == 1

    def test_cluster_timestamp_id(self):
        f = io.BytesIO(b"\xE7")
        eid, width = read_element_id(f)
        assert eid == 0xE7
        assert width == 1

    def test_track_entry_id(self):
        # 0xAE = width 1
        f = io.BytesIO(b"\xAE")
        eid, width = read_element_id(f)
        assert eid == 0xAE
        assert width == 1


class TestReadElementSize:
    """Element sizes strip the marker bit."""

    def test_small_size(self):
        # 0x85 = 1_0000101 → size = 5
        f = io.BytesIO(b"\x85")
        size, width = read_element_size(f)
        assert size == 5
        assert width == 1

    def test_size_zero(self):
        # 0x80 = 1_0000000 → size = 0
        f = io.BytesIO(b"\x80")
        size, width = read_element_size(f)
        assert size == 0
        assert width == 1

    def test_unknown_size_width_1(self):
        # 0xFF = all data bits set → unknown
        f = io.BytesIO(b"\xFF")
        size, width = read_element_size(f)
        assert size == UNKNOWN_SIZE
        assert width == 1

    def test_unknown_size_width_8(self):
        f = io.BytesIO(b"\x01\xFF\xFF\xFF\xFF\xFF\xFF\xFF")
        size, width = read_element_size(f)
        assert size == UNKNOWN_SIZE
        assert width == 8

    def test_large_size(self):
        # Width 2: 0x40 | value, marker at bit 14
        # Value 1000 = 0x03E8 → 0x4000 | 0x03E8 = 0x43E8
        f = io.BytesIO(b"\x43\xE8")
        size, width = read_element_size(f)
        assert size == 1000
        assert width == 2


class TestReadVintValue:
    """VINT used as data value (e.g. track number in block header)."""

    def test_track_1(self):
        # 0x81 → marker stripped → value 1
        f = io.BytesIO(b"\x81")
        val, width = read_vint_value(f)
        assert val == 1
        assert width == 1

    def test_track_2(self):
        f = io.BytesIO(b"\x82")
        val, width = read_vint_value(f)
        assert val == 2

    def test_track_127(self):
        # Width 1 max value = 0x7F - 1 = 126 (0x7F is reserved for unknown)
        # Track 127 needs width 2: 0x40 | 127 = 0x407F
        f = io.BytesIO(b"\x40\x7F")
        val, width = read_vint_value(f)
        assert val == 127
        assert width == 2


class TestEncodeElementId:
    def test_simple_block(self):
        assert encode_element_id(0xA3) == b"\xA3"

    def test_ebml_header(self):
        assert encode_element_id(0x1A45DFA3) == b"\x1A\x45\xDF\xA3"

    def test_cluster(self):
        assert encode_element_id(0x1F43B675) == b"\x1F\x43\xB6\x75"


class TestEncodeElementSize:
    def test_small(self):
        assert encode_element_size(5) == b"\x85"

    def test_zero(self):
        assert encode_element_size(0) == b"\x80"

    def test_large(self):
        assert encode_element_size(1000) == b"\x43\xE8"

    def test_unknown(self):
        result = encode_element_size(UNKNOWN_SIZE)
        assert result == b"\xFF"

    def test_unknown_width_8(self):
        result = encode_element_size(UNKNOWN_SIZE, width=8)
        assert result == b"\x01\xFF\xFF\xFF\xFF\xFF\xFF\xFF"

    def test_roundtrip(self):
        """Encoding then decoding should yield the original value."""
        for val in [0, 1, 5, 126, 127, 1000, 16383, 16384, 100000, 1_000_000]:
            encoded = encode_element_size(val)
            f = io.BytesIO(encoded)
            decoded, _ = read_element_size(f)
            assert decoded == val, f"roundtrip failed for {val}"


class TestEncodeVintValue:
    def test_track_1(self):
        assert encode_vint_value(1) == b"\x81"

    def test_track_2(self):
        assert encode_vint_value(2) == b"\x82"

    def test_roundtrip(self):
        for val in [1, 2, 5, 50, 126, 127, 200, 1000]:
            encoded = encode_vint_value(val)
            f = io.BytesIO(encoded)
            decoded, _ = read_vint_value(f)
            assert decoded == val, f"roundtrip failed for {val}"


class TestPeekElementId:
    def test_peek_preserves_position(self):
        f = io.BytesIO(b"\xA3\x85xxxxx")
        pos_before = f.tell()
        eid, width = peek_element_id(f)
        assert eid == 0xA3
        assert f.tell() == pos_before
