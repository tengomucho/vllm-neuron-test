#!/usr/bin/env python3
"""Repoint an ELF's GLIBC version requirement to an older version, in place.

The public torch-neuronx 2.33 wheels are built on glibc 2.43 (Ubuntu 26.04).
Their `_C.so` only needs GLIBC_2.43 for nine float math functions (acosf,
asinf, ...) that glibc 2.43 re-versioned; glibc 2.39 (Ubuntu 24.04) still
exports them as GLIBC_2.2.5. This rewrites the version-needed entry so the
loader binds them to the old versions. Workaround for testing only.

Usage:
    python scripts/patch_glibc_verneed.py <file.so> [--from GLIBC_2.43] [--to GLIBC_2.2.5]
"""
import argparse
import shutil
import struct

SHT_GNU_VERNEED = 0x6FFFFFFE


def elf_hash(name: bytes) -> int:
    h = 0
    for c in name:
        h = (h << 4) + c
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g
    return h & 0xFFFFFFFF


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--from", dest="src", default="GLIBC_2.43")
    parser.add_argument("--to", dest="dst", default="GLIBC_2.2.5")
    args = parser.parse_args()
    src, dst = args.src.encode(), args.dst.encode()

    data = bytearray(open(args.path, "rb").read())
    assert data[:4] == b"\x7fELF" and data[4] == 2 and data[5] == 1, "ELF64 little-endian only"
    e_shoff, = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)
    sections = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_type, = struct.unpack_from("<I", data, off + 4)
        sh_offset, sh_size = struct.unpack_from("<QQ", data, off + 0x18)
        sh_link, = struct.unpack_from("<I", data, off + 0x28)
        sections.append((sh_type, sh_offset, sh_size, sh_link))

    verneed = [s for s in sections if s[0] == SHT_GNU_VERNEED]
    assert len(verneed) == 1, "expected one .gnu.version_r section"
    _, vn_off, _, link = verneed[0]
    strtab_off, strtab_size = sections[link][1], sections[link][2]
    strtab = bytes(data[strtab_off : strtab_off + strtab_size])

    def name_at(idx: int) -> bytes:
        return strtab[idx : strtab.index(b"\0", idx)]

    # Point to an existing `dst` string: the file already references it.
    dst_idx = strtab.find(b"\0" + dst + b"\0")
    assert dst_idx >= 0, f"{args.dst} not in .dynstr"
    dst_idx += 1

    patched = 0
    entry = vn_off
    while True:
        _, vn_cnt, vn_file, vn_aux, vn_next = struct.unpack_from("<HHIII", data, entry)
        aux = entry + vn_aux
        for _ in range(vn_cnt):
            _, vna_flags, vna_other, vna_name, vna_next = struct.unpack_from("<IHHII", data, aux)
            if name_at(vna_name) == src:
                struct.pack_into("<IHHI", data, aux, elf_hash(dst), vna_flags, vna_other, dst_idx)
                print(f"{name_at(vn_file).decode()}: {args.src} -> {args.dst} (index {vna_other})")
                patched += 1
            if not vna_next:
                break
            aux += vna_next
        if not vn_next:
            break
        entry += vn_next

    if not patched:
        print(f"no {args.src} requirement found; nothing to do")
        return
    shutil.copy2(args.path, args.path + ".orig")
    open(args.path, "wb").write(data)
    print(f"patched {patched} entr{'y' if patched == 1 else 'ies'}; original kept as {args.path}.orig")


if __name__ == "__main__":
    main()
