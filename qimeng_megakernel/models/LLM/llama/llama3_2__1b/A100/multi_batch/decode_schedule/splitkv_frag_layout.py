"""SplitKV score tile fragment (tid, fi) → (row, col) in smem_score[M16, N64].

Single source of truth for PyVM debug dumps and layout verification; must match
`splitkv_score_smem_frag_coord` in decode_kernel/utils.cuh.
"""


def splitkv_score_frag_coord_py(tid: int, fi: int) -> tuple[int, int]:
    lane = tid & 31
    warp = tid >> 5
    row = (lane >> 2) | (((fi & 3) >> 1) << 3)
    col = (warp << 4) | ((lane & 3) << 1) | ((fi >> 2) << 3) | (fi & 1)
    return row, col
