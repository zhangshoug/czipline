from numpy cimport (
    float64_t,
    ndarray,
    uint8_t,
    uint32_t,
    intp_t
)


cdef class Block:
    # Not available in Python-space:
    cdef public intp_t start, end
    # Available in Python-space:
    cdef uint32_t[:] array

    def __init__(self, array, start, end):
        self.array = array
        self.start = start
        self.end = end

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = slice(idx.start - self.start, idx.stop - self.start)
        else:
            idx = idx - self.start
        return self.array[idx]
