from numpy cimport (
    float64_t,
    ndarray,
    uint8_t,
    uint32_t,
    intp_t,
)
from numpy import nan
from pandas import NaT
cimport cython


cdef class Block:
    # Not available in Python-space:
    cdef public intp_t start, end
    cdef uint32_t[:] array

    def __init__(self, array, start, end):
        self.array = array
        self.start = start
        self.end = end

    @cython.boundscheck(False)
    def get_idx(self, intp_t idx):
        return self.array[idx - self.start]

    def get_slice(self, slice_):
        cdef intp_t start, end
        start = slice_.start - self.start
        end = slice_.stop - self.start
        return self.array[start:end]

cdef class ColWrapper:

    cdef object carray
    cdef object _block

    def __init__(self, carray):
        self.carray = carray
        self._block = None

    def get_slice(self, slice_):
        if self._block is not None:
            if (self._block.start < slice_.start) and \
               (self._block.end > slice_.stop):
                return self._block.get_slice(slice_)

        start = max(slice_.start - 3900, 0)
        end = min(slice_.stop + 3900, len(self.carray))

        self._block = Block(self.carray[start:end], start, end)
        return self._block.get_slice(slice_)

    def get_idx(self, intp_t idx):
        if self._block is not None:
            if self._block.start < idx < self._block.end:
                return self._block.get_idx(idx)

        start = max(idx - 3900, 0)
        end = min(idx + 3900, len(self.carray))

        self._block = Block(self.carray[start:end], start, end)
        return self._block.get_idx(idx)


@cython.boundscheck(False)
cpdef get_last_value(reader, sid, dt, field):
    cdef intp_t minute_pos, initial_minute_pos, last_idx, last_checked, i
    cdef uint32_t value
    minute_pos = reader._find_position_of_minute(dt)
    col = reader._open_minute_file(field, sid)
    value = col.get_idx(minute_pos)
    if value != 0:
        if field == 'volume':
            return dt, value
        else:
            return dt, value * reader._ohlc_inverse
    else:
        try:
            last_idx, value, last_checked = reader._last_traded_cache[sid]
            if minute_pos - last_checked <= 1:
                reader._last_traded_cache[sid] = (last_idx, value, minute_pos)
                if field == 'volume':
                    return reader._minute_index[last_idx], value
                else:
                    return reader._minute_index[last_idx], \
                        value * reader._ohlc_inverse
        except KeyError:
            pass

        initial_minute_pos = minute_pos

        while True:
            if minute_pos == 1:
                return NaT, nan
            chunksize = 780
            start = max(0, minute_pos - chunksize)
            candidates = col.get_slice(slice(start, minute_pos))
            for i in xrange(len(candidates) - 1, 0, -1):
                minute_pos -= 1
                value = candidates[i]
                if value != 0:
                    reader._last_traded_cache[sid] = (minute_pos,
                                                      value,
                                                      initial_minute_pos)
                    if field == 'volume':
                        return reader._minute_index[minute_pos], value
                    else:
                        return reader._minute_index[minute_pos], \
                            value * reader._ohlc_inverse
