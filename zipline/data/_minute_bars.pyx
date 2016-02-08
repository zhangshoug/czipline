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
    # Available in Python-space:
    cdef public uint32_t[:] array

    def __init__(self, array, start, end):
        self.array = array
        self.start = start
        self.end = end

    @cython.boundscheck(False)
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.array[idx.start - self.start:idx.stop - self.start]
        else:
            return self.array[idx - self.start]


@cython.boundscheck(False)
cpdef get_last_value(reader, sid, dt, field):
    cdef intp_t minute_pos, initial_minute_pos, last_idx, last_checked
    cdef uint32_t value
    minute_pos = reader._find_position_of_minute(dt)
    col = reader._open_minute_file(field, sid)
    value = col[minute_pos]
    if value != 0:
        if field == 'volume':
            return dt, value
        else:
            return dt, value * reader._ohlc_inverse
    else:
        try:
            last_idx, last_checked = reader._last_traded_cache[sid]
            if minute_pos - last_checked <= 1:
                reader._last_traded_cache[sid] = (last_idx, minute_pos)
                value = col[last_idx]
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
            candidates = col[start:minute_pos]
            for i in xrange(len(candidates) - 1, 0, -1):
                minute_pos -= 1
                value = candidates[i]
                if value != 0:
                    reader._last_traded_cache[sid] = (minute_pos,
                                                    initial_minute_pos)
                    if field == 'volume':
                        return reader._minute_index[minute_pos], value
                    else:
                        return reader._minute_index[minute_pos], \
                            value * reader._ohlc_inverse
