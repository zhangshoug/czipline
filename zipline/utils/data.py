#
# Copyright 2013 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
from copy import deepcopy


def _ensure_index(x):
    if not isinstance(x, pd.Index):
        x = pd.Index(sorted(x))

    return x


class RollingPanel(object):
    """
    Preallocation strategies for rolling window over expanding data set

    Restrictions: major_axis can only be a DatetimeIndex for now
    """

    def __init__(self, window, items, sids, cap_multiple=2, dtype=np.float64):

        self.pos = 0
        self.window = window

        self.items = _ensure_index(items)
        self.minor_axis = _ensure_index(sids)

        self.cap_multiple = cap_multiple
        self.cap = cap_multiple * window

        self.dtype = dtype
        self.date_buf = np.empty(self.cap, dtype='M8[ns]')

        self.buffer = self._create_buffer()

    def _oldest_frame_idx(self):
        return max(self.pos - self.window, 0)

    def oldest_frame(self):
        """
        Get the oldest frame in the panel.
        """
        return self.buffer.iloc[:, self._oldest_frame_idx(), :]

    def set_sids(self, sids):
        self.minor_axis = _ensure_index(sids)
        self.buffer = self.buffer.reindex(minor_axis=self.minor_axis)

    def _create_buffer(self):
        panel = pd.Panel(
            items=self.items,
            minor_axis=self.minor_axis,
            major_axis=range(self.cap),
            dtype=self.dtype,
        )
        return panel

    def add_frame(self, tick, frame):
        """
        """
        if self.pos == self.cap:
            self._roll_data()

        self.buffer.loc[:, self.pos, :] = frame.T.astype(self.dtype)
        self.date_buf[self.pos] = tick

        self.pos += 1

    def get_current(self):
        """
        Get a Panel that is the current data in view. It is not safe to persist
        these objects because internal data might change
        """
        where = slice(self._oldest_frame_idx(), self.pos)
        major_axis = pd.DatetimeIndex(deepcopy(self.date_buf[where]), tz='utc')
        return pd.Panel(self.buffer.values[:, where, :], self.items,
                        major_axis, self.minor_axis, dtype=self.dtype)

    def set_current(self, panel):
        """
        Set the values stored in our current in-view data to be values of the
        passed panel.  The passed panel must have the same indices as the panel
        that would be returned by self.get_current.
        """
        where = slice(self._oldest_frame_idx(), self.pos)
        self.buffer.values[:, where, :] = panel.values

    def current_dates(self):
        where = slice(self._oldest_frame_idx(), self.pos)
        return pd.DatetimeIndex(deepcopy(self.date_buf[where]), tz='utc')

    def _roll_data(self):
        """
        Roll window worth of data up to position zero.
        Save the effort of having to expensively roll at each iteration
        """

        self.buffer.values[:, :self.window, :] = \
            self.buffer.values[:, -self.window:, :]
        self.date_buf[:self.window] = self.date_buf[-self.window:]
        self.pos = self.window
