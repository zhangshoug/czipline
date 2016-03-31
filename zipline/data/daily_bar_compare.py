#
# Copyright 2016 Quantopian, Inc.
#


class DailyBarComparison(object):

    def __init__(self, reader_a, reader_b,
                 assets=None,
                 start_date=None,
                 end_date=None):
        self.reader_a = reader_a
        self.reader_b = reader_b
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date

    def compare(self):
        pass
