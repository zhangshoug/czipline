import os
from unittest import TestCase

import pandas as pd
from six import iteritems
from testfixtures import TempDirectory

from zipline.data.quandl import (
    build_database,
    format_wiki_url,
    quandl_data_path,
)
import zipline.data.paths as pth
from zipline.utils.test_utils import (
    patch_os_environment,
    read_compressed,
    test_resource_path,
)
from zipline.utils.tradingcalendar import trading_day


class QuandlLoaderTestCase(TestCase):

    start_date = pd.Timestamp('2014')
    end_date = pd.Timestamp('2015')
    symbols = ['AAPL', 'MSFT', 'BRK_A', 'ZEN']
    api_key = 'notactuallyanapikey'

    frames = {}
    for symbol in symbols:
        url = format_wiki_url(api_key, symbol, start_date, end_date)
        frames[url] = read_compressed(
            test_resource_path('quandl_samples', symbol + '.csv.gz'),
        )

    @classmethod
    def setUpClass(cls):
        cls.response_data = {}
        cls.frames = {}
        for symbol in cls.symbols:
            
            cls.frames[symbol] = pd.read_csv(cls.response_data[url])

        cls.calendar = pd.date_range(
            cls.start_date,
            cls.end_date,
            freq=trading_day,
        )

    def test_quandl_path_no_root_set(self):
        zipline_root = pth.zipline_root(environ=None)
        path = quandl_data_path(['foobar'])
        self.assertEqual(
            path, os.path.join(zipline_root, 'data', 'quandl', 'foobar')
        )

    def test_quandl_path_override_environ_with_dict(self):
        path = quandl_data_path(
            ['foobar'], environ={'ZIPLINE_ROOT': '/tmp/zproot'}
        )
        self.assertEqual(path, '/tmp/zproot/data/quandl/foobar')

    def test_quandl_path_with_os_environ(self):
        with patch_os_environment(ZIPLINE_ROOT='/tmp/zproot'):
            path = quandl_data_path(
                ['foobar'], environ={'ZIPLINE_ROOT': '/tmp/zproot'}
            )
            self.assertEqual(path, '/tmp/zproot/data/quandl/foobar')

    def test_build_database(self):
        with TempDirectory() as tempdir:
            build_database(
                self.api_key,
                pd.Series(self.symbols),
                self.calendar,
                show_progress=False,
                environ={"ZIPLINE_ROOT": tempdir.path},
            )
            import nose.tools; nose.tools.set_trace()
