from zipline.data.bundles.sqldata import gen_asset_metadata
from zipline.data.bundles.cnquandl import _to_sid

metadata = gen_asset_metadata(False)
symbol_map = metadata.symbol[:30]
for _, symbol in symbol_map.iteritems():
    print(_to_sid(symbol), symbol)

