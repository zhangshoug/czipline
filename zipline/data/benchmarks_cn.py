import pandas as pd
from cswd.sql.base import session_scope
from cswd.sql.models import IndexDaily


def get_cn_benchmark_returns(symbol):
    """
    获取指数基准收益率，默认代码`000001`

    Parameters
    ----------
    symbol : str
        Benchmark symbol for which we're getting the returns.
    """
    with session_scope() as sess:
        query = sess.query(
            IndexDaily.date, IndexDaily.close
        ).filter(IndexDaily.code == symbol)
        df = pd.DataFrame.from_records(query.all())
        df.columns = ['date', 'close']
        df.index = pd.DatetimeIndex(df['date'])
        df.drop('date', axis=1, inplace=True)
        return df.sort_index().tz_localize('UTC').pct_change(1).iloc[1:]
