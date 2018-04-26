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
            IndexDaily.date, 
            IndexDaily.change_pct
        ).filter(
            IndexDaily.code == symbol
        )
        df = pd.DataFrame.from_records(query.all())
        s = pd.Series(
            data=df[1].values / 100,
            index=pd.DatetimeIndex(df[0].values)
        )
        return s.sort_index().tz_localize('UTC').dropna()