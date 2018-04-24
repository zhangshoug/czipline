import blaze as bz

from cswd.sql.base import db_path

def sqlite_db_path(db_name):
    r_dir = db_path(db_name)
    return "sqlite:///{}".format(r_dir)

def sqlite_table_path(db_name, table_name):
    return '{}::{}'.format(sqlite_db_path(db_name), table_name)

STOCK_DB = bz.data(sqlite_db_path('stock.db'))