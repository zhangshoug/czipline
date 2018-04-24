import sys
import logbook
# 设置显示日志
logbook.set_datetime_format('local')
logbook.StreamHandler(sys.stdout).push_application()

from zipline.pipeline.fundamentals.ctable import convert_sql_data_to_bcolz


def main():
    tables = ['balance_sheets', 'profit_statements', 'cashflow_statements',
              'chnls', 'cznls', 'ylnls', 'yynls', 'zyzbs',
              'special_treatments', 'adjustments', 'margins', 'issues',
              'regions', 'thx_industries', 'csrc_industries',
              'concepts']
    convert_sql_data_to_bcolz(tables)


if __name__ == '__main__':
    main()
