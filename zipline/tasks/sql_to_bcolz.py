import sys
import logbook
# 设置显示日志
logbook.set_datetime_format('local')
logbook.StreamHandler(sys.stdout).push_application()

from zipline.pipeline.fundamentals.static import write_static_info_to_bcolz
from zipline.pipeline.fundamentals.financal import write_financal_data_to_bcolz
from zipline.pipeline.fundamentals.dynamic import write_dynamic_data_to_bcolz


def main():
    write_static_info_to_bcolz()
    write_financal_data_to_bcolz()
    write_dynamic_data_to_bcolz()


if __name__ == '__main__':
    main()
