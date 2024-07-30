from __future__ import annotations

import os

from simppl.simple_pipeline import SimplePipeline

from ugvc import logger
from ugbio_core.exec_utils import print_and_execute


class IntervalFile:
    def __init__(
        self,
        sp: SimplePipeline = None,
        interval: str | None = None,
        ref: str | None = None,
        ref_dict: str | None = None,
    ):
        self.sp = sp
        # determine the file type and create the other temporary copy
        if interval is None:
            self._is_none: bool = True
            self._interval_list_file_name: str | None = None
            self._bed_file_name: str | None = None

        elif interval.endswith(".interval_list"):
            self._interval_list_file_name = interval
            # create the interval bed file
            if not os.path.exists(f"{os.path.splitext(interval)[0]}.bed"):
                self.__execute(f"picard IntervalListToBed I={interval} O={os.path.splitext(interval)[0]}.bed")

            self._bed_file_name = f"{os.path.splitext(interval)[0]}.bed"
            self._is_none = False

        elif interval.endswith(".bed"):
            self._bed_file_name = interval
            # deduce ref_dict
            if ref_dict is None:
                ref_dict = f"{ref}.dict"
            if not os.path.isfile(ref_dict):
                logger.error(f"dict file does not exist: {ref_dict}")

            # create the interval list file
            if not os.path.exists(f"{os.path.splitext(interval)[0]}.interval_list"):
                self.__execute(
                    f"picard BedToIntervalList I={interval} "
                    f"O={os.path.splitext(interval)[0]}.interval_list SD={ref_dict}"
                )

            self._interval_list_file_name = f"{os.path.splitext(interval)[0]}.interval_list"
            self._is_none = False
        else:
            logger.error("the cmp_intervals should be of type interval list or bed")
            self._is_none = True
            self._interval_list_file_name = None
            self._bed_file_name = None

    def as_bed_file(self):
        return self._bed_file_name

    def as_interval_list_file(self):
        return self._interval_list_file_name

    def is_none(self):
        return self._is_none

    def __execute(self, command: str, output_file: str = None):
        print_and_execute(command, output_file=output_file, simple_pipeline=self.sp, module_name=__name__)
