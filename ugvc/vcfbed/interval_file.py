import os
from typing import Optional

from simppl.simple_pipeline import SimplePipeline

from ugvc import logger
from ugvc.utils.exec_utils import print_and_execute


class IntervalFile:
    def __init__(
            self,
            sp: SimplePipeline = None,
            cmp_intervals: Optional[str] = None,
            ref: Optional[str] = None,
            ref_dict: Optional[str] = None,

    ):
        self.sp = sp
        # determine the file type and create the other temporary copy
        if cmp_intervals is None:
            self._is_none: bool = True
            self._interval_list_file_name: Optional[str] = None
            self._bed_file_name: Optional[str] = None

        elif cmp_intervals.endswith(".interval_list"):
            self._interval_list_file_name = cmp_intervals
            # create the interval bed file
            self.__execute(
                f"picard IntervalListToBed I={cmp_intervals} O={os.path.splitext(cmp_intervals)[0]}.bed")

            self._bed_file_name = f"{os.path.splitext(cmp_intervals)[0]}.bed"
            self._is_none = False

        elif cmp_intervals.endswith(".bed"):
            self._bed_file_name = cmp_intervals
            # deduce ref_dict
            if ref_dict is None:
                ref_dict = f"{ref}.dict"
            if not os.path.isfile(ref_dict):
                logger.error(f"dict file does not exist: {ref_dict}")

            # create the interval list file
            self.__execute(f"picard BedToIntervalList I={cmp_intervals} "
                           f"O={os.path.splitext(cmp_intervals)[0]}.interval_list SD={ref_dict}")

            self._interval_list_file_name = (
                f"{os.path.splitext(cmp_intervals)[0]}.interval_list"
            )
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
        print_and_execute(command, output_file=output_file, simple_pipeline=self.sp)
