#!/env/python
# Copyright 2024 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Add recalibrated qualities to the annotated de novo VCF

from __future__ import annotations

import sys

from ugvc.joint import denovo_refinement


def run(argv: list):
    "Add qualities recalibrated through somatic calling to the annotated de novo VCF"
    denovo_refinement.main(argv[1:])


if __name__ == "__main__":
    run(sys.argv)
