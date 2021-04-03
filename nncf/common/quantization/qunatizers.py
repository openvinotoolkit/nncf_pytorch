# """
#  Copyright (c) 2021 Intel Corporation
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# """

def calculate_symmetric_level_ranges(num_bits, signed, narrow_range=False):
    """
    :param num_bits:
    :param signed:
    :param narrow_range:
    :return:
    """
    levels = 2 ** num_bits

    if signed:
        level_high = (levels // 2) - 1
        level_low = -(levels // 2)
    else:
        level_high = levels - 1
        level_low = 0

    if narrow_range:
        level_low = level_low + 1

    return level_low, level_high, levels


def calculate_asymmetric_level_ranges(num_bits, narrow_range=False):
    """
    :param num_bits:
    :param narrow_range:
    :return:
    """
    levels = 2 ** num_bits
    level_high = levels - 1
    level_low = 0

    if narrow_range:
        level_low = level_low + 1

    return level_low, level_high, levels