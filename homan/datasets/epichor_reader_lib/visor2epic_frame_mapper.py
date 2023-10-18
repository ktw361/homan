import re
import bisect
import math
from libzhifan import io


"""
Check for all vid:
index of EPIC is less than index of VISOR,
so we round-up EPIC and round-down VISOR, for consistency.
"""

class Visor2EpicMapper:
    """ Mapping from VISOR frame to EPIC frame """

    def __init__(self, mapping: str):
        """
        Args:
            mapping: e.g. './meta_infos/mapping_visor_to_epic.json'
        """
        mapping = io.read_json(mapping)
        cvt = lambda x : int(re.search('\d{10}', x).group(0))
        vids = mapping.keys()

        self.visor = dict()
        self.epic = dict()
        for vid in vids:
            src, dst = list(zip(*mapping[vid].items()))
            src = list(map(cvt, src))
            dst = list(map(cvt, dst))
            self.visor[vid] = src
            self.epic[vid] = dst

    def __call__(self, vid, frame: int) -> int:
        i = bisect.bisect_right(self.visor[vid], frame)  # first i s.t visor[i] strictly greater than frame
        if i == 0:
            a, b = 0, self.visor[vid][i]
            p, q = 0, self.epic[vid][i]
        elif i == len(self.visor[vid]):
            a = self.visor[vid][i-1]
            p = self.epic[vid][i-1]
            return frame - a + p
        else:
            a, b = self.visor[vid][i-1], self.visor[vid][i]
            p, q = self.epic[vid][i-1], self.epic[vid][i]
        k = (frame - a) / (b - a)
        y = k * (q - p) + p
        return math.ceil(y)  # EPIC is lower than VISOR so we round up


class Epic2VisorMapper:
    """ Reversed version of Visor2EpicMapper """

    def __init__(self, mapping: str):
        """
        Args:
            mapping: e.g. './meta_infos/mapping_visor_to_epic.json'
        """
        mapping = io.read_json(mapping)
        cvt = lambda x : int(re.search('\d{10}', x).group(0))
        vids = mapping.keys()

        self.visor = dict()
        self.epic = dict()
        for vid in vids:
            src, dst = list(zip(*mapping[vid].items()))
            src = list(map(cvt, src))
            dst = list(map(cvt, dst))
            self.visor[vid] = src
            self.epic[vid] = dst

    def __call__(self, vid, frame: int) -> int:
        i = bisect.bisect_right(self.epic[vid], frame)  # first i s.t epic[i] strictly greater than frame
        if i == 0:
            a, b = 0, self.epic[vid][i]
            p, q = 0, self.visor[vid][i]
        elif i == len(self.epic[vid]):
            a = self.epic[vid][i-1]
            p = self.visor[vid][i-1]
            return frame - a + p
        else:
            a, b = self.epic[vid][i-1], self.epic[vid][i]
            p, q = self.visor[vid][i-1], self.visor[vid][i]
        k = (frame - a) / (b - a)
        y = k * (q - p) + p
        return math.floor(y)  # VISOR is greater than VISOR so we round down


if __name__ == '__main__':
    import numpy as np
    visor2epic = Visor2EpicMapper('./meta_infos/mapping_visor_to_epic.json')
    epic2visor = Epic2VisorMapper('./meta_infos/mapping_visor_to_epic.json')
    for i in range(50):
        v = np.random.randint(0, 10000)
        e = visor2epic('P01_01', v)
        v2 = epic2visor('P01_01', e)
        assert v == v2