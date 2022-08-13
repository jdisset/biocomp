from .library import PartsLibrary as PartsLibrary
import jax
import numpy as np

part_type_to_parameter_name = {'promoter': 'tx_rate', 'uORF': 'tl_rate'}


class Slot:
    def __init__(self, f):
        self.resolve_function = f
        self.part = None  # list means multiple parts that should map to a single parameter. Otherwise single string
        self.maps_to_parameter = None
        self.is_resolved = False

    def resolve(self, lib, *args, **kwargs):
        if not self.is_resolved:
            self.part = self.resolve_function(lib, *args, **kwargs)
            if self.part == [] or self.part == [None]:
                self.part = None
            if isinstance(self.part, list):
                mapped = [self.__mapped_parameter(lib, p) for p in self.part if p is not None]
                if len(mapped) != 1:
                    raise ValueError(f'{self.part} maps to {len(mapped)} parameters ({mapped})')
                self.maps_to_parameter = mapped[0]
            else:
                self.maps_to_parameter = self.__mapped_parameter(lib, self.part)
            if self.maps_to_parameter is not None and not isinstance(self.part, list):
                self.part = [self.part]
            self.is_resolved = True

    def __mapped_parameter(self, lib, part_name, category_to_param=part_type_to_parameter_name):
        if part_name is not None:
            if part_name in lib.pc.index:
                category = lib.pc.loc[part_name, 'category']
                if category in category_to_param:
                    return category_to_param[category]
            else:
                raise ValueError(f'Unknown part: {part_name}')
        return None

    def __repr__(self):
        if self.is_resolved:
            if self.maps_to_parameter is None:
                if self.part is None:
                    return '<empty slot>'
                else:
                    return f'<{self.part}>'
            return f'<{self.part} -> {self.maps_to_parameter}>'
        else:
            return f'<slot(unresolved, {self.resolve_function})>'


# util for a slot that resolves to a single part
def Part(name):
    return Slot(lambda *_, **__: name)


# transcription unit: 1 per L1, multiple per L2
class TranscriptionUnit:
    def __init__(self, slots):
        self.name = ''
        self.parts_slots = slots
        self.quantized_params = {}
        self.is_resolved = False

    def resolve_all_slots(self, lib, random_seed=1, random_order=True):
        rdm = jax.random.PRNGKey(random_seed)
        allrdm = jax.random.split(rdm, len(self.parts_slots))
        order = list(range(len(self.parts_slots)))
        if random_order:
            order = jax.random.permutation(rdm, len(self.parts_slots))
        for i, r in zip(order, allrdm):
            if not self.parts_slots[i].is_resolved:
                self.parts_slots[i].resolve(lib, l1=self, rdm_key=r)

        self.__get_quantized_parameters()

        assert all(s.is_resolved for s in self.parts_slots)


    def __get_quantized_parameters(self):
        for s in self.parts_slots:
            assert s.is_resolved
            if s.maps_to_parameter is not None:
                assert s.maps_to_parameter not in self.quantized_params
                self.quantized_params[s.maps_to_parameter] = s.part

    def __repr__(self):
        return f'L1({self.parts_slots})'


# a DNA source is basically a plasmid. It contains one or several transcription units,
# and can be either an L1 or a L2.
class Source:

    def __transcription_unit_from_L1(self, l1id, lib):
        l0_cols = ["insulator", "promoter", "5'UTR", "gene", "3'UTR", "terminator"]
        L0s = lib.L1s.loc[l1id][l0_cols].tolist()
        part_cols = [f'part_{i}' for i in range(1,7)]
        parts = []
        for l in L0s:
            parts += [p for p in lib.L0s.loc[l][part_cols].tolist() if p]
        tu = TranscriptionUnit([Part(p) for p in parts])
        tu.resolve_all_slots(lib)
        return tu


    def __init__(self, ratio, pid, lib):
        self.ratio = ratio
        self.pid = pid
        if self.pid in lib.L1s.index:
            self.level = 1
            self.transcription_units = [self.__transcription_unit_from_L1(self.pid, lib)]
        elif self.pid in lib.L2s.index:
            self.level = 2
            slot_cols=[f'slot_{i}' for i in range(1, 7)]
            l1ids = [s for s in lib.L2s.loc['pGW0010'][slot_cols].tolist() if s]
            self.transcription_units = [self.__transcription_unit_from_L1(l1id, lib) for l1id in l1ids]
        else:
            raise (ValueError(f'Unknown plasmid: {self.pid}'))


    def __repr__(self):
        return f'(ratio={self.ratio:.2f}, id={self.pid}), transcription units: {self.transcription_units}'


class Aggregation:
    def __init__(self, agobj, lib):
        ratios = np.array([o['qtty'] for o in agobj])
        self.qtty = ratios.sum()
        ratios = ratios / self.qtty
        self.sources = [Source(r, o['plasmid'], lib) for (r, o) in zip(ratios, agobj)]

    def __repr__(self):
        return f'total qtty = {self.qtty}, sources = {self.sources}'


class Run:
    def __init__(self, obj, lib):
        self.datafile = obj['datafile']
        self.name = obj['name']
        self.aggregations = [Aggregation(o, lib) for o in obj['content']['aggregations']]

    def __repr__(self):
        return f'{self.name}, agg = {self.aggregations}'
