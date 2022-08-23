def __init__(self, name=None, function=None, DNA=None):
    self.name = name
    self.function = function
    self.DNA = DNA



if __name__ == '__main__':
    po = {
        'name': 'pGW0001',

        'contained_in': 'pGW0001',  # in case of L1 / L2 put in the parent.
        'contained_in': None,  # now it's the top level (L2, or L1 not in L2)

        'function': 'Promoter',  # case L0. Can infer position in L1 from functional component
        'function': 'ST1-2',  # case L1. Can infer L2 position from BB
        'function': 'L2.PB',  # case L2

        'DNA': ['hef1a'],  # case L0
        'DNA': ['CasErec', 'uORF', 'uORF'],  # case L0 special?    
        'DNA': ['pGW0002obj', 'pGW0003obj', 'pGW0004obj'],  # case L1. point to L0 plasmid objects
        'DNA': ['pGW0005obj', 'pGW0006obj', 'pGW0007obj'],  # case L2. point to L1 plasmid objects
    } 
