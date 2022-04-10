class Genotype:
    def __init__(self, genotype: str):
        """
        @param genotype - / separated genotype indices. e.g '0/0', '0/1', '0', '1'
        """
        self.genotype_str = genotype

    def convert_to_reference(self) -> str:
        return "/".join(["0" for _ in self.genotype_str.split("/")])

    def is_reference(self):
        return all(["0" == a for a in self.genotype_str.split("/")])
