class BedWriter:

    def __init__(self, output_file: str):
        self.fh = open(output_file, 'w')

    def write(self, chrom: str, start: int, end: int, description: str = None):
        self.fh.write(f'{chrom}\t{start}\t{end}')
        if description is not None:
            self.fh.write(f'\t{description}')
        self.fh.write('\n')

    def close(self):
        self.fh.close()
