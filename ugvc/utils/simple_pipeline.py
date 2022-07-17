import logging
import os
import subprocess
import sys
import time
from multiprocessing.pool import ThreadPool

STDERR_LEVEL = 25
logging.addLevelName(STDERR_LEVEL, "STDERR_INFO")


class SimplePipeline:
    """
    Provide a simple framework for running a pipeline of bash commands.
    Each pipeline stage can consist of a single command, or a list of commands to be run sequentially
    user can control initial -fc and final -lc stages to run
    can dry-run with -d
    prints each executed command together with timestamp, and stage number
    """
    command_counter = 0

    def __init__(self, start, end, debug=False, print_timing=True, name: str = __name__, output_stream=sys.stderr):
        if debug:
            self.execute = False
        else:
            self.execute = True
        self.start = int(start)
        self.end = int(end)
        self.print_timing = print_timing
        self.name = name
        self.logging_extra = None

        # logger = logging.getLogger('simple_pipeline')
        logger = logging.getLogger('ugvc.utils.simple_pipeline')
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if output_stream is not None:
            # create formatter
            formatter = logging.Formatter("%(asctime)s - %(module_var)s - %(levelname)s - %(message)s")

            # create console handler and set level to debug
            ch = logging.StreamHandler(stream=output_stream)
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            logger.handlers = [ch]


        self.logger = logger

    def print_and_run(self, command: str, out: str = None, err: str = None, module_name: str = None):
        self.command_counter += 1
        if out is not None:
            command += ' > ' + out
        if err is not None:
            command += ' 2> ' + err
        if module_name is None:
            self.logging_extra = {'module_var': self.name}
        else:
            self.logging_extra = {'module_var': module_name}
        if self.command_counter < self.start or self.command_counter > self.end:
            self._print_skip_command(command)
        else:
            self._private_print_and_run(command, command)

    def print_and_run_clt(self, clt, positional_args: list, optional_args: dict, flags: set = {}):
        self.command_counter += 1
        positional_args = [str(x) for x in positional_args]

        tool_name = clt.__module__.split('.')[-1]
        all_args = positional_args
        [all_args.extend([str(k), str(v)]) for k, v in optional_args.items()]
        [all_args.extend([str(k)]) for k in flags]
        command_for_print = f'python -m <module> {tool_name} ' + ' '.join(all_args)
        if self.command_counter < self.start or self.command_counter > self.end:
            self._print_skip_command(command_for_print)
            return
        self._print_command(command_for_print)
        if self.execute:
            start = time.time()
            clt(['internal_tool'] + all_args)
            end = time.time()
            if self.print_timing:
                self.logger.info('Time elapsed %s: %d s' % (tool_name, end - start), extra=self.logging_extra)

    def run_parallel(self, commands, max_num_of_processes):
        self.command_counter += 1
        if self.command_counter < self.start or self.command_counter > self.end:
            for command in commands:
                self._print_skip_command(command)
            return

        if self.execute:
            start = time.time()
            pool = ThreadPool(max_num_of_processes)
            results = []
            program_name = self.get_program_name(commands[0])
            sub_command_index = 1
            for command in commands:
                sub_command_index_symbol = f'[{sub_command_index}]'
                self._print_command(f'{sub_command_index_symbol} {command}')
                results.append(pool.apply_async(self._run_and_time, [command, sub_command_index_symbol]))
                sub_command_index += 1
            pool.close()
            pool.join()

            for result in results:
                return_value, time_to_run, command = result._value
                if return_value != 0:
                    raise RuntimeError(f'Exit SimplePipeline due to failure of command: {command} in runParallel call')

            if self.print_timing:
                end = time.time()
                self.logger.info('Time elapsed %s: %d s' % (program_name, end - start), extra=self.logging_extra)
        else:
            [self._print_command(command) for command in commands]

    @staticmethod
    def get_program_name(command):
        if 'nbtk.jar' in command:
            arr = command.split()
            jar_field_was_last = False
            for field in arr:
                if 'nbtk.jar' in field:
                    jar_field_was_last = True
                elif jar_field_was_last:
                    return field
        if command.split()[0] in ['python', 'python2.7', 'python3.7', 'perl', 'java', 'sh', 'bash']:
            program_name = os.path.basename(command.split()[1])
        else:
            program_name = os.path.basename(command.split()[0])
        return program_name

    @staticmethod
    def add_parse_args(parser):
        parser.add_argument('-d', action='store_const', const=True, help='Print debug info instead of executing')
        parser.add_argument('-fc', default=0, help='Index of the first command to execute')
        parser.add_argument('-lc', default=10000, help='Index of the last command to execute')

    def _private_print_and_run(self, command: str, command_to_print: str):
        program_name = self.get_program_name(command)

        self._print_command(command_to_print)
        if self.execute:
            rv, time_in_seconds, command = self._run_and_time(command)
            if rv != 0:
                raise RuntimeError("Failed: %d) %s" % (self.command_counter, command_to_print))
            if self.print_timing:
                self.logger.info('Time elapsed %s: %d s' % (program_name, time_in_seconds), extra=self.logging_extra)

    def _print_command(self, command_to_print):
        self.logger.info(f'{self.command_counter}) {command_to_print}', extra=self.logging_extra)

    def _print_skip_command(self, command_to_print):
        self.logger.info(f'Skip: {self.command_counter}) {command_to_print}', extra=self.logging_extra)

    def _run_and_time(self, command, command_index_message=''):
        """

        :param command: command to execute
        :param command_index_message: in case running multiprocesses the index of the command in batch 1-based,
                              leave 0 for single command.
        :return: returncode, time_in_seconds, command
        """
        start = time.time()
        completed_process = subprocess.run(['/bin/bash', '-c', command], capture_output=True, encoding='UTF8')
        rv = completed_process.returncode
        for line in completed_process.stdout.split('\n'):
            if line != '':
                self.logger.info(f'{command_index_message} {line}', extra=self.logging_extra)
        for line in completed_process.stderr.split('\n'):
            if line != '':
                lower_case = line.lower()
                if 'error' in lower_case or 'failure' in lower_case:
                    self.logger.error(f'{command_index_message} {line}', extra=self.logging_extra)
                else:
                    self.logger.log(STDERR_LEVEL, f'{command_index_message} {line}', extra=self.logging_extra)
        end = time.time()
        return rv, end - start, command
