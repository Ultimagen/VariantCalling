import subprocess

from simppl.simple_pipeline import SimplePipeline
from ugvc import logger


def print_and_execute(command: str,
                      output_file: str = None,
                      simple_pipeline: SimplePipeline = None):
    """
    Print and execute command through simple_pipeline or subprocess
    Parameters
    ----------
    command - shell command as a string
    logger - logger to log the command before execution
    output_file - output_file to save stdout to
    simple_pipeline - optional simple pipeline object to use as execution engine
    """
    if simple_pipeline is None:
        logger.info(command)
        cmd = command.split(' ')
        if output_file is not None:
            with open(output_file, "w") as f:
                subprocess.call(cmd, stdout=f)
        else:
            subprocess.check_call(cmd)
    else:
        if output_file is not None:
            simple_pipeline.print_and_run(f'{command} > {output_file}')
        else:
            simple_pipeline.print_and_run(command)
