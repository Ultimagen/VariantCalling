import subprocess

from ugvc.utils.simple_pipeline import SimplePipeline

from ugvc import logger


def print_and_execute(command: str, output_file: str = None, simple_pipeline: SimplePipeline = None, module_name: str = None):
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
        cmd = command.split(" ")
        if output_file is not None:
            with open(output_file, "w", encoding="utf-8") as f:
                subprocess.call(cmd, stdout=f)
        else:
            subprocess.check_call(cmd)
    elif output_file is not None:
        simple_pipeline.print_and_run(command, out=output_file, module_name=module_name)
    else:
        simple_pipeline.print_and_run(command, module_name=module_name)
