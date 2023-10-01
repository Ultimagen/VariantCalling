import subprocess

from simppl.simple_pipeline import SimplePipeline

from ugvc import logger


def print_and_execute(
    command: str,
    output_file: str = None,
    simple_pipeline: SimplePipeline = None,
    module_name: str = None,
    shell: bool = False,
):
    """
    Print and execute command through simple_pipeline or subprocess

    Parameters
    ----------
    command: str
        shell command as a string
    output_file: str, optional
        output_file to save stdout to
    simple_pipeline: SimplePipeline, optional
        optional simple pipeline object to use as execution engine
    module_name: str, optional
        optional module name to use for simple pipeline
    shell: bool, optional
        whether to use shell for subprocess execution, default False, only relevant if simple_pipeline is None
    """
    if simple_pipeline is None:
        logger.info(command)
        cmd = command if shell else [x.strip() for x in command.strip().split(" ") if x]
        if output_file is not None:
            with open(output_file, "w", encoding="utf-8") as f:
                subprocess.call(cmd, stdout=f, shell=shell)
        else:
            subprocess.check_call(cmd, shell=shell)
    elif output_file is not None:
        simple_pipeline.print_and_run(command, out=output_file, module_name=module_name)
    else:
        simple_pipeline.print_and_run(command, module_name=module_name)
