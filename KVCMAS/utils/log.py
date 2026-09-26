import os
import sys
from pathlib import Path
from typing import Optional, Union
from loguru import logger
from KVCMAS.utils.const import KVCMAS_ROOT

_IMG_RUN = __import__("re").compile(r"(?:<image>){3,}|(?:<video>){3,}")


def _collapse_image_runs(record):
    """VLM prompts carry 1k-7k literal <image> tokens; collapse runs to <image>xN in every log record so [MODE]/prompt lines stay readable. """
    m = record["message"]
    if "<image><image><image>" in m or "<video><video><video>" in m:
        record["message"] = _IMG_RUN.sub(
            lambda x: (f"<video>x{x.group(0).count(chr(60)+chr(118))}" if x.group(0).startswith("<video>") else f"<image>x{x.group(0).count(chr(60)+chr(105))}"), m)


def configure_logging(
    print_level: str = "INFO",
    logfile_level: str = "DEBUG",
    log_path: Optional[Union[str, Path]] = None,
) -> None:
    logger.configure(patcher=_collapse_image_runs)
    """Console and file logging levels; log_path defaults to logs/log.txt. """
    logger.remove()
    logger.add(sys.stderr, level=print_level)
    target_path = Path(log_path) if log_path is not None else KVCMAS_ROOT / 'logs/log.txt'
    if target_path.is_dir():
        target_path = target_path / "log.txt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(target_path, level=logfile_level, rotation="10 MB")

def initialize_log_file(experiment_name: str, time_stamp: str) -> Path:
    """Initialize the log file with a start message and return its path. """
    try:
        log_file_path = KVCMAS_ROOT / f'result/{experiment_name}/logs/log_{time_stamp}.txt'
        os.makedirs(log_file_path.parent, exist_ok=True)
        with open(log_file_path, 'w') as file:
            file.write("============ Start ============\n")
    except OSError as error:
        logger.error(f"Error initializing log file: {error}")
        raise
    return log_file_path

def swarmlog(sender: str, text: str, cost: float,  prompt_tokens: int, complete_tokens: int, log_file_path: str) -> None:
    """Custom log function for swarm operations. """

    formatted_message = (
        f"{sender} | 💵Total Cost: ${cost:.5f} | "
        f"Prompt Tokens: {prompt_tokens} | "
        f"Completion Tokens: {complete_tokens} | \n {text}"
    )
    logger.info(formatted_message)

    try:
        os.makedirs(log_file_path.parent, exist_ok=True)
        with open(log_file_path, 'a') as file:
            file.write(f"{formatted_message}\n")
    except OSError as error:
        logger.error(f"Error initializing log file: {error}")
        raise

def main():
    configure_logging()

    swarmlog("SenderName", "This is a test message.", 0.123)

if __name__ == "__main__":
    main()
