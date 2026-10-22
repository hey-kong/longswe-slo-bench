#!/usr/bin/env python3
"""SWE-Bench entry point using the shared SLO search and request measurements."""
from longswe_slo import cli_main


if __name__ == '__main__':
    cli_main(dataset_kind='swebench')
