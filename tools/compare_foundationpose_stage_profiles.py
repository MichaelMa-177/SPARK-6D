#!/usr/bin/env python3
"""Compare JSON stage profiles emitted by FoundationPose variants."""

import argparse
import csv
import json
import os
import re
import sys


def parse_profile(value):
  if '=' not in value:
    raise argparse.ArgumentTypeError('expected LABEL=PATH')
  label, path = value.split('=', 1)
  if not label or not path:
    raise argparse.ArgumentTypeError('expected non-empty LABEL=PATH')
  return label, path


def select_summary(payload, steady, mode):
  if mode:
    try:
      return payload['mode_summary'][mode]
    except KeyError as error:
      available = ', '.join(sorted(payload.get('mode_summary', {})))
      raise ValueError(
          f'mode {mode!r} is unavailable (available: {available})') from error
  return payload['steady_state_summary' if steady else 'summary']


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--profile', action='append', type=parse_profile, required=True,
      metavar='LABEL=PATH', help='profile to compare; repeat for each variant')
  parser.add_argument('--metric', choices=('gpu', 'cpu'), default='gpu')
  parser.add_argument(
      '--stat', choices=(
          'mean_ms', 'median_ms', 'p95_ms', 'p99_ms', 'min_ms', 'max_ms'),
      default='mean_ms')
  parser.add_argument(
      '--steady', action='store_true',
      help='use records at or after --stage_profile_warmup')
  parser.add_argument(
      '--mode', default=None,
      help='compare one frame mode, such as neural_refine or kalman_predict')
  parser.add_argument(
      '--match', default=None,
      help='regular expression used to select stage names')
  parser.add_argument(
      '--output_csv', default=None,
      help='optional CSV path; otherwise print a tab-separated table')
  args = parser.parse_args()

  profiles = []
  for label, path in args.profile:
    with open(path, encoding='utf-8') as profile_file:
      payload = json.load(profile_file)
    profiles.append((label, path, select_summary(
        payload, args.steady, args.mode)))

  pattern = re.compile(args.match) if args.match else None
  stages = sorted({
      stage
      for _, _, summary in profiles
      for stage in summary
      if pattern is None or pattern.search(stage)
  })
  rows = []
  for stage in stages:
    row = [stage]
    for _, _, summary in profiles:
      metric_summary = summary.get(stage, {}).get(args.metric)
      value = metric_summary.get(args.stat) if metric_summary else None
      row.append('' if value is None else f'{value:.6f}')
    rows.append(row)

  header = ['stage'] + [label for label, _, _ in profiles]
  if args.output_csv:
    output_path = os.path.abspath(args.output_csv)
    output_dir = os.path.dirname(output_path)
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as output_file:
      writer = csv.writer(output_file)
      writer.writerow(header)
      writer.writerows(rows)
    print(output_path)
  else:
    writer = csv.writer(sys.stdout, dialect='excel-tab', lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)


if __name__ == '__main__':
  main()
