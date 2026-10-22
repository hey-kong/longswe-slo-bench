"""Timing regressions. No GPU, model, dataset, or running server required."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import longswe_slo as bench


class Clock:
    def __init__(self):
        self.now = 0.0

    def read(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class HealthResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def get(self, url):
        return HealthResponse()


def options(output_dir='unused', **overrides):
    values = dict(
        output_dir=output_dir, base_url='http://unused', output_len=8,
        request_timeout=300, min_completed=1, ttft_slo=10, tps_slo=15,
        budget_seconds=10, probe_seconds=4, confirm_seconds=3,
        start_concurrency=8, max_concurrency=64, label='test',
        keep_cache=False, seed=42, allow_prefix_reuse=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def row():
    return dict(ok=True, ttft_s=1.0, e2e_s=4.0, decode_tps=20.0,
                e2e_tps=2.0, completion_tokens=8, first_chunk_tokens=1)


class TimingTests(unittest.IsolatedAsyncioTestCase):
    async def run_main(self, args, clock, stage_impl):
        with (
            patch.object(bench.time, 'perf_counter', clock.read),
            patch.object(bench, 'stage', side_effect=stage_impl),
            patch.object(bench, 'request', new=AsyncMock(return_value=row())),
            patch.object(aiohttp, 'ClientSession', return_value=Session()),
            patch.object(aiohttp, 'TCPConnector', return_value=None),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            await bench.main_async(args, ['prompt'], {})
        return json.loads((Path(args.output_dir) / 'summary.json').read_text())

    async def test_stage_sends_until_deadline_then_drains_all_requests(self):
        clock = Clock()
        args = options()
        starts = []

        async def generate(session, args, prompt, rid, **kwargs):
            if clock.read() >= kwargs['admission_deadline']:
                return None
            admitted = clock.read()
            starts.append(admitted)
            clock.advance(4)
            result = row()
            result['admitted_elapsed_s'] = admitted
            return result

        raw = io.StringIO()
        with (
            patch.object(bench.time, 'perf_counter', clock.read),
            patch.object(bench, 'request', side_effect=generate),
            patch.object(bench, 'flush_cache', new=AsyncMock()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = await bench.stage(
                None, args, ['prompt'], 1, 10, 'confirm', raw, 'test', 10,
                benchmark_start=0,
            )
        self.assertEqual(starts, [0, 4, 8])
        self.assertEqual(result['completed'], 3)
        self.assertEqual(result['drain_seconds'], 2)
        self.assertEqual(result['wall_seconds'], 12)
        self.assertEqual(len(raw.getvalue().splitlines()), 3)

    async def test_request_guard_never_posts_at_or_after_deadline(self):
        clock = Clock()
        clock.now = 10
        session = Mock()
        session.post.side_effect = AssertionError('generation admitted too late')
        with patch.object(bench.time, 'perf_counter', clock.read):
            result = await bench.request(
                session, options(), 'prompt', 'expired', admission_deadline=10,
            )
        self.assertIsNone(result)
        session.post.assert_not_called()

    async def test_expired_stage_does_not_even_flush(self):
        clock = Clock()
        clock.now = 10
        flush = AsyncMock()
        with (
            patch.object(bench.time, 'perf_counter', clock.read),
            patch.object(bench, 'flush_cache', new=flush),
        ):
            result = await bench.stage(
                None, options(), ['prompt'], 1, 10, 'probe', io.StringIO(),
                'expired', 10,
            )
        self.assertIsNone(result)
        flush.assert_not_awaited()

    async def test_short_confirmation_uses_all_remaining_send_time(self):
        clock = Clock()
        calls = []

        async def stage(session, args, prompts, c, seconds, kind, raw, run_id,
                        deadline, benchmark_start=None):
            calls.append((kind, c, seconds))
            if kind == 'probe':
                clock.advance(8)  # Four seconds sending, four draining.
            else:
                self.assertEqual(seconds, 2)
                clock.advance(seconds + 2)  # Drain ends after the global deadline.
            return dict(kind=kind, concurrency=c, status='PASS',
                        last_request_started_elapsed_s=clock.read()-2.1)

        with tempfile.TemporaryDirectory() as tmp:
            args = options(str(Path(tmp) / 'run'))
            report = await self.run_main(args, clock, stage)
        self.assertEqual([x[0] for x in calls], ['probe', 'confirm'])
        self.assertEqual(report['confirmed_concurrency'], 8)
        self.assertEqual(report['send_window_seconds'], 10)
        self.assertEqual(report['final_drain_seconds'], 2)
        self.assertEqual(report['elapsed_wall_seconds'], 12)
        self.assertLess(report['last_request_started_elapsed_s'], 10)

    async def test_insufficient_probe_continues_instead_of_ending_early(self):
        clock = Clock()
        kinds = []

        async def stage(session, args, prompts, c, seconds, kind, raw, run_id,
                        deadline, benchmark_start=None):
            kinds.append(kind)
            clock.advance(5 if kind == 'probe' else seconds)
            return dict(kind=kind, concurrency=c,
                        status='INSUFFICIENT' if kind == 'probe' else 'PASS',
                        last_request_started_elapsed_s=clock.read()-.1)

        with tempfile.TemporaryDirectory() as tmp:
            args = options(str(Path(tmp) / 'run'))
            report = await self.run_main(args, clock, stage)
        self.assertEqual(kinds, ['probe', 'continue'])
        self.assertEqual(report['highest_passing_concurrency'], 8)
        self.assertIsNone(report['confirmed_concurrency'])
        self.assertEqual(report['status'], 'PASSING_CANDIDATE')

    async def test_binary_search_and_confirmation_are_preserved(self):
        clock = Clock()
        probes = []
        outcomes = {8: 'FAIL', 4: 'PASS', 6: 'PASS', 7: 'FAIL'}

        async def stage(session, args, prompts, c, seconds, kind, raw, run_id,
                        deadline, benchmark_start=None):
            if kind == 'probe':
                probes.append(c)
            clock.advance(seconds)
            return dict(kind=kind, concurrency=c, status=outcomes[c],
                        last_request_started_elapsed_s=clock.read()-.1)

        with tempfile.TemporaryDirectory() as tmp:
            args = options(str(Path(tmp) / 'run'), budget_seconds=100,
                           probe_seconds=10, confirm_seconds=10)
            report = await self.run_main(args, clock, stage)
        self.assertEqual(probes, [8, 4, 6, 7])
        self.assertEqual(report['confirmed_concurrency'], 6)
        self.assertEqual(report['smallest_observed_fail'], 7)
        self.assertEqual(report['elapsed_wall_seconds'], 100)

    async def test_failed_reconfirmation_is_not_reported_as_stable(self):
        clock = Clock()

        async def stage(session, args, prompts, c, seconds, kind, raw, run_id,
                        deadline, benchmark_start=None):
            clock.advance(seconds)
            return dict(kind=kind, concurrency=c,
                        status='PASS' if kind == 'probe' else 'FAIL',
                        last_request_started_elapsed_s=clock.read()-.1)

        with tempfile.TemporaryDirectory() as tmp:
            args = options(str(Path(tmp) / 'run'), max_concurrency=8)
            report = await self.run_main(args, clock, stage)
        self.assertIsNone(report['confirmed_concurrency'])
        self.assertEqual(report['inconsistent_concurrencies'], [8])
        self.assertEqual(report['status'], 'INCONSISTENT')

    async def test_error_stops_search_instead_of_retrying_unsafe_workload(self):
        clock = Clock()
        kinds = []

        async def stage(session, args, prompts, c, seconds, kind, raw, run_id,
                        deadline, benchmark_start=None):
            kinds.append(kind)
            clock.advance(1)
            return dict(kind=kind, concurrency=c, status='ERROR',
                        last_request_started_elapsed_s=.1)

        with tempfile.TemporaryDirectory() as tmp:
            args = options(str(Path(tmp) / 'run'))
            report = await self.run_main(args, clock, stage)
        self.assertEqual(kinds, ['probe'])
        self.assertEqual(report['status'], 'INCONCLUSIVE')


if __name__ == '__main__':
    unittest.main()
