from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable

from trading.config import LiveTradingConfig
from trading.graphs import (
    TradingGraphContext,
    build_opening_buy_graph,
    build_position_monitor_graph,
    build_post_close_graph,
    build_pre_open_graph,
)
from trading.reconciliation import OrderReconciler
from trading.scheduled_orders import execute_due_scheduled_orders


@dataclass
class JobResult:
    job: str
    status: str
    result: dict | None = None
    error: str | None = None


class LiveTradingService:
    """External scheduler for finite LangGraph jobs."""

    def __init__(
        self,
        context: TradingGraphContext,
        reconciler: OrderReconciler | None = None,
    ):
        self.context = context
        self.config: LiveTradingConfig = context.config
        self.pre_open_graph = build_pre_open_graph(context)
        self.opening_buy_graph = build_opening_buy_graph(context)
        self.position_monitor_graph = build_position_monitor_graph(context)
        self.post_close_graph = build_post_close_graph(context)
        self.reconciler = reconciler

    def _execute_once(
        self,
        job: str,
        job_key: str,
        graph,
        now: datetime,
    ) -> JobResult:
        if not self.context.store.claim_job(job_key, {"job": job, "now": now.isoformat()}):
            return JobResult(job, "DUPLICATE_SKIPPED")
        try:
            result = graph.invoke({"now": now}, config={"recursion_limit": 1000})
            self.context.store.finish_job(job_key, "COMPLETED", result.get("report", {}))
            return JobResult(job, "COMPLETED", result=result)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self.context.store.finish_job(job_key, "FAILED", {"error": message})
            self.context.store.audit("JOB_FAILED", job_key, {"job": job, "error": message})
            return JobResult(job, "FAILED", error=message)

    def run_pre_open(self, now: datetime) -> JobResult:
        local = now.astimezone(self.config.timezone)
        key = f"{local.date().isoformat()}:pre_open:{self.config.strategy_version}"
        return self._execute_once("pre_open", key, self.pre_open_graph, now)

    def run_opening_buy(self, now: datetime) -> JobResult:
        local = now.astimezone(self.config.timezone)
        key = f"{local.date().isoformat()}:opening_buy:{self.config.strategy_version}"
        return self._execute_once("opening_buy", key, self.opening_buy_graph, now)

    def run_scheduled_orders(self, now: datetime) -> JobResult:
        try:
            report = execute_due_scheduled_orders(self.context, now)
            return JobResult("scheduled_orders", "COMPLETED", result=report)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            return JobResult("scheduled_orders", "FAILED", error=message)

    def run_monitor(self, now: datetime) -> JobResult:
        local = now.astimezone(self.config.timezone)
        bucket = local.replace(second=0, microsecond=0).isoformat()
        key = f"{bucket}:position_monitor:{self.config.strategy_version}"
        return self._execute_once(
            "position_monitor", key, self.position_monitor_graph, now
        )

    def run_post_close(self, now: datetime) -> JobResult:
        local = now.astimezone(self.config.timezone)
        key = f"{local.date().isoformat()}:post_close:{self.config.strategy_version}"
        return self._execute_once("post_close", key, self.post_close_graph, now)

    def run_reconciliation(
        self, now: datetime, *, post_close: bool = False
    ) -> JobResult:
        if self.reconciler is None:
            return JobResult("order_reconciliation", "NOT_CONFIGURED")
        local = now.astimezone(self.config.timezone)
        bucket = (
            f"{local.date().isoformat()}:post_close"
            if post_close else local.replace(second=0, microsecond=0).isoformat()
        )
        key = f"{bucket}:order_reconciliation:{self.config.strategy_version}"
        if not self.context.store.claim_job(
            key, {"job": "order_reconciliation", "now": now.isoformat()}
        ):
            return JobResult("order_reconciliation", "DUPLICATE_SKIPPED")
        try:
            report = asdict(self.reconciler.reconcile())
            self.context.store.finish_job(key, "COMPLETED", report)
            return JobResult("order_reconciliation", "COMPLETED", result=report)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self.context.store.finish_job(key, "FAILED", {"error": message})
            return JobResult("order_reconciliation", "FAILED", error=message)

    def run_due(self, now: datetime | None = None) -> list[JobResult]:
        now = (now or datetime.now(self.config.timezone)).astimezone(
            self.config.timezone
        )
        if not self.context.calendar.is_trading_day(now.date()):
            return []
        results: list[JobResult] = []
        current = now.time().replace(tzinfo=None)

        if current >= self.config.pre_open_at:
            results.append(self.run_pre_open(now))
        if self.context.calendar.phase(now) == "REGULAR":
            results.append(self.run_scheduled_orders(now))
        if self.config.entry_at <= current < self.config.market_close_at:
            results.append(self.run_opening_buy(now))
        if self.context.calendar.phase(now) == "REGULAR":
            results.append(self.run_reconciliation(now))
            results.append(self.run_monitor(now))
        if current >= self.config.post_close_at:
            results.append(self.run_reconciliation(now, post_close=True))
            results.append(self.run_post_close(now))
        return results

    def run_forever(
        self,
        *,
        poll_seconds: float = 1.0,
        on_result: Callable[[JobResult], None] | None = None,
    ) -> None:
        print(
            "Live trading scheduler started "
            f"(dry_run={self.config.dry_run}, timezone={self.config.timezone})",
            flush=True,
        )
        try:
            while True:
                for result in self.run_due():
                    if result.status != "DUPLICATE_SKIPPED":
                        if on_result:
                            on_result(result)
                        else:
                            print(
                                f"[{datetime.now(self.config.timezone).isoformat()}] "
                                f"{result.job}: {result.status}"
                                + (f" ({result.error})" if result.error else ""),
                                flush=True,
                            )
                time.sleep(max(0.2, poll_seconds))
        except KeyboardInterrupt:
            print("Live trading scheduler stopped", flush=True)
