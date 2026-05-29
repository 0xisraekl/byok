"""
main.py — BYOK Command Line Interface

Commands:
    byok serve          Start the proxy server (what Hermes connects to)
    byok route          Classify a task and show which model would be chosen
    byok models         Show your configured model pool
    byok log            Show recent routing decisions
    byok spend          Show monthly spend per model vs. limits
"""

from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

load_dotenv()

console = Console()

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "models.yaml"
DB_PATH = Path("byok.db")


# ─────────────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """
    BYOK — Bring Your Own Key

    Intelligent model routing for AI agent frameworks.
    Point your agent at the BYOK proxy and it will automatically
    route each task to the best model in your pool.
    """
    pass


# ── byok serve ────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--port", default=8000, help="Port to listen on (default: 8000)")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev mode)")
def serve(port: int, host: str, reload: bool):
    """
    Start the BYOK proxy server.

    After this is running, point Hermes Agent at:
        base_url = "http://localhost:8000/v1"
        api_key  = "byok"
    """
    import uvicorn
    from byok.core.registry import ModelRegistry

    # Show startup summary
    try:
        reg = ModelRegistry(CONFIG_PATH)
        available = reg.available_models()
    except FileNotFoundError:
        console.print(f"[red]✗ Could not find {CONFIG_PATH}[/red]")
        raise SystemExit(1)

    console.print(Panel.fit(
        f"[bold green]BYOK Proxy Server[/bold green]\n\n"
        f"  URL:     [cyan]http://{host}:{port}[/cyan]\n"
        f"  Docs:    [cyan]http://{host}:{port}/docs[/cyan]\n\n"
        f"  [bold]Models in your pool:[/bold] {len(available)}\n"
        + "\n".join(f"  • {m.name} ({m.provider})" for m in available)
        + "\n\n"
        f"  [dim]Hermes config →  base_url = \"http://localhost:{port}/v1\"[/dim]\n"
        f"  [dim]               api_key  = \"byok\"[/dim]",
        title="🔀 BYOK",
        border_style="green",
    ))

    uvicorn.run(
        "byok.proxy.server:app",
        host=host,
        port=port,
        reload=reload,
    )


# ── byok route ────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("message", required=False)
@click.option("--task", "-t", help="Force a task type (coding, reasoning, etc.)")
@click.option("--tools", is_flag=True, help="Simulate a request that includes tools")
@click.option("--private", is_flag=True, help="Simulate a private/local-only request")
@click.option(
    "--mode",
    default=None,
    type=click.Choice(["balanced", "cheap", "quality", "private", "speed"]),
    help="Routing mode: balanced, cheap, quality, private, or speed. Overrides agent policy.",
)
@click.option("--agent", help="Simulate a sub-agent role, e.g. coder, research_agent, writer")
@click.option("--max-cost", type=float, help="Maximum estimated USD cost allowed for this routed call")
@click.option("--run-id", help="Agent run/session id for shared run budget checks")
@click.option("--max-run-cost", type=float, help="Maximum estimated USD spend allowed across this run/session")
@click.option("--max-output-tokens", type=int, help="Maximum output tokens allowed for this routed call")
def route(
    message: str,
    task: str,
    tools: bool,
    private: bool,
    mode: str | None,
    agent: str | None,
    max_cost: float | None,
    run_id: str | None,
    max_run_cost: float | None,
    max_output_tokens: int | None,
):
    """
    Classify a task and show which model BYOK would choose.

    Test routing without making any real API calls.

    Examples:
        byok route "Write a function to parse JSON"
        byok route "Summarize this article" --task summarization
        byok route "Analyze these contracts" --private
    """
    from byok.core.classifier import TaskClassifier, PRIVACY_SIGNALS
    from byok.core.policy import RoutingPolicy
    from byok.core.registry import ModelRegistry
    from byok.core.router import ModelRouter
    from byok.core.token_budget import TokenBudgeter
    from byok.storage.spend_tracker import SpendTracker

    if not message:
        message = click.prompt("Enter a task message")

    reg = ModelRegistry(CONFIG_PATH)
    policy = RoutingPolicy(Path(__file__).parent.parent.parent / "config" / "routing_policy.yaml")
    tracker = SpendTracker(DB_PATH)
    clf = TaskClassifier()

    # Build a fake messages list
    messages = [{"role": "user", "content": message}]
    if private:
        messages.insert(0, {"role": "system", "content": "Keep this private and confidential."})

    fake_tools = [{"type": "function", "function": {"name": "search"}}] if tools else []
    task_profile = clf.classify(messages, fake_tools)
    if agent:
        task_profile.agent_role = agent

    if task:
        task_profile.task_type = task  # allow manual override
    elif task_profile.agent_role:
        policy_task = policy.task_for_agent(task_profile.agent_role)
        if policy_task:
            task_profile.task_type = policy_task

    controls = policy.controls_for(
        agent_role=task_profile.agent_role,
        explicit_mode=mode,
        explicit_max_cost_usd=max_cost,
        explicit_max_output_tokens=max_output_tokens,
    )
    run_spent = tracker.get_run_spend(run_id) if run_id else 0.0
    run_remaining = _remaining_run_budget(run_spent, max_run_cost)
    effective_max_cost = _combine_cost_limits(controls.max_cost_usd, run_remaining)
    rtr = ModelRouter(reg, tracker, mode=controls.mode)

    decision = rtr.route(
        task_profile,
        max_cost_usd=effective_max_cost,
        requested_max_tokens=controls.max_output_tokens,
    )
    if decision is not None:
        token_budget = TokenBudgeter().budget_for_model_cost(
            task=task_profile,
            cost_per_1k_input=decision.selected_model.cost_per_1k_input,
            cost_per_1k_output=decision.selected_model.cost_per_1k_output,
            mode=controls.mode,
            requested_max_tokens=controls.max_output_tokens,
            max_cost_usd=effective_max_cost,
        )
    else:
        token_budget = TokenBudgeter().budget_for(task_profile, controls.mode, controls.max_output_tokens)

    # ── Display results ──────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        f"[bold]Message:[/bold] {message[:80]}{'...' if len(message) > 80 else ''}",
        border_style="dim",
    ))

    # Task profile
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="bold")
    t.add_row("Mode", f"[magenta]{controls.mode}[/magenta]")
    t.add_row("Policy source", controls.source)
    if task_profile.agent_role:
        t.add_row("Agent role", task_profile.agent_role)
    if controls.max_cost_usd is not None:
        t.add_row("Max cost", f"${controls.max_cost_usd:.5f}")
    if run_id:
        t.add_row("Run ID", run_id)
        t.add_row("Run spent", f"${run_spent:.5f}")
    if max_run_cost is not None:
        t.add_row("Max run cost", f"${max_run_cost:.5f}")
    if effective_max_cost is not None:
        t.add_row("Effective max cost", f"${effective_max_cost:.5f}")
    t.add_row("Task type", f"[cyan]{task_profile.task_type}[/cyan]")
    t.add_row("Difficulty", task_profile.difficulty)
    t.add_row("Context tokens", str(task_profile.context_tokens))
    t.add_row("Has tools", "yes" if task_profile.has_tools else "no")
    t.add_row("Privacy required", "[red]yes[/red]" if task_profile.privacy_required else "no")
    t.add_row("Confidence", f"{task_profile.confidence:.0%}")
    console.print(Panel(t, title="Task Profile", border_style="blue"))

    b = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    b.add_column(style="dim")
    b.add_column(style="bold")
    b.add_row("Max output tokens", str(token_budget.max_output_tokens))
    b.add_row("Raw estimate", str(token_budget.raw_estimated_output_tokens))
    b.add_row("Saved tokens", f"{token_budget.saved_tokens} ({token_budget.savings_pct:.0f}%)")
    b.add_row("Reason", token_budget.reason)
    console.print(Panel(b, title="Token budget", border_style="magenta"))

    if decision is None:
        console.print("[red]✗ No model available for this task.[/red]")
        console.print("Check your models.yaml and .env file.")
        return

    # Routing decision
    m = decision.selected_model
    r = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    r.add_column(style="dim")
    r.add_column(style="bold")
    r.add_row("Selected model", f"[green]{m.name}[/green]")
    r.add_row("Provider", m.provider)
    r.add_row("Reason", decision.reason)
    r.add_row("Score", str(round(decision.score, 1)))
    r.add_row("Quality estimate", f"{decision.quality_estimate:.0%}")
    r.add_row("Est. cost", f"${decision.estimated_cost_usd:.5f}")
    r.add_row("Est. output tokens", str(decision.estimated_output_tokens))
    if decision.best_quality_model:
        r.add_row("Best quality in pool", decision.best_quality_model)
    if decision.premium_reference_cost_usd is not None:
        r.add_row("Premium ref. cost", f"${decision.premium_reference_cost_usd:.5f}")
    if decision.estimated_savings_usd is not None and decision.estimated_savings_usd > 0:
        r.add_row("Est. savings", f"${decision.estimated_savings_usd:.5f} ({decision.estimated_savings_pct:.0f}%)")
    r.add_row("Latency", m.latency)

    if decision.alternatives:
        alts = ", ".join(f"{n} ({s:.1f})" for n, s in decision.alternatives)
        r.add_row("Runners-up", alts)

    console.print(Panel(r, title="✓ Routing Decision", border_style="green"))
    console.print()


def _remaining_run_budget(run_spent_usd: float, max_run_cost_usd: float | None) -> float | None:
    if max_run_cost_usd is None:
        return None
    return max(max_run_cost_usd - run_spent_usd, 0.0)


def _combine_cost_limits(*limits: float | None) -> float | None:
    active = [limit for limit in limits if limit is not None]
    if not active:
        return None
    return min(active)


# ── byok doctor ───────────────────────────────────────────────────────────────

@cli.command()
def doctor():
    """Check BYOK config, model keys, local services, and spend DB."""
    from byok.core.doctor import build_doctor_checks
    from byok.core.registry import ModelRegistry

    console.print()
    console.print(Panel.fit("[bold]BYOK Doctor[/bold]\nChecking local setup without making paid API calls.", border_style="cyan"))

    models_list = None
    try:
        models_list = ModelRegistry(CONFIG_PATH).all_models()
    except FileNotFoundError:
        models_list = None

    checks = build_doctor_checks(CONFIG_PATH, DB_PATH, models_list)

    table = Table(box=box.ROUNDED)
    table.add_column("Status", width=10)
    table.add_column("Check", style="bold")
    table.add_column("Details")

    icon = {
        "ok": "[green]✅ ok[/green]",
        "warning": "[yellow]⚠ warn[/yellow]",
        "error": "[red]✗ error[/red]",
    }
    for check in checks:
        table.add_row(icon.get(check.status, check.status), check.name, check.detail)

    console.print(table)

    if any(c.status == "error" for c in checks):
        console.print("[red]Fix errors before running the proxy.[/red]\n")
    elif any(c.status == "warning" for c in checks):
        console.print("[yellow]BYOK can run, but warnings may limit routing options.[/yellow]\n")
    else:
        console.print("[green]BYOK setup looks healthy.[/green]\n")


# ── byok specialties ──────────────────────────────────────────────────────────

@cli.command()
def specialties():
    """Show which configured models are best for each task type."""
    from byok.core.classifier import TaskProfile
    from byok.core.registry import ModelRegistry
    from byok.core.router import ModelRouter
    from byok.storage.spend_tracker import SpendTracker

    tasks = [
        "coding",
        "reasoning",
        "math",
        "writing",
        "summarization",
        "extraction",
        "data_analysis",
        "tool_calling",
        "simple_chat",
    ]
    reg = ModelRegistry(CONFIG_PATH)
    router = ModelRouter(reg, SpendTracker(DB_PATH), mode="balanced")
    models_list = [m for m in reg.all_models() if m.enabled]

    table = Table(title="Best models by task", box=box.ROUNDED, show_lines=True)
    table.add_column("Task", style="cyan")
    table.add_column("Best quality", style="bold green")
    table.add_column("Best value", style="bold yellow")
    table.add_column("Notes")

    for task_type in tasks:
        profile = TaskProfile(
            task_type=task_type,
            secondary_types=[],
            difficulty="medium",
            context_tokens=1000,
            has_tools=(task_type == "tool_calling"),
            privacy_required=False,
            confidence=0.9,
        )
        candidates = [m for m in models_list if not profile.has_tools or m.supports_tools]
        if not candidates:
            table.add_row(task_type, "—", "—", "no configured model supports this task")
            continue

        ranked_quality = sorted(
            candidates,
            key=lambda m: (-router._quality_for(m, profile), router._estimate_cost(m, profile), m.priority),
        )
        value_candidates = [
            m for m in candidates
            if router._quality_for(m, profile) >= router.GOOD_ENOUGH_BY_DIFFICULTY[profile.difficulty]
        ] or candidates
        ranked_value = sorted(
            value_candidates,
            key=lambda m: (-(router._quality_for(m, profile) / max(router._estimate_cost(m, profile), 0.000001)), -router._quality_for(m, profile)),
        )
        best_quality = ranked_quality[0]
        best_value = ranked_value[0]
        q_score = router._quality_for(best_quality, profile)
        v_score = router._quality_for(best_value, profile)
        v_cost = router._estimate_cost(best_value, profile)
        table.add_row(
            task_type,
            f"{best_quality.name} ({q_score:.0%})",
            f"{best_value.name} ({v_score:.0%}, ${v_cost:.5f})",
            "uses task_quality overrides when present",
        )

    console.print(table)


# ── byok models ───────────────────────────────────────────────────────────────

@cli.command()
def models():
    """Show all models in your configured pool."""
    from byok.core.registry import ModelRegistry
    from byok.storage.spend_tracker import SpendTracker

    reg = ModelRegistry(CONFIG_PATH)
    tracker = SpendTracker(DB_PATH)
    monthly_spend = tracker.get_all_monthly_spend()

    table = Table(
        title="Your Model Pool",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Status", width=6)
    table.add_column("Name", style="bold")
    table.add_column("Provider")
    table.add_column("Strengths")
    table.add_column("Context")
    table.add_column("Cost/1k in")
    table.add_column("Spend this month")
    table.add_column("Limit")

    for m in reg.all_models():
        if not m.enabled:
            status = "[dim]○ off[/dim]"
        elif not m.has_valid_key:
            status = "[yellow]⚠ key[/yellow]"
        else:
            status = "[green]● on[/green]"

        spent = monthly_spend.get(m.name, 0.0)
        limit = m.spend_limit_monthly_usd

        if limit > 0:
            pct = spent / limit
            if pct >= 1.0:
                limit_str = f"[red]${limit:.0f} (FULL)[/red]"
                spend_str = f"[red]${spent:.3f}[/red]"
            elif pct >= 0.8:
                limit_str = f"[yellow]${limit:.0f}[/yellow]"
                spend_str = f"[yellow]${spent:.3f}[/yellow]"
            else:
                limit_str = f"${limit:.0f}"
                spend_str = f"${spent:.3f}"
        else:
            limit_str = "[dim]unlimited[/dim]"
            spend_str = f"${spent:.3f}" if spent > 0 else "[dim]$0[/dim]"

        table.add_row(
            status,
            m.name,
            m.provider,
            ", ".join(m.strengths[:3]),
            f"{m.context_window // 1000}k",
            f"${m.cost_per_1k_input:.5f}",
            spend_str,
            limit_str,
        )

    console.print()
    console.print(table)
    console.print()


# ── byok log ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--limit", "-n", default=15, help="Number of recent entries to show")
def log(limit: int):
    """Show recent routing decisions."""
    from byok.storage.spend_tracker import SpendTracker

    tracker = SpendTracker(DB_PATH)
    records = tracker.get_recent(limit)

    if not records:
        console.print("[dim]No routing history yet. Start the server and send some requests.[/dim]")
        return

    table = Table(title=f"Last {limit} Routing Decisions", box=box.ROUNDED)
    table.add_column("Time", style="dim", width=20)
    table.add_column("Task Type")
    table.add_column("Difficulty")
    table.add_column("Model", style="bold")
    table.add_column("Tokens in/out")
    table.add_column("Cost")
    table.add_column("Reason", style="dim")

    for r in records:
        ts = r.timestamp[:19].replace("T", " ")
        table.add_row(
            ts,
            f"[cyan]{r.task_type}[/cyan]",
            r.difficulty,
            f"[green]{r.model_name}[/green]",
            f"{r.input_tokens} / {r.output_tokens}",
            f"${r.cost_usd:.5f}",
            r.routing_reason[:40],
        )

    console.print()
    console.print(table)

    total = tracker.total_spent()
    total_reqs = tracker.total_requests()
    console.print(
        f"  [dim]Total: {total_reqs} requests  |  ${total:.4f} spent all-time[/dim]\n"
    )


# ── byok spend ────────────────────────────────────────────────────────────────

@cli.command()
def spend():
    """Show monthly spend per model vs. your configured limits."""
    from byok.core.registry import ModelRegistry
    from byok.storage.spend_tracker import SpendTracker

    reg = ModelRegistry(CONFIG_PATH)
    tracker = SpendTracker(DB_PATH)
    monthly = tracker.get_all_monthly_spend()

    table = Table(title="Monthly Spend vs. Limits", box=box.ROUNDED)
    table.add_column("Model", style="bold")
    table.add_column("Spent")
    table.add_column("Limit")
    table.add_column("Remaining")
    table.add_column("Status")

    for m in reg.all_models():
        spent = monthly.get(m.name, 0.0)
        limit = m.spend_limit_monthly_usd

        if limit > 0:
            remaining = max(0.0, limit - spent)
            pct = spent / limit
            if pct >= 1.0:
                status = "[red]● LIMIT REACHED[/red]"
                rem_str = "[red]$0.00[/red]"
            elif pct >= 0.8:
                status = "[yellow]⚠ near limit[/yellow]"
                rem_str = f"[yellow]${remaining:.3f}[/yellow]"
            else:
                status = "[green]● ok[/green]"
                rem_str = f"${remaining:.3f}"
            limit_str = f"${limit:.2f}"
        else:
            status = "[dim]unlimited[/dim]"
            rem_str = "[dim]∞[/dim]"
            limit_str = "[dim]—[/dim]"

        table.add_row(
            m.name,
            f"${spent:.4f}",
            limit_str,
            rem_str,
            status,
        )

    console.print()
    console.print(table)
    grand_total = tracker.total_spent()
    console.print(f"  [dim]All-time total: ${grand_total:.4f}[/dim]\n")


if __name__ == "__main__":
    cli()
