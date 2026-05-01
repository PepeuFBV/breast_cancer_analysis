from __future__ import annotations

import sys
import time
from pathlib import Path

from pipeline.config import load_experiment_config
from pipeline.experiments import ExperimentStateStore, launch_background_runner

try:
    import streamlit as st
except ImportError as error:  # pragma: no cover - executed only when streamlit is missing
    raise SystemExit("Streamlit is required for the dashboard. " "Install dependencies with `python -m pip install -r requirements.txt`.") from error


PROJECT_ROOT = Path(__file__).resolve().parent


def _load_store(config_path: str, artifacts_dir: str) -> ExperimentStateStore:
    experiment_config = load_experiment_config(config_path or None)
    project_paths = experiment_config.resolve_project_paths(artifacts_dir=(artifacts_dir or None)).ensure_artifact_dirs()
    return ExperimentStateStore(project_paths)


def _build_launch_args(
    *,
    config_path: str,
    artifacts_dir: str,
    rerun_failed: bool,
    rerun_completed: bool,
    limit: int,
) -> list[str]:
    forwarded_args: list[str] = []
    if config_path:
        forwarded_args.extend(["--config", config_path])
    if artifacts_dir:
        forwarded_args.extend(["--artifacts-dir", artifacts_dir])
    if rerun_failed:
        forwarded_args.append("--rerun-failed")
    if rerun_completed:
        forwarded_args.append("--rerun-completed")
    if limit > 0:
        forwarded_args.extend(["--limit", str(limit)])
    return forwarded_args


def _render_header() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(
                    circle at top left,
                    rgba(232, 239, 229, 0.9),
                    transparent 35%
                ),
                linear-gradient(180deg, #f7f3eb 0%, #f1ebdf 100%);
        }
        .runner-card {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(62, 97, 77, 0.18);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            box-shadow: 0 12px 32px rgba(65, 53, 34, 0.08);
        }
        .runner-kicker {
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #6f7f70;
            font-size: 0.78rem;
            margin-bottom: 0.25rem;
        }
        .runner-title {
            font-size: 2rem;
            color: #243428;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }
        .runner-copy {
            color: #455247;
            font-size: 1rem;
            line-height: 1.5;
        }
        </style>
        <div class="runner-card">
            <div class="runner-kicker">Iterative Execution</div>
            <div class="runner-title">Experiment Runner Dashboard</div>
            <div class="runner-copy">
                Start, stop and resume experiment batches safely. Progress is read from
                persisted state, so the page can refresh at any time without
                losing control.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Experiment Runner Dashboard",
        layout="wide",
    )
    _render_header()

    with st.sidebar:
        st.subheader("Runner Settings")
        config_path = st.text_input(
            "Config path",
            value="configs/experiment.default.json",
            help="Experiment config used to build the execution queue.",
        )
        artifacts_dir = st.text_input(
            "Artifacts dir",
            value="",
            help="Leave empty to use the artifacts path from the config file.",
        )
        rerun_failed = st.checkbox(
            "Rerun failed",
            value=False,
            help=("Failed tasks stay paused by default until you explicitly " "ask to rerun them."),
        )
        rerun_completed = st.checkbox(
            "Rerun completed",
            value=False,
            help="Use with care: completed tasks are skipped by default.",
        )
        limit = st.number_input(
            "Run limit",
            min_value=0,
            value=0,
            step=1,
            help="Optional cap for how many runnable tasks this launch should execute.",
        )
        auto_refresh = st.toggle("Auto refresh", value=True)
        refresh_seconds = st.slider(
            "Refresh interval (seconds)",
            min_value=2,
            max_value=15,
            value=5,
        )

    try:
        store = _load_store(config_path, artifacts_dir)
    except Exception as error:
        st.error(f"Could not load experiment configuration: {error}")
        return

    status = store.summarize()
    state = store.load_state(config_path=Path(config_path))
    counts = status["counts"]
    current_task = status["current_task"]

    action_col, stop_col, refresh_col = st.columns([1.2, 1.0, 0.9])
    if action_col.button("Run", use_container_width=True, type="primary"):
        if store.has_active_run():
            st.warning("A runner process is already active for this artifacts directory.")
        else:
            launch_args = _build_launch_args(
                config_path=config_path,
                artifacts_dir=artifacts_dir,
                rerun_failed=rerun_failed,
                rerun_completed=rerun_completed,
                limit=int(limit),
            )
            process = launch_background_runner(
                script_path=PROJECT_ROOT / "run_experiments.py",
                forwarded_args=launch_args,
                cwd=PROJECT_ROOT,
                logs_dir=store.project_paths.experiment_logs_dir,
            )
            config_source = load_experiment_config(config_path or None).source_path
            command = [sys.executable, str(PROJECT_ROOT / "run_experiments.py"), "run", *launch_args]
            store.write_pid_record(
                config_path=config_source,
                command=command,
                pid=process.process.pid,
                stdout_log_path=process.stdout_path,
                stderr_log_path=process.stderr_path,
            )
            st.success(f"Runner started in background with pid={process.process.pid}.")
            st.rerun()

    if stop_col.button("Stop", use_container_width=True):
        stop_path = store.request_stop(reason="dashboard")
        st.info("Stop requested. The runner will finish the current experiment " "before stopping.")
        st.caption(f"Stop flag: {stop_path}")
        st.rerun()

    if refresh_col.button("Refresh", use_container_width=True):
        st.rerun()

    metric_cols = st.columns(5)
    metric_cols[0].metric("Overall", str(status["overall_status"]).replace("_", " "))
    metric_cols[1].metric("Total", int(status["total"]))
    metric_cols[2].metric("Completed", int(counts["completed"]))
    metric_cols[3].metric("Failed", int(counts["failed"]))
    metric_cols[4].metric("Pending", int(counts["pending"]) + int(counts["stopped"]))

    details_col, paths_col = st.columns([1.3, 1.0])
    with details_col:
        st.subheader("Current Execution")
        if current_task:
            st.markdown(f"""
                **Task ID:** `{current_task['id']}`  
                **Model:** `{current_task['model_name']}`  
                **Preprocessing:** `{current_task['preproc_id']}`  
                **Parameters:** `{current_task['param_display']}`  
                **Started at:** `{current_task['started_at']}`
                """)
        else:
            st.write("No experiment is actively running right now.")

        if status["stop_requested"]:
            st.warning("A graceful stop has been requested and is waiting for a safe point.")

    with paths_col:
        st.subheader("Saved Outputs")
        st.markdown(f"""
            **State:** `{status['state_path']}`  
            **Summary:** `{status['summary_path']}`  
            **History:** `{status['history_dir']}`  
            **Predictions:** `{status['predictions_dir']}`  
            **Log:** `{status['log_path']}`
            """)

    with st.expander("Queue Snapshot", expanded=True):
        task_rows = []
        for task in state["tasks"]:
            task_rows.append(
                {
                    "id": task["id"],
                    "status": task["status"],
                    "model_name": task["model_name"],
                    "preproc_id": task["preproc_id"],
                    "param_display": task["param_display"],
                    "attempts": task.get("attempts", 0),
                    "updated_at": task.get("updated_at"),
                    "error_summary": task.get("error_summary"),
                }
            )
        if task_rows:
            st.dataframe(task_rows, use_container_width=True, hide_index=True)
        else:
            st.write("The queue has not been materialized yet. Click Run to create it.")

    with st.expander("Reset Controls"):
        purge_results = st.checkbox(
            "Also remove saved history and predictions",
            value=False,
        )
        confirm_reset = st.checkbox("Confirm reset", value=False)
        if st.button("Reset runner state", use_container_width=True):
            if not confirm_reset:
                st.error("Confirm reset before removing runner state.")
            else:
                store.reset(purge_results=purge_results)
                st.success("Runner state was reset.")
                st.rerun()

    if auto_refresh and status["overall_status"] in {"running", "stopping"}:
        time.sleep(refresh_seconds)
        st.rerun()


if __name__ == "__main__":
    main()
