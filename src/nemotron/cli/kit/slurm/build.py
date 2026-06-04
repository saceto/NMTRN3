# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build command - build a recipe stage's Dockerfile into a .sqsh on a Slurm cluster.

The build counterpart of ``nemotron kit slurm squash``: ``squash`` imports an
existing image; ``build`` adds a ``podman build`` of a recipe-owned Dockerfile in
front of the same enroot import. Slurm-only and explicit about it (hence the
``kit slurm`` home); driven by an ``env.toml`` profile via nemo_runspec.

Code transport: unlike ``squash``, ``build`` needs the repo (Dockerfile, build
context, and the shared ``build_container.sh``) on the cluster, so it submits via
a nemo-run ``SlurmExecutor`` with a ``CodePackager``. The packager rsyncs the
local checkout to ``/nemo_run/code`` on the cluster — no manual sync, no
``--repo-root``. Commit your changes first; the packager ships the git tree.

The build logic itself lives in the shared, recipe-agnostic
``build_container.sh`` next to this file; this command only resolves inputs and
submits it as the executor payload.

Usage:
    nemotron kit slurm build dlw --recipe ultra3 --stage pretrain \
        --build-arg MEGATRON_BRIDGE_BRANCH=dev-0603
    nemotron kit slurm build dlw --dockerfile src/.../Dockerfile --output my.sqsh
"""

from __future__ import annotations

import shlex
import tempfile
from pathlib import Path, PurePosixPath

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nemo_runspec.env import load_env_profile
from nemo_runspec.squash import resolve_build_cache_dir, resolve_build_image, resolve_build_partition, resolve_build_time

console = Console()

# Build-time podman/enroot container (pyxis launches it on the compute node).
DEFAULT_BUILD_IMAGE = "docker://quay.io#podman/stable:v5.3"
# Where CodePackager mounts the rsynced repo inside the build container.
REMOTE_CODE_ROOT = "/nemo_run/code"

# Stage registry — consolidates the per-recipe STAGES dicts the retired
# ultra3/omni3 build.py dispatchers used to own. Maps recipe -> stage alias ->
# (stage_dir, build-time image tag, output .sqsh basename).
RECIPES: dict[str, dict[str, tuple[str, str, str]]] = {
    "ultra3": {
        "pretrain": ("stage0_pretrain", "nemotron/ultra3-pretrain:latest", "ultra3-pretrain.sqsh"),
        "sft": ("stage1_sft", "nemotron/ultra3-sft:latest", "ultra3-sft.sqsh"),
    },
    "omni3": {
        "sft": ("stage0_sft", "nemotron/omni3-sft:latest", "omni3-sft.sqsh"),
        "rl": ("stage1_rl", "nemotron/omni3-rl:latest", "omni3-rl.sqsh"),
    },
}


def _local_repo_root() -> Path:
    # repo root is five levels up: .../<root>/src/nemotron/cli/kit/slurm/build.py
    return Path(__file__).resolve().parents[5]


def build(
    profile: str = typer.Argument(..., help="Env profile name from env.toml (e.g., 'dlw')."),
    recipe: str | None = typer.Option(None, "--recipe", help="Recipe to build (e.g. 'ultra3', 'omni3')."),
    stage: str | None = typer.Option(None, "--stage", help="Stage alias within the recipe (e.g. 'pretrain', 'sft')."),
    dockerfile: str | None = typer.Option(None, "--dockerfile", help="Generic: repo-relative path to a Dockerfile (overrides --recipe/--stage)."),
    output: str | None = typer.Option(None, "--output", help="Generic: output .sqsh basename (with --dockerfile)."),
    build_arg: list[str] = typer.Option([], "--build-arg", help="Docker build-arg, repeatable (e.g. MEGATRON_BRIDGE_BRANCH=dev-0603)."),
    dry_run: bool = typer.Option(False, "-d", "--dry-run", help="Show the resolved plan without submitting."),
    detach: bool = typer.Option(True, "--detach/--attach", help="Detach after submit (default) or stream the build."),
) -> None:
    """Build a recipe stage's Dockerfile into an enroot .sqsh on a Slurm cluster.

    Examples:
        nemotron kit slurm build dlw --recipe ultra3 --stage pretrain \\
            --build-arg MEGATRON_BRIDGE_BRANCH=dev-0603
        nemotron kit slurm build dlw --dockerfile src/nemotron/recipes/ultra3/stage0_pretrain/Dockerfile \\
            --output ultra3-pretrain.sqsh
    """
    try:
        env_config = load_env_profile(profile)
    except (FileNotFoundError, KeyError) as e:
        console.print(f"[red bold]Error:[/red bold] {e}")
        raise typer.Exit(1)

    host = env_config.get("host")
    user = env_config.get("user")
    remote_job_dir = env_config.get("remote_job_dir")
    if not host or not user:
        console.print(f"[red bold]Error:[/red bold] Profile '{profile}' missing host or user for SSH")
        raise typer.Exit(1)
    if not remote_job_dir:
        console.print(f"[red bold]Error:[/red bold] Profile '{profile}' missing remote_job_dir")
        raise typer.Exit(1)

    local_root = _local_repo_root()

    # Resolve build inputs. Paths come in two forms: a LOCAL path (CodePackager
    # anchor) and the matching IN-CONTAINER path under REMOTE_CODE_ROOT.
    if dockerfile:
        rel = dockerfile if not Path(dockerfile).is_absolute() else str(Path(dockerfile).relative_to(local_root))
        local_dockerfile = local_root / rel
        df_in = f"{REMOTE_CODE_ROOT}/{rel}"
        sqsh_name = output or (PurePosixPath(rel).parent.name + ".sqsh")
        image_tag = f"nemotron/{PurePosixPath(sqsh_name).stem}:latest"
        manifest_key = PurePosixPath(sqsh_name).stem
    else:
        if not recipe or not stage or recipe not in RECIPES or stage not in RECIPES.get(recipe, {}):
            console.print("[red bold]Error:[/red bold] provide a known --recipe and --stage, or --dockerfile/--output.")
            for r, stages in RECIPES.items():
                console.print(f"  {r}: {', '.join(stages)}")
            raise typer.Exit(1)
        stage_dir, image_tag, sqsh_name = RECIPES[recipe][stage]
        rel = f"src/nemotron/recipes/{recipe}/{stage_dir}/Dockerfile"
        local_dockerfile = local_root / rel
        df_in = f"{REMOTE_CODE_ROOT}/{rel}"
        manifest_key = f"{recipe}-{stage_dir}"

    if not local_dockerfile.is_file():
        console.print(f"[red bold]Error:[/red bold] Dockerfile not found locally: {local_dockerfile}")
        raise typer.Exit(1)

    context_in = str(PurePosixPath(df_in).parent)
    inner_in = f"{REMOTE_CODE_ROOT}/src/nemotron/cli/kit/slurm/build_container.sh"

    # Cluster cache (Lustre): mounted at its host path; .sqsh + manifest land here.
    build_cache_dir = str(resolve_build_cache_dir(env_config, Path(remote_job_dir) / "nemotron-cache"))
    containers_dir = str(PurePosixPath(build_cache_dir) / "containers")
    sqsh = str(PurePosixPath(containers_dir) / sqsh_name)
    manifest = str(PurePosixPath(containers_dir) / "manifest.yaml")
    build_args = " ".join(f"--build-arg {a}" for a in build_arg)
    build_image = resolve_build_image(env_config, DEFAULT_BUILD_IMAGE)
    partition = resolve_build_partition(env_config)
    walltime = resolve_build_time(env_config, "02:00:00")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")
    table.add_row("Profile", f"[cyan]{profile}[/cyan]")
    table.add_row("Host", f"{user}@{host}")
    table.add_row("Code transport", f"CodePackager: {local_root} -> {REMOTE_CODE_ROOT}")
    table.add_row("Dockerfile", df_in)
    table.add_row("Build image", build_image)
    table.add_row("Partition / time", f"{partition} / {walltime}")
    table.add_row("Image tag", image_tag)
    table.add_row("Output", sqsh)
    table.add_row("Manifest", f"{manifest} (key: {manifest_key})")
    table.add_row("Build args", build_args or "<none>")
    console.print(Panel(table, title="[bold]Build Configuration[/bold]", expand=False))
    console.print()

    # Inner payload: export explicit inputs, then run the shared build script.
    inner_env = {
        "DOCKERFILE": df_in,
        "CONTEXT": context_in,
        "IMAGE_TAG": image_tag,
        "SQSH": sqsh,
        "BUILD_CACHE_DIR": build_cache_dir,
        "MANIFEST": manifest,
        "MANIFEST_KEY": manifest_key,
        "BUILD_ARGS": build_args,
    }
    inline = "; ".join(f"export {k}={shlex.quote(v)}" for k, v in inner_env.items())
    inline += f"; bash {shlex.quote(inner_in)}"

    if dry_run:
        console.print("[yellow]Dry-run mode - no submission.[/yellow]")
        console.print("Payload (inside pyxis podman container):")
        console.print(f"  [dim]{inline}[/dim]")
        return

    try:
        import nemo_run as run
    except ImportError:
        console.print("[red bold]Error:[/red bold] nemo-run is required for build")
        raise typer.Exit(1)

    from nemo_runspec.execution import materialize_podman_auth_from_enroot

    tunnel = run.SSHTunnel(host=host, user=user, job_dir=remote_job_dir)
    tunnel.connect()

    # Mounts: the build cache (Lustre) at its host path, plus best-effort podman
    # auth bridged from the user's enroot credentials so FROM nvcr.io/... pulls.
    mounts = [f"{build_cache_dir}:{build_cache_dir}"]
    try:
        auth_path = materialize_podman_auth_from_enroot(tunnel, f"{build_cache_dir}/.auth")
        if auth_path:
            mounts.append(f"{auth_path}:/root/.config/containers/auth.json:ro")
    except Exception as exc:  # best-effort; rely on cluster creds / cached base otherwise
        console.print(f"[dim]podman auth bridge skipped: {exc}[/dim]")

    tunnel.run(f"mkdir -p {containers_dir}", hide=True)

    # CodePackager ships the local checkout to REMOTE_CODE_ROOT. A placeholder
    # train config keeps the packager happy; the Dockerfile is the real anchor.
    tmp_cfg = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    tmp_cfg.write("{}\n")
    tmp_cfg.close()

    try:
        from nemo_runspec.packaging import CodePackager

        packager = CodePackager(script_path=str(local_dockerfile), train_path=tmp_cfg.name)
        executor = run.SlurmExecutor(
            account=env_config.get("account"),
            partition=partition,
            nodes=1,
            ntasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=env_config.get("build_cpus", 16),
            time=walltime,
            container_image=build_image,
            container_mounts=mounts,
            # Run the build container as root so the in-container `enroot import`
            # can unpack root-owned image files (rootless unpack fails with
            # Permission denied and hangs). The host has no podman and enroot
            # reads no portable archive, so the import must stay in-container via
            # podman:// — hence root here rather than a host-side import.
            srun_args=["--container-remap-root", "--container-writable"],
            tunnel=tunnel,
            packager=packager,
            mem=env_config.get("build_mem") or env_config.get("mem"),
            launcher=None,
        )
        task = run.Script(inline=inline, entrypoint="bash")
        name = f"kit-build-{manifest_key}"
        console.print(f"[bold]Submitting build '{name}' to {host} ({partition})...[/bold]")
        console.print(f"  -> {sqsh}")
        with run.Experiment(name) as exp:
            exp.add(task, executor=executor, name=name)
            exp.run(detach=detach)
        console.print(
            Panel(
                f"[green]Build submitted.[/green]\nOn success: {sqsh}\n\n"
                f"[dim]run with: ... run.env.container_image={sqsh}[/dim]",
                title="[bold green]Submitted[/bold green]",
                border_style="green",
                expand=False,
            )
        )
    finally:
        Path(tmp_cfg.name).unlink(missing_ok=True)
