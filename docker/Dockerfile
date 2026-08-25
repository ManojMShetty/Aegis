# Eval image for the Aegis-RAG lab.
#
# This image is built to be run on a Docker network declared `internal: true`
# (see docker-compose.yml). An internal network has no route to the WAN, which
# means a container on it cannot reach a hosted model endpoint EITHER - so this
# image is for a fully-mocked or local-model configuration, not for reproducing
# the recorded Groq runs. That limitation is stated in SECURITY.md rather than
# papered over here.
#
# Nothing in this file copies `.env` or accepts a credential build-arg. The
# startup tripwire (aegis.config.sandbox) refuses to run if one appears anyway.

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# uv from its published image: pinned by digest-able tag, no curl-to-shell.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependency layer first: the lockfile changes far less often than the source,
# so an edit to a security layer does not re-resolve the whole environment.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra evals

# Source and policy after the dependency layer.
COPY config ./config
COPY evals ./evals
COPY scripts ./scripts
COPY tests ./tests

ENV PATH="/app/.venv/bin:$PATH" \
    AEGIS_TOOLS=mock

# Unprivileged by default. `results/` is the one path a run writes to, and it is
# a mounted volume in compose; everything else can be read-only.
RUN useradd --create-home --uid 10001 aegis \
    && mkdir -p /app/results \
    && chown -R aegis:aegis /app
USER aegis

# `--help` by default: an image whose default command spends benchmark quota is
# a footgun. A real run passes its own flags.
ENTRYPOINT ["python", "-m", "evals.agentdojo.runner"]
CMD ["--help"]
