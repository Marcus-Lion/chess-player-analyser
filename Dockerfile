# ---------------------------------------------------------------------------
# Stage 1: compile the native self-play engine (engine/) into a wheel.
# Kept in a separate stage so the Rust toolchain never lands in the runtime
# image. On Linux the default toolchain uses the system linker, so unlike the
# local Windows build there's no MSVC/GNU toolchain juggling.
# ---------------------------------------------------------------------------
FROM python:3.14-slim AS engine-builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain (minimal profile: rustc + cargo, default linux-gnu host).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --no-cache-dir maturin

WORKDIR /build
COPY engine ./engine
# Plain linux wheel (no manylinux repair): the runtime image below shares this
# exact base, so the wheel is ABI-compatible without auditwheel.
RUN maturin build --release --compatibility linux \
    -i python3.14 -m engine/Cargo.toml --out /wheels

# ---------------------------------------------------------------------------
# Stage 2: runtime image (pure Python + the prebuilt native engine wheel).
# ---------------------------------------------------------------------------
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Install the required native chess_engine wheel into the app venv.
COPY --from=engine-builder /wheels /wheels
RUN uv pip install --python /app/.venv/bin/python /wheels/*.whl

COPY app ./app

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8080
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
