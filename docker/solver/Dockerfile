FROM mambaorg/micromamba:1.5.10

COPY --chown=$MAMBA_USER:$MAMBA_USER docker/solver/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

ARG CYCLOID_GEARBOX_REF=0d608077cf4a68cd736cd3dd585e17d5c6d00aea
ENV MECH_BENCH_CYCLOID_GEARBOX_REF=$CYCLOID_GEARBOX_REF
ENV MECH_BENCH_CYCLOID_GEARBOX_PATH=/home/$MAMBA_USER/CycloidGearBox
RUN micromamba create -y -n freecad -c conda-forge python=3.12 freecad git \
    && micromamba run -n freecad git init "$MECH_BENCH_CYCLOID_GEARBOX_PATH" \
    && micromamba run -n freecad git -C "$MECH_BENCH_CYCLOID_GEARBOX_PATH" remote add origin https://github.com/iplayfast/CycloidGearBox.git \
    && micromamba run -n freecad git -C "$MECH_BENCH_CYCLOID_GEARBOX_PATH" fetch --depth 1 origin "$CYCLOID_GEARBOX_REF" \
    && micromamba run -n freecad git -C "$MECH_BENCH_CYCLOID_GEARBOX_PATH" checkout FETCH_HEAD \
    && mkdir -p "/home/$MAMBA_USER/bin" \
    && printf '%s\n' '#!/usr/bin/env bash' \
        'set -euo pipefail' \
        'if micromamba run -n freecad bash -lc "command -v FreeCADCmd >/dev/null 2>&1"; then' \
        '  exec micromamba run -n freecad FreeCADCmd "$@"' \
        'fi' \
        'exec micromamba run -n freecad freecadcmd "$@"' \
        > "/home/$MAMBA_USER/bin/mech-freecadcmd" \
    && chmod +x "/home/$MAMBA_USER/bin/mech-freecadcmd" \
    && micromamba clean --all --yes
ENV MECH_BENCH_FREECAD_CMD=/home/$MAMBA_USER/bin/mech-freecadcmd

ENV OMP_NUM_THREADS=1
ENV OMP_PROC_BIND=false
ENV KMP_AFFINITY=disabled
ENV KMP_INIT_AT_FORK=FALSE

WORKDIR /workspace
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace
RUN micromamba run -n base python -m pip install -e ".[dev,traces]"

ENTRYPOINT ["micromamba", "run", "-n", "base", "python", "-m", "mech_bench"]
CMD ["oracle-smoke", "--require-real"]
