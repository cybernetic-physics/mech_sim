FROM mambaorg/micromamba:1.5.10

COPY --chown=$MAMBA_USER:$MAMBA_USER docker/solver/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

ENV OMP_NUM_THREADS=1
ENV OMP_PROC_BIND=false
ENV KMP_AFFINITY=disabled
ENV KMP_INIT_AT_FORK=FALSE

WORKDIR /workspace
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace
RUN micromamba run -n base python -m pip install -e ".[dev,traces]"

ENTRYPOINT ["micromamba", "run", "-n", "base", "python", "-m", "mech_bench"]
CMD ["oracle-smoke", "--require-real"]
