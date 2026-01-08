import os
import time
import logging
import warnings
import pyEXP

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI = None
    MPI_AVAILABLE = False

def compute_exp_coefs(
    particle_data,
    basis,
    component,
    coefs_file,
    unit_system,
    return_coefs=False):
    """
    Compute EXP expansion coefficients in serial (non-MPI).

    This routine performs coefficient construction and HDF5 output on a
    single process. It is intended for environments where MPI is not
    available or parallel execution is not desired.

    Parameters
    ----------
    particle_data : dict
        Dictionary containing particle information with the following keys:

        - ``"mass"`` : array_like
            Particle masses.
        - ``"pos"`` : array_like, shape (N, 3)
            Particle positions.
        - ``"snapshot_time"`` : float
            Physical time of the snapshot.

    basis : pyEXP.basis.Basis
        EXP basis object.

    component : str
        Name of the EXP component under which coefficients are stored.

    coefs_file : str
        Path to the output HDF5 coefficient file. If the file exists,
        coefficients are appended; otherwise, a new file is created.

    unit_system : object
        Unit system compatible with ``pyEXP.coefs.Coefs.setUnits``.

    Returns
    -------
    info : dict
        Dictionary containing diagnostic information:

        - ``"snapshot_time"`` : float
        - ``"nparticles"`` : int
        - ``"elapsed_time"`` : float
    """

    start_time = time.time()

    coef = basis.createFromArray(
        particle_data["mass"],
        particle_data["pos"],
        time=particle_data["snapshot_time"],
    )

    coefs = pyEXP.coefs.Coefs.makecoefs(coef, name=component)
    coefs.add(coef)
    coefs.setUnits(unit_system)

    if os.path.exists(coefs_file):
        coefs.ExtendH5Coefs(coefs_file)
    else:
        coefs.WriteH5Coefs(coefs_file)

    elapsed_time = time.time() - start_time

    snap_time = particle_data["snapshot_time"]
    nparticles = len(particle_data["mass"])

    logging.info(
        "Coefficients for snapshot t=%s with nparticles=%d "
        "computed in %.2f s (serial)",
        snap_time,
        nparticles,
        elapsed_time,
    )

    if return_coefs is True:
        return coefs
    else:
        return {
            "snapshot_time": snap_time,
            "nparticles": nparticles,
            "elapsed_time": elapsed_time,
            }



def compute_exp_coefs_parallel(
    particle_data,
    basis,
    component,
    coefs_file,
    unit_system,
    *,
    comm=MPI.COMM_WORLD,
):
    """
    Compute EXP expansion coefficients in parallel using MPI.

    This routine performs collective coefficient construction across all MPI
    ranks via the underlying C++ EXP implementation. Disk I/O is restricted
    to rank 0 to ensure HDF5 safety on systems without parallel HDF5 support.

    All MPI ranks **must** call this function collectively.

    Parameters
    ----------
    particle_data : dict
        Dictionary containing particle information with the following keys:

        - ``"mass"`` : array_like
            Particle masses local to each MPI rank.
        - ``"pos"`` : array_like, shape (N, 3)
            Particle positions local to each MPI rank.
        - ``"snapshot_time"`` : float
            Physical time of the snapshot.

    basis : pyEXP.basis.Basis
        EXP basis object. The basis must be constructed consistently across
        all MPI ranks and support MPI-parallel coefficient construction.

    component : str
        Name of the EXP component (e.g., ``"halo"``, ``"disk"``) under which
        coefficients will be stored in the output file.

    coefs_file : str
        Path to the HDF5 coefficient file. If the file exists, coefficients
        are appended; otherwise, a new file is created. Only rank 0 performs
        file I/O.

    unit_system : object
        Unit system to associate with the coefficient container. The object
        must be compatible with ``pyEXP.coefs.Coefs.setUnits``.

    comm : mpi4py.MPI.Comm, optional
        MPI communicator over which the computation is collective.
        Defaults to ``MPI.COMM_WORLD``.

    Returns
    -------
    info : dict or None
        Dictionary containing diagnostic information, returned **only on
        rank 0**. Other ranks return ``None``.

        The dictionary contains:

        - ``"snapshot_time"`` : float
        - ``"nparticles"`` : int
        - ``"elapsed_time"`` : float

    Notes
    -----
    - This function assumes MPI has already been initialized by the calling
      application (e.g., via ``mpirun`` or ``srun``).
    - All ranks must invoke this function simultaneously.
    - HDF5 output is serialized on rank 0 to avoid file corruption.
    - No MPI finalization is performed inside this function.

    Examples
    --------
    >>> info = compute_exp_coefs_parallel(
    ...     particle_data,
    ...     basis,
    ...     component="halo",
    ...     coefs_file="halo_coefs.h5",
    ...     unit_system=units,
    ... )
    >>> if info is not None:
    ...     print(info["elapsed_time"])
    """

    # --------------------------------------------------
    # MPI setup
    # --------------------------------------------------
    my_rank = comm.Get_rank()

    # --------------------------------------------------
    # Synchronize and start timing
    # --------------------------------------------------
    comm.Barrier()
    start_time = time.time()

    # --------------------------------------------------
    # Collective coefficient construction
    # --------------------------------------------------
    coef = basis.createFromArray(
        particle_data["mass"],
        particle_data["pos"],
        time=particle_data["snapshot_time"],
    )

    if my_rank == 0:
        logging.info("EXP coefficient object created")

    # --------------------------------------------------
    # Coefficient container logic
    # --------------------------------------------------
    coefs = pyEXP.coefs.Coefs.makecoefs(coef, name=component)
    coefs.add(coef)

    if my_rank == 0:
        logging.info("Coefficient added to container")

    # --------------------------------------------------
    # MPI-safe HDF5 output (rank 0 only)
    # --------------------------------------------------
    if my_rank == 0:
        coefs.setUnits(unit_system)

        if os.path.exists(coefs_file):
            coefs.ExtendH5Coefs(coefs_file)
            logging.info("Extended HDF5 coefficient file: %s", coefs_file)
        else:
            coefs.WriteH5Coefs(coefs_file)
            logging.info("Created HDF5 coefficient file: %s", coefs_file)

    # --------------------------------------------------
    # Final synchronization and timing
    # --------------------------------------------------
    comm.Barrier()
    elapsed_time = time.time() - start_time

    # --------------------------------------------------
    # Diagnostics (rank 0 only)
    # --------------------------------------------------
    if my_rank == 0:
        snap_time = particle_data["snapshot_time"]
        nparticles = len(particle_data["mass"])

        logging.info(
            "Coefficients for snapshot t=%s with nparticles=%d "
            "computed in %.2f s",
            snap_time,
            nparticles,
            elapsed_time,
        )

        return {
            "snapshot_time": snap_time,
            "nparticles": nparticles,
            "elapsed_time": elapsed_time,
        }

    return None


def compute_expansion_coefficients(
    particle_data,
    basis,
    component,
    coefs_file,
    unit_system,
    *,
    parallel=False,
    comm=None,
    return_coefs=False,
):
    """
    Compute EXP expansion coefficients in serial or parallel mode.

    This function dispatches to either a serial or MPI-parallel
    implementation depending on user preference and MPI availability.

    Parameters
    ----------
    particle_data : dict
        Particle data dictionary (see ``compute_exp_coefs``).

    basis : pyEXP.basis.Basis
        EXP basis object.

    component : str
        Name of the EXP component.

    coefs_file : str
        Path to the HDF5 coefficient file.

    unit_system : object
        Unit system compatible with ``pyEXP.coefs.Coefs.setUnits``.

    parallel : bool, optional
        If ``True``, attempt MPI-parallel execution.
        Defaults to ``False``.

    comm : mpi4py.MPI.Comm, optional
        MPI communicator to use for parallel execution.
        Defaults to ``MPI.COMM_WORLD`` if not provided.

    Returns
    -------
    info : dict or None
        Diagnostic information. In parallel mode, this is returned
        only on rank 0; other ranks return ``None``.

    Warnings
    --------
    If ``parallel=True`` but ``mpi4py`` is not available, a warning is
    issued and the computation falls back to serial execution.
    """

    if parallel:
        if not MPI_AVAILABLE:
            warnings.warn(
                "Parallel computation requested, but mpi4py is not "
                "available. Falling back to serial execution.",
                RuntimeWarning,
            )
            return compute_exp_coefs(
                particle_data,
                basis,
                component,
                coefs_file,
                unit_system,
            )

        if comm is None:
            comm = MPI.COMM_WORLD

        return compute_exp_coefs_parallel(
            particle_data,
            basis,
            component,
            coefs_file,
            unit_system,
            comm=comm,
        )

    return compute_exp_coefs(
        particle_data,
        basis,
        component,
        coefs_file,
        unit_system,
        return_coefs,
    )
