# Changelog

## [0.2.2] – 2026-01-05

### Added
- Docstrings in the `basis` module.
- Support to build cylindrical basis functions.
- Updated unit tests for `basis` to match refactored changes.
- Unit tests for `Profiles` were added.
- Support to compute EXP coefficients from particle positions and masses.
- MPI support for EXP coefficients computation.
- Unit tests for EXP coefficients computation.
- Unit tests performed on a Milky Way simulated galaxy.
- Matplotlib `exptools` style implemented in `utils`.
- Unit tests for the exptools matplotlib style.

### Changed
- Renamed `basis_build` folder to match pyEXP naming conventions.
- `basis` module was refactored.
- Moved `profiles.py` from `basis` to `utils`.

### Fixed
- Fixed a typo in `make_model`.
- Prevented `log10` of zero in density smoothing.

